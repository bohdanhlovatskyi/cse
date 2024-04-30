from typing import Optional, Dict
from collections import defaultdict

import torch

from .camera import PerspectiveCamera
from .losses import (
    joints_loss_func,
    cse_loss_func,
    rotational_loss,
    rotational_loss_embeddding,
    rotational_loss_dense
)
from .model import SMPLXModel, SMPLXModelParams
from .prior import MaxMixturePrior
from .utils import move_to, rel_change
from .posendf import PoseNDF, axis_angle_to_quaternion, quat_flip
from .smplify_config import  SmplifyConfig

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

SMPL_CONVEX_PARTS = {
    "left_leg": ["leftUpLeg", "leftLeg"],  # "leftFoot", "leftToeBase",
    "right_leg": ["rightUpLeg", "rightLeg"],  # "rightFoot", "rightToeBase",
    "torso": ["hips", "spine", "spine1", "spine2"],
    "left_hand": ["leftShoulder", "leftArm", "leftForeArm"],  # "leftHand", "leftHandIndex1"
    "right_hand": ["rightShoulder", "rightArm", "rightForeArm"],  # "rightHand", "rightHandIndex1",
    "head": ["neck", "head"]  # "eyeballs", "leftEye", "rightEye"
}


class SMPLify:
    def __init__(self,
                 glctx,
                 model_path,
                 batch_size=1,
                 joints_mapper=None,
                 smplx2smpl_transfer_path=None,
                 prior_config_path: str = None,
                 prior_ckpt_path: str = None,
                 predecessors_path: str = None,
                 dists_matrix_path: str = None,
                 smpl_segmentation_path: str = None,
                 smpl_vert_embeddings_path: str = None,
                 device=torch.device('cuda')):

        self.glctx = glctx
        # Store options
        self.device = device
        # GMM pose prior
        # self.pose_prior = MaxMixturePrior(prior_folder='priors',
        #                                   num_gaussians=8,
        #                                   dtype=torch.float32).to(device)

        if prior_ckpt_path is not None:
            self.pose_prior = PoseNDF(prior_config_path)
            self.pose_prior.load_state_dict(
                torch.load(prior_ckpt_path, map_location='cpu')['model_state_dict'])
            self.pose_prior.eval()
            self.pose_prior = self.pose_prior.to(device)
        else:
            self.pose_prior = None

        # Load SMPL model
        self.model = SMPLXModel(
            glctx, model_path, batch_size=batch_size, num_betas=10, num_expression_coeffs=10,
            num_hand_pca_comps=6, use_vposer=False, joints_mapper=joints_mapper,
            smplx2smpl_transfer_path=smplx2smpl_transfer_path,
            smpl_vert_embeddings_path=smpl_vert_embeddings_path
        ).to(device)

        self.predecessors = torch.tensor(np.load(predecessors_path), device=device)
        self.dists_matrix = None if dists_matrix_path is None \
            else torch.tensor(np.load(dists_matrix_path), device=device)

        with open(smpl_segmentation_path) as h:
            self.smpl_segmentation = json.load(h)

        parts = []
        for convex_part in SMPL_CONVEX_PARTS:
            part_vert_ids = []
            for subpart in SMPL_CONVEX_PARTS[convex_part]:
                part_vert_ids.extend(self.smpl_segmentation[subpart])
            part_vert_ids = torch.tensor(np.array(part_vert_ids), device=device)
            parts.append(part_vert_ids)
        self.convex_parts = parts

        self.batch_size = batch_size

    def _init_model(self, smplx_params_init: Optional[SMPLXModelParams]):
        # init parameters of the model
        if smplx_params_init is not None:
            self.model.set_betas(smplx_params_init.betas)
            self.model.set_expression(smplx_params_init.expression)
            self.model.set_pose_params(
                translation=smplx_params_init.translation,
                global_orient=smplx_params_init.global_orient,
                body_pose=smplx_params_init.body_pose,
                jaw_pose=smplx_params_init.jaw_pose,
                leye_pose=smplx_params_init.leye_pose,
                reye_pose=smplx_params_init.reye_pose,
                left_hand_pose=smplx_params_init.left_hand_pose,
                right_hand_pose=smplx_params_init.right_hand_pose,
            )

            self.model.set_camera_params(smplx_params_init.camera)

    def __call__(self, data, config: Optional[SmplifyConfig],) -> None:
        if config.debug_dir is not None:
            os.makedirs(os.path.join(config.debug_dir, "cse"), exist_ok=True)

        hist = defaultdict(list)

        assert config.num_steps >= 0
        assert config.step_size >= 0

        data = move_to(data, self.device)

        params_init = {k: v for k, v in data['params_init'].items()}
        params_init['camera'] = PerspectiveCamera(**params_init['camera']).to(self.device)
        smplx_params_init = SMPLXModelParams(**params_init)

        init_pose = smplx_params_init.body_pose.detach().clone()
        init_pose.requires_grad_(False)

        if config.init_pose_to_zero:
            smplx_params_init.body_pose = torch.zeros_like(smplx_params_init.body_pose)

        self._init_model(smplx_params_init)
        self.model.enable_grads(
            transl=config.transl_grad_enabled,
            global_orient=config.global_orient_grad_enabled,
            pose=config.pose_orient_grad_enabled,
            face_pose=config.face_pose_grad_enabled,
            camera=config.camera_grad_enabled,
            shape=config.shape_grad_enabled,
            expr=config.expr_grad_enabled,
        )

        if config.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=config.step_size)
        elif config.optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS(
                self.model.parameters(),
                lr=config.step_size,
                history_size=config.history_size,
                max_iter=config.max_iter,
                line_search_fn="strong_wolfe"
            )
        else:
            raise NotImplemeted()

        optimizer.param_groups[0]['lr'] = self.batch_size * config.step_size

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', threshold=config.scheduler_threshold, factor=config.scheduler_factor,
        )

        joints_gt = data['keypoints']
        joints_gt, joints_conf_gt = joints_gt[..., :2], joints_gt[..., 2:3]

        verts_valid_flatten = data["vert_ids"]
        verts_valid_flatten_mask = data["vert_mask"]  # padding mask
        verts_valid_coords = data["vert_coords"]
        cse_segm = data["body_parts"]
        pixel_embeddings = data["pixel_embeddings"]

        prev_loss = None
        for step in range(config.num_steps):

            def closure(return_all_losses: bool = False):
                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                model_output = self.model()

                if config.joints_loss_weight != 0:
                    joints_loss = joints_loss_func(joints_gt, model_output.joints2d, joints_conf_gt)
                    joints_loss *= config.joints_loss_weight
                else:
                    joints_loss = model_output.joints2d.new_tensor(0.)

                if config.cse_loss_weight != 0:
                    cse_loss = cse_loss_func(verts_valid_coords, verts_valid_flatten, verts_valid_flatten_mask,
                                             model_output.vertices2d_smpl, config)
                    cse_loss *= config.cse_loss_weight
                else:
                    cse_loss = model_output.joints2d.new_tensor(0.)

                if config.pose_loss_weight != 0:
                    pose_loss = (model_output.smplx_params.body_pose - init_pose).pow(2).mean()
                    pose_loss *= config.pose_loss_weight
                else:
                    pose_loss = model_output.joints2d.new_tensor(0.)

                if self.pose_prior is not None:
                    # TODO: Seems to be broken, setup from scratch and fix it
                    pose_quat = axis_angle_to_quaternion(model_output.smplx_params.body_pose)
                    pose_prior_loss = self.pose_prior(pose_quat, train=False)['dist_pred'].mean()
                    pose_prior_loss *= config.pose_prior_loss_weight
                else:
                    pose_prior_loss = torch.zeros_like(pose_loss)

                if config.embedding_loss_weight != 0:
                    embedding_loss = torch.zeros(1, device=cse_segm.device)
                    for convex_part in self.convex_parts:
                        embedding_loss += rotational_loss_embeddding(
                            gt_sil=cse_segm,
                            gt_vert_ids=verts_valid_flatten,
                            gt_pixel_embeddings=pixel_embeddings,
                            convex_part_ids=convex_part,
                            cur_raster=model_output.smpl_frag,
                            cur_attr=model_output.attr
                        )
                    embedding_loss *= config.embedding_loss_weight
                else:
                    embedding_loss = model_output.joints2d.new_tensor(0.)

                if config.rotation_loss_weight != 0:
                    rotation_loss = torch.zeros(1, device=cse_segm.device)
                    for convex_part in self.convex_parts:
                        rotation_loss += rotational_loss_dense(
                            gt_sil=cse_segm,
                            gt_vert_ids=verts_valid_flatten,
                            vertices=model_output.vertices3d_smpl,
                            cur_raster=model_output.smpl_frag,
                            smpl_faces=self.model.smpl_faces[None],
                            convex_part_ids=convex_part,
                            predecessors=self.predecessors
                        )
                    rotation_loss *= config.rotation_loss_weight
                else:
                    rotation_loss = model_output.joints2d.new_tensor(0.)

                loss = joints_loss + cse_loss + pose_loss + pose_prior_loss + rotation_loss + embedding_loss

                if loss.requires_grad:
                    loss.backward()

                if not return_all_losses:
                    return loss
                else:
                    return loss, cse_loss, joints_loss, pose_loss, pose_prior_loss, rotation_loss, embedding_loss, model_output

            optimizer.step(closure)

            loss, cse_loss, joints_loss, pose_loss, pose_prior_loss, rotation_loss, embedding_loss, model_output \
                = closure(return_all_losses=True)

            scheduler.step(loss)
            lr = optimizer.param_groups[0]['lr']

            max_grad_changes = {name: torch.abs(var.grad.view(-1).max()).item() \
                                for (name, var) in self.model.named_parameters() if var.grad is not None}

            mean_grad_changes = {name: torch.abs(var.grad.view(-1).mean()).item() \
                                 for (name, var) in self.model.named_parameters() if var.grad is not None}

            print(f'It: {step}, '
                  f'J: {joints_loss.item():.3}, '
                  f'CSE: {cse_loss.item():.3}, '
                  f'Pose Loss: {pose_loss.item():.3}, '
                  f'Pose Prior Loss: {pose_prior_loss.item():.3}, '
                  f'Rotation Loss: {rotation_loss.item():.3}, '
                  f'Embedding Loss: {embedding_loss.item():.3}, '
                  f'LR: {lr:.3}, '
                  f'Total: {loss.item():.3}')

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if config.save_history:
                hist["J"].append(joints_loss.item())
                hist["CSE"].append(cse_loss.item())
                hist["PL"].append(pose_loss.item())
                hist["PP"].append(pose_prior_loss.item())
                hist["RotL"].append(rotation_loss.item())
                hist["EL"].append(embedding_loss.item())
                hist["MaxGC"].append(max_grad_changes)
                hist["MeanGC"].append(mean_grad_changes)
                hist["LR"].append(lr)

                if config.debug_dir is not None:
                    vis_cse_loss(
                        os.path.join(config.debug_dir, "cse", f"{step}.png".zfill(9)),
                        model_output.vertices2d_smpl.detach().cpu()[0],
                        data
                    )

            if config.early_stopping and (step > 0 and prev_loss is not None and cofig.ftol > 0):
                loss_rel_change = rel_change(prev_loss, loss.item())

                if 0 < loss_rel_change <= config.ftol:
                    print(f'Relative loss change is below threshold {loss_rel_change}/{ftol}, stopping!')
                    break

            if config.early_stopping and all([entry < config.gtol for entry in max_grad_changes.values()]):
                print('Grad update is below threshold, stopping!')
                break

            prev_loss = loss.item()

        return self.model.get_params(), hist


def plot2d_points(image, points2d, color=None, r=2, t=2):
    if color is None:
        color = (255, 0, 0)
    image = image.copy()
    for i, (x, y) in enumerate(points2d):
        cv2.circle(image, (int(x), int(y)), radius=r, color=color, thickness=t)
    return image


def vis_cse_loss(save_to: str, proj_vert: torch.Tensor, x: Dict) -> None:
    yc, xc = torch.where(x["cse"].segmentation > 0)
    verts_indexed = proj_vert[x["cse"].vert_ids.cpu()]
    verts_pred = torch.stack([xc, yc], dim=1) + 0.5

    verts_indexed = verts_indexed.cpu().numpy()
    verts_pred = verts_pred.cpu().numpy()

    sample_indices = np.random.choice(len(verts_indexed), size=min(5000, len(verts_indexed)), replace=False)
    verts_indexed = verts_indexed[sample_indices]
    verts_pred = verts_pred[sample_indices]

    x2, y2 = verts_indexed[:, 0], verts_indexed[:, 1]
    x1, y1 = verts_pred[:, 0], verts_pred[:, 1]

    u = x2 - x1
    v = y2 - y1

    magnitudes = np.sqrt(u ** 2 + v ** 2)
    norm = plt.Normalize(magnitudes.min(), magnitudes.max())
    cmap = plt.cm.jet

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.figure(figsize=(20, 20))
    plt.colorbar(sm)
    plt.imshow(plot2d_points(x['img'][0].cpu().numpy() * 255, verts_indexed, r=0, t=1).astype(np.uint8))
    plt.quiver(x1, y1, u, v, color=cmap(norm(magnitudes)), angles='xy', scale_units='xy', scale=1, headwidth=1)
    plt.axis("off")

    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()
