import os.path as osp
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import deepdish as dd
import smplx
import torch
import torch.nn as nn

import numpy as np
import nvdiffrast.torch as dr

from smplifyx.camera import PerspectiveCamera
from smplifyx.raster import rasterize_mesh

@dataclass
class SMPLXModelParams:
    betas: torch.Tensor
    expression: torch.Tensor
    translation: torch.Tensor
    global_orient: torch.Tensor
    body_pose: torch.Tensor
    jaw_pose: torch.Tensor
    leye_pose: torch.Tensor
    reye_pose: torch.Tensor
    left_hand_pose: torch.Tensor
    right_hand_pose: torch.Tensor
    camera: PerspectiveCamera
    body_pose_emb: Optional[torch.Tensor] = None
    left_hand_pose_emb: Optional[torch.Tensor] = None
    right_hand_pose_emb: Optional[torch.Tensor] = None


@dataclass
class SMPLXModelOutput:
    smplx_params: SMPLXModelParams
    joints3d: torch.Tensor
    joints2d: torch.Tensor
    vertices3d: torch.Tensor
    vertices2d: torch.Tensor
    vertices3d_smpl: Optional[torch.Tensor] = None
    vertices2d_smpl: Optional[torch.Tensor] = None


class SMPLXModel(nn.Module):
    def __init__(self, glctx, model_path: str, batch_size: int = 1, dtype=torch.float32,
                 num_betas: int = 300, num_expression_coeffs: int = 100, num_hand_pca_comps: int = 6,
                 body_pose_latent_dim: int = 32, use_vposer: bool = False, vposer_ckpt: Optional[str] = None,
                 joints_mapper=None, smplx2smpl_transfer_path=None, smpl_vert_embeddings_path=None):
        super().__init__()
        self.glctx = glctx

        self.batch_size = batch_size
        self.num_betas = num_betas
        self.num_expression_coeffs = num_expression_coeffs
        self.use_vposer = use_vposer

        self.smplx = smplx.SMPLX(
            model_path=model_path,
            joint_mapper=joints_mapper,
            num_betas=num_betas,
            num_expression_coeffs=num_expression_coeffs,
            num_pca_comps=num_hand_pca_comps,
            use_face_contour=False,
            gender='neutral',
            dtype=dtype,
            batch_size=batch_size,
            ext='npz',
            flat_hand_mean=False,
            create_global_orient=False,
            create_body_pose=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            create_betas=False,
            create_expression=False,
            create_transl=False,
        )

        smpl = smplx.SMPL(model_path.replace("/smplx/", "/smpl/").replace("SMPLX", "SMPL").replace(".npz", ".pkl"))
        self.register_buffer('smpl_faces', torch.from_numpy(smpl.faces.astype(np.int64)))
        self.register_buffer('smpl_vert_embeddings', torch.from_numpy(np.load(smpl_vert_embeddings_path)))

        # VPoser
        if use_vposer and vposer_ckpt is not None:
            from human_body_prior.models.vposer_model import VPoser
            from human_body_prior.tools.model_loader import load_model

            vposer_ckpt = osp.expandvars(vposer_ckpt)

            self.vposer, _ = load_model(vposer_ckpt, model_code=VPoser, remove_words_in_model_weights='vp_model.',
                                        disable_grad=True)
            self.vposer.eval()

        if smplx2smpl_transfer_path is not None:
            smplx2smpl_transfer = dd.io.load(smplx2smpl_transfer_path)
            self.register_buffer('smplx2smpl_v', torch.from_numpy(smplx2smpl_transfer['vals']))
            self.register_buffer('smplx2smpl_i', torch.from_numpy(smplx2smpl_transfer['idxs']))
            self.has_smplx2smpl_transfer = True
        else:
            self.has_smplx2smpl_transfer = False

        # ============= BEGIN PARAMETERS ===============
        self.camera = PerspectiveCamera(batch_size=batch_size, dtype=dtype)

        self.betas = nn.Parameter(torch.zeros(batch_size, num_betas, dtype=dtype))
        self.expression = nn.Parameter(torch.zeros(batch_size, num_expression_coeffs, dtype=dtype))
        self.translation = nn.Parameter(torch.zeros(batch_size, 3, dtype=dtype))
        self.global_orient = nn.Parameter(torch.zeros(batch_size, 1, 3, dtype=dtype))

        if use_vposer:
            self.body_pose_emb = nn.Parameter(torch.zeros(batch_size, body_pose_latent_dim, dtype=dtype))
        else:
            self.body_pose = nn.Parameter(torch.zeros(batch_size, 21, 3, dtype=dtype))
        self.jaw_pose = nn.Parameter(torch.zeros(batch_size, 1, 3, dtype=dtype))
        self.leye_pose = nn.Parameter(torch.zeros(batch_size, 1, 3, dtype=dtype))
        self.reye_pose = nn.Parameter(torch.zeros(batch_size, 1, 3, dtype=dtype))
        # pca components for hand pose
        self.left_hand_pose_emb = nn.Parameter(torch.zeros(batch_size, num_hand_pca_comps, dtype=dtype))
        self.right_hand_pose_emb = nn.Parameter(torch.zeros(batch_size, num_hand_pca_comps, dtype=dtype))
        # ============= END PARAMETERS =================

    def enable_grads(
        self,
        shape: Optional[bool] = None,
        expr: Optional[bool] = None,
        pose: Optional[bool] = None,
        face_pose: Optional[bool] = None, 
        camera: Optional[bool] = None, 
        transl: Optional[bool] = None, 
        global_orient: Optional[bool] = None, 
    ) -> None:
        if shape is not None:
            self.betas.requires_grad = shape
            if not shape:
                self.betas.grad = None

        if expr is not None:
            self.expression.requires_grad = expr
            if not expr:
                self.expression.grad = None

        if transl is not None:
            self.translation.requires_grad = transl
            if not transl:
                self.translation.grad = None

        if global_orient is not None:
            self.global_orient.requires_grad = global_orient
            if not global_orient:
                self.global_orient.grad = None

        if face_pose is not None:
            self.jaw_pose.requires_grad = face_pose
            self.leye_pose.requires_grad = pose
            self.reye_pose.requires_grad = pose
            if not face_pose:
                self.jaw_pose.grad = None
                self.leye_pose.grad = None
                self.reye_pose.grad = None

        if pose is not None:
            if self.use_vposer:
                self.body_pose_emb.requires_grad = pose
            else:
                self.body_pose.requires_grad = pose
            self.left_hand_pose_emb.requires_grad = pose
            self.right_hand_pose_emb.requires_grad = pose
            if not pose:
                if self.use_vposer:
                    self.body_pose_emb.grad = None
                else:
                    self.body_pose.grad = None
                self.left_hand_pose_emb.grad = None
                self.right_hand_pose_emb.grad = None

        if camera is not None:
            self.camera.f.requires_grad = self.camera.f_grad and camera
            self.camera.pp.requires_grad = self.camera.pp_grad and camera
            self.camera.R.requires_grad = self.camera.R_grad and camera
            self.camera.T.requires_grad = self.camera.T_grad and camera
            if not camera:
                self.camera.f.grad = None
                self.camera.pp.grad = None
                self.camera.R.grad = None
                self.camera.T.grad = None

    @torch.no_grad()
    def set_betas(self, betas: torch.Tensor):
        """Set shape parameters. If provided more than needed truncate. If less than needed - pad with zeros"""
        num_betas = betas.size(1)
        if num_betas > self.num_betas:
            betas = betas[..., :self.num_betas]
        elif num_betas < self.num_betas:
            betas = torch.cat([betas, betas.new_zeros(*betas.shape[:-1], self.num_betas - num_betas)], dim=-1)

        self.betas.data.copy_(betas.data)

    @torch.no_grad()
    def set_expression(self, expression: torch.Tensor):
        """Set expression parameters. If provided more than needed truncate. If less than needed - pad with zeros"""
        num_expression_coeffs = expression.size(1)
        if num_expression_coeffs > self.num_expression_coeffs:
            expression = expression[..., :self.num_expression_coeffs]
        elif num_expression_coeffs < self.num_expression_coeffs:
            expression = torch.cat([
                expression,
                expression.new_zeros(*expression.shape[:-1], self.num_expression_coeffs - num_expression_coeffs)
            ], dim=-1)

        self.expression.data.copy_(expression.data)

    @torch.no_grad()
    def set_pose_params(self, translation: torch.Tensor, global_orient: torch.Tensor,
                        body_pose: torch.Tensor, jaw_pose: torch.Tensor,
                        leye_pose: torch.Tensor, reye_pose: torch.Tensor,
                        left_hand_pose: torch.Tensor, right_hand_pose: torch.Tensor):
        """Set pose params to given once. Convert hands to lower-dim PCA space"""
        batch_size = global_orient.size(0)
        self.translation.data.copy_(translation.data)
        self.global_orient.data.copy_(global_orient.data)
        self.jaw_pose.data.copy_(jaw_pose.data)
        self.leye_pose.data.copy_(leye_pose.data)
        self.reye_pose.data.copy_(reye_pose.data)

        # encode body pose with VPoser
        if self.use_vposer:
            body_pose_emb = self.vposer.encode(body_pose).loc
            self.body_pose_emb.data.copy_(body_pose_emb.data)
        else:
            self.body_pose.data.copy_(body_pose.data)

        # encode with PCA components
        left_hand_pose_emb = torch.matmul(
            left_hand_pose.view(batch_size, -1),
            torch.pinverse(self.smplx.left_hand_components)
        )
        self.left_hand_pose_emb.data.copy_(left_hand_pose_emb.data)

        right_hand_pose_emb = torch.matmul(
            right_hand_pose.view(batch_size, -1),
            torch.pinverse(self.smplx.right_hand_components)
        )
        self.right_hand_pose_emb.data.copy_(right_hand_pose_emb.data)

    @torch.no_grad()
    def set_camera_params(self, camera: PerspectiveCamera):
        self.camera.set_focal_length(camera.f)
        self.camera.set_pp(camera.pp)
        self.camera.set_rotation(camera.R)
        self.camera.set_translation(camera.T)

    @torch.no_grad()
    def get_params(self) -> SMPLXModelParams:
        if self.use_vposer:
            body_pose = self.vposer.decode(self.body_pose_emb.detach())['pose_body'].contiguous()
        else:
            body_pose = self.body_pose.detach()

        left_hand_pose = torch.matmul(
            self.left_hand_pose_emb.detach(),
            self.smplx.left_hand_components
        ).view(self.batch_size, -1, 3)

        right_hand_pose = torch.matmul(
            self.right_hand_pose_emb.detach(),
            self.smplx.right_hand_components
        ).view(self.batch_size, -1, 3)

        camera = deepcopy(self.camera)

        return SMPLXModelParams(
            self.betas.detach(), self.expression.detach(), self.translation.detach(),
            self.global_orient.detach(), body_pose,
            self.jaw_pose.detach(), self.leye_pose.detach(), self.reye_pose.detach(),
            left_hand_pose, right_hand_pose, camera
        )

    def apply_smplx2smpl_transfer(self, vertices3d: torch.Tensor):
        return (vertices3d[:, self.smplx2smpl_i] * self.smplx2smpl_v[..., None]).sum(-2)

    def forward(self):
        if self.use_vposer:
            body_pose_emb = self.body_pose_emb
            body_pose = self.vposer.decode(body_pose_emb)['pose_body'].contiguous()
        else:
            body_pose_emb = None
            body_pose = self.body_pose

        smplx_output = self.smplx(
            betas=self.betas,
            transl=self.translation,
            global_orient=self.global_orient,
            body_pose=body_pose,
            left_hand_pose=self.left_hand_pose_emb,
            right_hand_pose=self.right_hand_pose_emb,
            expression=self.expression,
            jaw_pose=self.jaw_pose,
            leye_pose=self.leye_pose,
            reye_pose=self.reye_pose,
            return_verts=True,

        )
        joints3d = smplx_output.joints  # (B, N, 3)
        joints2d = self.camera(joints3d)

        vertices3d = smplx_output.vertices
        vertices2d = self.camera(vertices3d)

        smplx_params = SMPLXModelParams(
            self.betas, self.expression, self.translation, self.global_orient, body_pose, self.jaw_pose,
            self.leye_pose, self.reye_pose, self.left_hand_pose_emb, self.right_hand_pose_emb,
            self.camera, body_pose_emb, self.left_hand_pose_emb, self.right_hand_pose_emb
        )

        output = SMPLXModelOutput(smplx_params, joints3d, joints2d, vertices3d, vertices2d, )

        if self.has_smplx2smpl_transfer:
            output.vertices3d_smpl = self.apply_smplx2smpl_transfer(vertices3d)
            output.vertices2d_smpl = self.camera(output.vertices3d_smpl)

            # TODO: do something with hardcoded value and device conversion
            frag, attr = rasterize_mesh(
                self.glctx, self.camera, output.vertices3d_smpl,
                self.smpl_faces[None], (768, 768),
                attributes=self.smpl_vert_embeddings[None],
            )
            output.smpl_frag = frag
            output.attr = attr.permute(0, 3, 1, 2) # BHWC -> BCHW

        return output
