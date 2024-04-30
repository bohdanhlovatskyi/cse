from typing import Tuple, List, Optional

import json
import os
import os.path as osp

import cv2
import deepdish as dd
import numpy as np
import torch
from torch.utils.data import Dataset

from smplifyx.utils import (
    smpl_to_openpose,
    get_all_paths,
    get_all_hbw_paths,
    get_all_ssp3d_paths,
    get_all_ehf_paths,
)


def create_dataset(dataset='openpose', **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(**kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)
    return keypoints


class OpenPose(Dataset):
    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(
            self,
            dataset_name: Optional[str] = "images",
            images_root: Optional[str] = ".",
            predictions_root: Optional[str] = ".",
            use_hands: bool = True,
            use_face: bool = True,
            dtype=torch.float32,
            model_type='smplx',
            joints_to_ign=None,
            use_face_contour=False,
            openpose_format='coco25',
            **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)


        self.images_root = images_root
        self.predictions_root = predictions_root

        if dataset_name == "images":
            self.img_paths = get_all_paths(self.images_root)
        elif dataset_name == "ehf":
            self.img_paths = get_all_ehf_paths(self.images_root)
        else:
            raise NotImplementedError("Unknown dataset {}".format(dataset_name))

        self.img_paths = sorted(self.img_paths)
        self.cnt = 0

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    # TODO: probably broken
    def get_left_shoulder(self):
        return 2

    # TODO: probably broken
    def get_right_shoulder(self):
        return 5

    # TODO: probably broken
    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    def read_item(self, img_path):
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        img_fn, _ = osp.splitext(osp.split(img_path)[1])

        keypoint_fn = osp.join(self.predictions_root, "keypoints",
                               img_fn + '_keypoints.json')

        keypoints = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                   use_face=self.use_face,
                                   use_face_contour=self.use_face_contour)

        # cse = dd.io.load(os.path.join(self.predictions_root, "cse", img_fn + "_cse.h5"))
        if len(keypoints) < 1:
            return {}

        keypoints = keypoints[0]

        params_init_path = osp.join(self.predictions_root, "smplx_init", img_fn + '.h5')
        params_init = dd.io.load(params_init_path)

        # camera
        # set pp to the middle of the image and adjust x and y translation accordingly
        H, W, _ = img.shape
        pp_new = np.array([W / 2, H / 2], dtype=np.float32)
        transl_xy, transl_z = params_init['transl'][:2], params_init['transl'][2]
        focal = params_init['focal']
        pp = params_init['princpt']
        transl_xy_new = transl_xy + transl_z / focal * (pp - pp_new)

        # set f to sqrt(h^2 + w^2), adjust z accordingly
        f_new = np.sqrt(W ** 2 + H ** 2)

        transl_z_new = transl_z * f_new / params_init['focal'].mean()
        transl = np.array([transl_xy_new[0], transl_xy_new[1], transl_z_new], dtype=np.float32)

        camera = dict(
            f=np.array([f_new, f_new])[None],
            pp=pp_new[None]
        )

        # convert to bbox space
        img_bbox, tform = crop_square_bbox(img, params_init['bbox'], target_size=768, scale=1.0,
                                           borderValue=(0.5, 0.5, 0.5))
        keypoints_bbox = transform_points2d(keypoints[:, :2], tform)
        keypoints_bbox = np.concatenate([keypoints_bbox, keypoints[:, 2:3]], axis=1)
        camera_bbox = dict(
            f=torch.tensor(camera['f'] * tform[[0, 1], [0, 1]], dtype=torch.float32),
            pp=torch.tensor(camera['pp'] * tform[[0, 1], [0, 1]] + tform[:2, 2], dtype=torch.float32)
        )

        output_dict = {
            'fn': img_fn,
            'img_path': img_path,
            'init_path': params_init_path,
            'keypoints': torch.from_numpy(keypoints_bbox),
            'img': img_bbox,
            'img_full': img,
            'bbox': params_init["bbox"],
            'transl': params_init["transl"],
            'focal': params_init['focal'],
            'princpt': params_init['princpt'],
            'faces': params_init['faces'],

            # 'vert_ids': cse.vert_ids,
            # 'silhouette': cse.segmentation,
            # 'embedding': cse.embedding,

            'params_init': dict(
                betas=torch.from_numpy(params_init['betas']),
                expression=torch.from_numpy(params_init['expression']),
                translation=torch.from_numpy(transl),
                global_orient=torch.from_numpy(params_init['global_orient']),
                body_pose=torch.from_numpy(params_init['body_pose']),
                jaw_pose=torch.from_numpy(params_init['jaw_pose']),
                leye_pose=torch.from_numpy(params_init['leye_pose']),
                reye_pose=torch.from_numpy(params_init['reye_pose']),
                left_hand_pose=torch.from_numpy(params_init['left_hand_pose']),
                right_hand_pose=torch.from_numpy(params_init['right_hand_pose']),
                camera=camera_bbox
            )}

        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.cnt]
        self.cnt += 1

        return self.read_item(img_path)


def transform_points2d(points2d, tform):
    n, _ = points2d.shape
    points2d_hom = np.concatenate([points2d, np.ones((n, 1))], axis=1)
    points2d_hom = np.matmul(points2d_hom, tform.T)
    points2d = points2d_hom[..., :2] / points2d_hom[..., 2:]
    return points2d


def bbox_transform(bbox, target_size: int, scale: float = 1.0):
    left, top, width, height = bbox
    size = 0.5 * scale * max(width, height)
    center_x, center_y = 0.5 * width + left, 0.5 * height + top

    src_pts = np.stack([
        np.array([center_x - size, center_y - size]),
        np.array([center_x - size, center_y + size]),
        np.array([center_x + size, center_y - size]),
        np.array([center_x + size, center_y + size]),
    ]).astype(np.float32)

    # crop square around person and resize to low res
    dst_pts = np.array([
        [0., 0.],
        [0., target_size],
        [target_size, 0.],
        [target_size, target_size],
    ], dtype=np.float32)

    tform = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return tform


def crop_square_bbox(image, bbox, target_size: int, scale: float = 1.0, borderValue: Tuple[float, float, float] = 0.):
    tform = bbox_transform(bbox, target_size, scale)
    dst_image = cv2.warpPerspective(image, tform, (target_size, target_size), borderValue=borderValue)

    return dst_image, tform
