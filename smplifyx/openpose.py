

import torch
import numpy as np

from .utils import smpl_to_openpose

# https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html#autotoc_md42
class Openpose:

    NUM_BODY_JOINTS = 25

    def __init__(
            self,
            dtype=torch.float32,
            model_type='smplx',
            joints_to_ign=np.array([1, 8, 9, 12]), # Neck, MidHip, RighgHip, LeftHip
            openpose_format='coco25',
            **kwargs,
    ):
        self.use_hands = False
        self.use_face = False
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = False

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS)

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_right_shoulder(self):
        return 2

    def get_left_shoulder(self):
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
