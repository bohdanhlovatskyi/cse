from __future__ import division
import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import torch.nn as nn
from .network.net_modules import  StructureEncoder, DFNet
from torch.autograd import grad

import yaml
def load_config(path):
    """ load config file"""
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


class PoseNDF(torch.nn.Module):

    def __init__(self, opt_path):
        super().__init__()

        opt = load_config(opt_path)

        self.device = opt['train']['device']

        # create all the models:
        # self.shape_model = ShapeNet().to(self.device)
        # self.pose_model = PoseNet().to(self.device)
        self.enc = None
        if opt['model']['StrEnc']['use']:
            self.enc = StructureEncoder(opt['model']['StrEnc']).to(self.device)

        self.dfnet = DFNet(opt['model']['DFNet']).to(self.device)
        
        
        #geo_weights = np.load(os.path.join(DATA_DIR, 'real_g5_geo_weights.npy'))  todo: do we need this???
        self.loss = opt['train']['loss_type']
        self.batch_size= opt['train']['batch_size']


        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()
        
       
    def train(self, mode=True):
        super().train(mode)


    def forward(self, pose, dist_gt=None, man_poses=None, train=True,eikonal=0.0 ):

        pose = pose.to(device=self.device).reshape(-1,21,4)
        if train and eikonal > 0.0:
            pose.requires_grad=True
        #pose = torch.nn.functional.normalize(pose.to(device=self.device),dim=-1)

        if train:
            dist_gt = dist_gt.to(device=self.device).reshape(-1)
        if self.enc:
            rand_pose_in = self.enc(pose)
        dist_pred = self.dfnet(rand_pose_in)
        if train:
            #calculate distance for manifold poses
            man_poses = man_poses.to(device=self.device).reshape(-1,21,4)
            #man_poses = torch.nn.functional.normalize(man_poses.to(device=self.device),dim=-1)
            if self.enc:
                man_pose_in = self.enc(man_poses)
            dist_man = self.dfnet(man_pose_in)
            loss = self.loss_l1(dist_pred[:,0], dist_gt)
            loss_man = (dist_man.abs()).mean()
            if eikonal > 0.0:
                # eikonal term loss
                grad_val = gradient(pose, dist_pred)
                eikonal_loss =  ((grad_val.norm(2, dim=-1) - 1) ** 2).mean()
                return loss, {'dist': loss , 'man_loss': loss_man, 'eikonal': eikonal_loss}

            return loss, {'dist': loss,  'man_loss': loss_man }
        else:
            return {'dist_pred': dist_pred}

def quat_flip(pose_in):
    is_neg = pose_in[:,:,0] <0
    pose_in[is_neg] = (-1)*pose_in[is_neg]
    return pose_in, is_neg

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions