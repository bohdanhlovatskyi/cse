from dataclasses import dataclass
from typing import Tuple, Optional

import nvdiffrast.torch as dr
import torch

from .camera import PerspectiveCamera

def camera_to_gl_mvp(
        camera: PerspectiveCamera,
        image_shape: Tuple[int, int],
        near: float,
        far: float) -> torch.Tensor:
    """Return camera pose (translation) and model-view-projection matrix used by opengl"""
    world2cam = torch.cat((
        torch.cat((camera.R[0], camera.T[0].T), dim=1),  # (3,3) + (3,1) = (3, 4)
        camera.R.new_tensor([[0., 0., 0., 1]])  # (1, 4)
    ), dim=0)

    swap_yz = world2cam.new_tensor([
        [1., 0., 0., 0.],
        [0., -1., 0., 0.],
        [0., 0., -1., 0.],
        [0., 0., 0., 1.],
    ])

    # calculate projection matrix from perspective camera params and near/far clipping planes
    H, W = image_shape
    # if near and far are None estimate them by +- 1 around zero

    fx, fy = camera.f[0, 0, 0].item(), camera.f[0, 0, 1].item()
    canvas_x, canvas_y = near * W / fx, near * H / fy

    cx, cy = camera.pp[0, 0, 0].item(), camera.pp[0, 0, 1].item()
    top = canvas_y * cy / H
    bottom = -canvas_y * (1 - cy / H)
    left = -canvas_x * cx / W
    right = canvas_x * (1 - cx / W)

    proj4x4 = world2cam.new_tensor([
        [2 * near / (right - left), 0, (right + left) / (right - left), 0],
        [0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]
    ])

    mv = swap_yz @ world2cam
    mvp = proj4x4 @ mv

    return mvp.unsqueeze(0)


@dataclass
class Fragments:
    mask: torch.Tensor
    pix_to_face: torch.Tensor
    bary_coords: torch.Tensor




def rasterize_mesh(
        glctx, camera: PerspectiveCamera,
        vertices: torch.Tensor, faces: torch.Tensor,
        image_shape: Tuple[int, int],
        attributes: Optional[torch.Tensor] = None):

    vertices_hom = torch.cat((
        vertices,  # (B=1, N, 3)
        vertices.new_ones(*vertices.shape[:2], 1)
    ), dim=-1)

    mvp = camera_to_gl_mvp(
        camera, image_shape,
        near=0.01,
        far=100.
    )
    v_pos_clip = torch.bmm(vertices_hom, mvp.transpose(1, 2))

    fragments_t, _ = dr.rasterize(glctx, v_pos_clip, faces[0].int(), image_shape)
    if attributes is not None:
        attr, _ = dr.interpolate(attributes, fragments_t, faces[0].int())
        attr = dr.antialias(attr, fragments_t, v_pos_clip, faces[0].int())
        attr = torch.flip(attr, dims=(1, )).contiguous()

    fragments_t = torch.flip(fragments_t, dims=(1,)).contiguous()

    bary_coords = fragments_t[..., :2]  # [B, H, W, 2]
    bary_coords = torch.cat((
        bary_coords,
        1 - bary_coords.sum(-1, keepdim=True)
    ), dim=-1)
    pix_to_face = fragments_t[..., 3]

    fragments_t = Fragments(
        mask=pix_to_face > 0,
        pix_to_face=(pix_to_face - 1).long(),
        bary_coords=bary_coords
    )

    if attributes is not None:
        return fragments_t, attr
    return fragments_t
