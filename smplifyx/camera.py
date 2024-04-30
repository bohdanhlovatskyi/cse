from typing import Optional

import torch
import torch.nn as nn


class PerspectiveCamera(nn.Module):
    """Implementation of camera for SMPL-X projection.
    No camera rotation is used as it is incorporated in SMPL-X parameters"""
    FOCAL_LENGTH = 600

    def __init__(self, f: Optional[torch.Tensor] = None, pp: Optional[torch.Tensor] = None,
                 R: Optional[torch.Tensor] = None, T: Optional[torch.Tensor] = None,
                 f_grad: bool = False, pp_grad: bool = False,
                 R_grad: bool = False, T_grad: bool = False,
                 batch_size: int = 1, dtype=torch.float32):
        super().__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        self.f_grad = f_grad
        self.pp_grad = pp_grad
        self.R_grad = R_grad
        self.T_grad = T_grad

        if f is None:
            f = torch.full((self.batch_size, 1, 2), fill_value=self.FOCAL_LENGTH, dtype=dtype)
        self.f = nn.Parameter(f.type(dtype), requires_grad=f_grad)  # [B, 1, 2]

        if pp is None:
            pp = torch.zeros(self.batch_size, 1, 2, dtype=dtype)
        self.pp = nn.Parameter(pp.type(dtype), requires_grad=pp_grad)  # [B, 1, 2]

        if R is None:
            R = torch.eye(3, dtype=dtype).expand(self.batch_size, -1, -1)
        self.R = nn.Parameter(R.type(dtype), requires_grad=R_grad)  # [B, 3, 3]

        if T is None:
            T = torch.zeros(self.batch_size, 1, 3, dtype=dtype)
        self.T = nn.Parameter(T.type(dtype), requires_grad=T_grad)  # [B, 1, 3]

    @torch.no_grad()
    def set_focal_length(self, focal_length: torch.Tensor):
        self.f.data.copy_(focal_length.data)

    @torch.no_grad()
    def set_pp(self, pp: torch.Tensor):
        self.pp.data.copy_(pp.data)

    @torch.no_grad()
    def set_rotation(self, rotation: torch.Tensor):
        self.R.data.copy_(rotation.data)

    @torch.no_grad()
    def set_translation(self, translation: torch.Tensor):
        self.T.data.copy_(translation.data)

    @torch.no_grad()
    def apply_similarity(self, tform: torch.Tensor):
        """Apply similarity transform to the camera transformation"""
        tform_s = tform[:, [0, 1], [0, 1]].unsqueeze(1)
        tform_t = tform[:, :2, 2].unsqueeze(1)

        self.f.mul_(tform_s),
        self.pp.mul_(tform_s).add_(tform_t)
        return self

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        points = torch.bmm(points, self.R.transpose(1, 2)) + self.T
        points = points[..., :2] / points[..., 2:3]
        points = self.f * points + self.pp
        return points
