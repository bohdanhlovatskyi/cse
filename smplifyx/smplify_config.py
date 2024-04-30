
from typing import Optional
from dataclasses import dataclass

@dataclass
class SmplifyConfig:
    step_size: float = 1e-2
    num_steps: int = 50
    ftol: float = 1e-5
    gtol: float = 1e-3

    early_stopping: bool = False
    init_pose_to_zero: bool = True

    optimizer: str = "lbfgs"
    history_size: int = 30
    max_iter: int = 4

    scheduler_threshold: float = 1e-2
    scheduler_factor: float = 3. / 4

    transl_grad_enabled: bool = True
    global_orient_grad_enabled: bool = True
    pose_orient_grad_enabled: bool = True
    face_pose_grad_enabled: bool = False
    camera_grad_enabled: bool = True
    shape_grad_enabled: bool = True
    expr_grad_enabled: bool = False

    joints_loss_weight: float = 0.
    cse_loss_weight: float = 1.
    pose_loss_weight: float = 1e4
    pose_prior_loss_weight: float = 0.
    rotation_loss_weight: float = 0.
    embedding_loss_weight: float = 0.

    gmof_sigma: float = 25

    save_history: bool = False
    debug_dir: str = None
