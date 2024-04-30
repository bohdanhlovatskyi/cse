from typing import Optional

import torch
import torch.nn.functional as F



def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
        do not include global orient 
    """
    return torch.exp(
        pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2


def joints_loss_func(joints_gt: torch.Tensor, joints_pred: torch.Tensor,
                     joints_conf_gt: Optional[torch.Tensor] = None,
                     joints_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Weighted L1 loss function for joints
    """
    if joints_conf_gt is None:
        joints_conf_gt = joints_gt.new_ones(*joints_gt.shape[:2], 1)
    if joints_weights is None:
        joints_weights = joints_gt.new_ones(*joints_gt.shape[:2], 1)

    joints_loss = (joints_gt - joints_pred).abs()
    joints_loss = joints_loss * joints_weights * joints_conf_gt
    joints_loss = joints_loss.sum(-1).mean()  # sum over points dim, mean over points and batches
    return joints_loss



def cse_loss_func(
        verts_valid_coords: torch.Tensor,
        verts_valid: torch.Tensor,
        verts_valid_mask: torch.Tensor,
        projected_vertices: torch.Tensor,
        config
) -> torch.Tensor:
    '''
    projected_vertices: [BS, 6890, 2]
    '''

    verts_valid[~verts_valid_mask.bool()] += 1
    verts_indexed = torch.gather(
        projected_vertices, 1, verts_valid.unsqueeze(2).expand(-1, -1, 2))
    err = gmof(verts_indexed - verts_valid_coords, sigma=config.gmof_sigma)
    err = err * verts_valid_mask.unsqueeze(2).expand(-1, -1, 2)
    return err.mean(dim=1).mean()


def rotational_loss_embeddding(
        gt_sil: torch.Tensor,
        gt_vert_ids: torch.Tensor,
        gt_pixel_embeddings: torch.Tensor,
        convex_part_ids: torch.Tensor,
        cur_raster,
        cur_attr: torch.Tensor
) -> torch.Tensor:
    cur_sil, gt_sil = cur_raster.mask.bool(), gt_sil.bool()
    vert_id_hw = -1 * torch.ones_like(gt_sil, dtype=torch.long)
    vert_id_hw[gt_sil] = gt_vert_ids
    inter = cur_sil & torch.isin(vert_id_hw, convex_part_ids)
    return (inter.clone().detach() * (gt_pixel_embeddings - cur_attr) ** 2).sum(-1).mean()


def rotational_loss(
        gt_sil: torch.Tensor,
        gt_vert_ids: torch.Tensor,
        vertices: torch.Tensor,
        projected_vertices: torch.Tensor,
        cur_raster,
        faces: torch.Tensor,
        predecessors: torch.Tensor,
        convex_part_ids: torch.Tensor,
        dists_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    cur_sil, gt_sil = cur_raster.mask.bool(), gt_sil.bool()

    # get all vertex ids that are visible in raster
    visible = torch.unique(faces[0, cur_raster.pix_to_face[cur_raster.pix_to_face != -1]])
    # select only those that belong to the convex part in question
    visible = visible[torch.isin(visible, convex_part_ids)]

    # create a hw map of raster vertices
    vert_id_hw = -1 * torch.ones_like(gt_sil, dtype=torch.long)
    vert_id_hw[gt_sil] = gt_vert_ids

    # hw mask of visible projected vertices that intersect with a convex body part
    # that is currently processed
    inter = cur_sil & torch.isin(vert_id_hw, convex_part_ids)

    projected_vertices_for_sampling = projected_vertices.clone()
    projected_vertices_for_sampling[..., 0] /= inter.size(-1)
    projected_vertices_for_sampling[..., 1] /= inter.size(-2)
    projected_vertices_for_sampling = 2 * projected_vertices_for_sampling - 1
    projected_vertices_for_sampling = projected_vertices_for_sampling.unsqueeze(1)

    gt_vert_id_under_projected = F.grid_sample(
        vert_id_hw.unsqueeze(0).float(),
        projected_vertices_for_sampling,
        mode="nearest"
    ).squeeze(1).squeeze(1).long()

    # ==========================================================================================
    # get a mask for projected vertices to compute loss only for those that are visible
    # and intersect with pseudo ground truth cse mask
    intersect_cse = F.grid_sample(
        inter.unsqueeze(0).float(),
        projected_vertices_for_sampling,
        mode="nearest"
    ).squeeze(1).squeeze(1)

    visible_mask = torch.zeros_like(intersect_cse)
    visible_mask[:, visible] = 1.

    visible_and_intersect_with_cse = intersect_cse.bool() & visible_mask.bool()
    # ==========================================================================================

    cur_vert_ids_batch, cur_vert_ids = torch.where(visible_and_intersect_with_cse)
    gt_vert_id_under_projected = gt_vert_id_under_projected[visible_and_intersect_with_cse]

    return rotation_between_vertices(gt_vert_id_under_projected, cur_vert_ids, predecessors, vertices)

def rotational_loss_dense(
        gt_sil: torch.Tensor,
        gt_vert_ids: torch.Tensor,
        vertices: torch.Tensor,
        cur_raster,
        smpl_faces: torch.Tensor,
        convex_part_ids: torch.Tensor,
        predecessors: torch.Tensor,
):
    assert gt_sil.shape[0] == 1
    gt_sil = gt_sil.bool()

    gt_vert_id_hw = -1 * torch.ones_like(gt_sil, dtype=torch.long)
    gt_vert_id_hw[gt_sil] = gt_vert_ids

    vert_ids = rast_info2emb(cur_raster.pix_to_face[0], cur_raster.bary_coords[0], smpl_faces)[None]
    cur_vert_id_hw = -1 * torch.ones_like(cur_raster.mask, dtype=torch.long)
    cur_vert_id_hw[cur_raster.mask] = vert_ids

    inter = torch.isin(cur_vert_id_hw, convex_part_ids) & torch.isin(gt_vert_id_hw, convex_part_ids)

    gt_vert_id_valid = gt_vert_id_hw[inter]
    cur_vert_id_valid = cur_vert_id_hw[inter]

    return rotation_between_vertices(gt_vert_id_valid, cur_vert_id_valid, predecessors, vertices)


def rotation_between_vertices(
    gt_vert_id_valid: torch.Tensor,
    cur_vert_id_valid: torch.Tensor,
    predecessors: torch.Tensor,
    vertices: torch.Tensor,
) -> torch.Tensor:

    nxt = predecessors[gt_vert_id_valid, cur_vert_id_valid].long()
    loss_mask = (nxt != -9999)
    if loss_mask.shape[0] == 0:
        return torch.tensor([0.0], requires_grad=True, device=loss_mask.device)

    nxt[nxt == gt_vert_id_valid] = cur_vert_id_valid[nxt == gt_vert_id_valid]

    nxt_masked = nxt[loss_mask]
    start_masked = gt_vert_id_valid[loss_mask]

    # we actually want other direction to rotate
    grad_dir = (vertices[:, nxt_masked].detach() - vertices[:, start_masked]).abs().sum(dim=-1)

    return grad_dir.mean()

def rast_info2emb(triangles_hw, bary_coords, faces):
    bary_coords[..., 2] = 1 - (bary_coords[..., 0] + bary_coords[..., 1])
    h, w = torch.where(triangles_hw != -1)
    human_faces = faces[0, triangles_hw[h, w]]
    max_bary = bary_coords.argmax(axis=2)[h, w]
    vert_ids = human_faces[torch.arange(human_faces.shape[0]), max_bary]
    return vert_ids


def rotational_loss_embeddding(
        gt_sil: torch.Tensor,
        gt_vert_ids: torch.Tensor,
        gt_pixel_embeddings: torch.Tensor,
        convex_part_ids: torch.Tensor,
        cur_raster,
        cur_attr: torch.Tensor
) -> torch.Tensor:
    cur_sil, gt_sil = cur_raster.mask.bool(), gt_sil.bool()
    vert_id_hw = -1 * torch.ones_like(gt_sil, dtype=torch.long)
    vert_id_hw[gt_sil] = gt_vert_ids
    inter = cur_sil & torch.isin(vert_id_hw, convex_part_ids)
    return (inter.clone().detach() * (gt_pixel_embeddings - cur_attr) ** 2).sum(-1).mean()


def rotational_loss(
        gt_sil: torch.Tensor,
        gt_vert_ids: torch.Tensor,
        vertices: torch.Tensor,
        projected_vertices: torch.Tensor,
        cur_raster,
        faces: torch.Tensor,
        predecessors: torch.Tensor,
        convex_part_ids: torch.Tensor,
        dists_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    cur_sil, gt_sil = cur_raster.mask.bool(), gt_sil.bool()

    # get all vertex ids that are visible in raster
    visible = torch.unique(faces[0, cur_raster.pix_to_face[cur_raster.pix_to_face != -1]])
    # select only those that belong to the convex part in question
    visible = visible[torch.isin(visible, convex_part_ids)]

    # create a hw map of raster vertices
    vert_id_hw = -1 * torch.ones_like(gt_sil, dtype=torch.long)
    vert_id_hw[gt_sil] = gt_vert_ids

    # hw mask of visible projected vertices that intersect with a convex body part
    # that is currently processed
    inter = cur_sil & torch.isin(vert_id_hw, convex_part_ids)

    projected_vertices_for_sampling = projected_vertices.clone()
    projected_vertices_for_sampling[..., 0] /= inter.size(-1)
    projected_vertices_for_sampling[..., 1] /= inter.size(-2)
    projected_vertices_for_sampling = 2 * projected_vertices_for_sampling - 1
    projected_vertices_for_sampling = projected_vertices_for_sampling.unsqueeze(1)

    gt_vert_id_under_projected = F.grid_sample(
        vert_id_hw.unsqueeze(0).float(),
        projected_vertices_for_sampling,
        mode="nearest"
    ).squeeze(1).squeeze(1).long()

    # ==========================================================================================
    # get a mask for projected vertices to compute loss only for those that are visible
    # and intersect with pseudo ground truth cse mask
    intersect_cse = F.grid_sample(
        inter.unsqueeze(0).float(),
        projected_vertices_for_sampling,
        mode="nearest"
    ).squeeze(1).squeeze(1)

    visible_mask = torch.zeros_like(intersect_cse)
    visible_mask[:, visible] = 1.

    visible_and_intersect_with_cse = intersect_cse.bool() & visible_mask.bool()
    # ==========================================================================================

    cur_vert_ids_batch, cur_vert_ids = torch.where(visible_and_intersect_with_cse)
    gt_vert_id_under_projected = gt_vert_id_under_projected[visible_and_intersect_with_cse]

    return rotation_between_vertices(gt_vert_id_under_projected, cur_vert_ids, predecessors, vertices)

def rotational_loss_dense(
        gt_sil: torch.Tensor,
        gt_vert_ids: torch.Tensor,
        vertices: torch.Tensor,
        cur_raster,
        smpl_faces: torch.Tensor,
        convex_part_ids: torch.Tensor,
        predecessors: torch.Tensor,
):
    assert gt_sil.shape[0] == 1
    gt_sil = gt_sil.bool()

    gt_vert_id_hw = -1 * torch.ones_like(gt_sil, dtype=torch.long)
    gt_vert_id_hw[gt_sil] = gt_vert_ids

    vert_ids = rast_info2emb(cur_raster.pix_to_face[0], cur_raster.bary_coords[0], smpl_faces)[None]
    cur_vert_id_hw = -1 * torch.ones_like(cur_raster.mask, dtype=torch.long)
    cur_vert_id_hw[cur_raster.mask] = vert_ids

    inter = torch.isin(cur_vert_id_hw, convex_part_ids) & torch.isin(gt_vert_id_hw, convex_part_ids)

    gt_vert_id_valid = gt_vert_id_hw[inter]
    cur_vert_id_valid = cur_vert_id_hw[inter]

    return rotation_between_vertices(gt_vert_id_valid, cur_vert_id_valid, predecessors, vertices)


def rotation_between_vertices(
    gt_vert_id_valid: torch.Tensor,
    cur_vert_id_valid: torch.Tensor,
    predecessors: torch.Tensor,
    vertices: torch.Tensor,
) -> torch.Tensor:

    nxt = predecessors[gt_vert_id_valid, cur_vert_id_valid].long()
    loss_mask = (nxt != -9999)
    if loss_mask.shape[0] == 0:
        return torch.tensor([0.0], requires_grad=True, device=loss_mask.device)

    nxt[nxt == gt_vert_id_valid] = cur_vert_id_valid[nxt == gt_vert_id_valid]

    nxt_masked = nxt[loss_mask]
    start_masked = gt_vert_id_valid[loss_mask]

    # we actually want other direction to rotate
    grad_dir = (vertices[:, nxt_masked].detach() - vertices[:, start_masked]).abs().sum(dim=-1)

    return grad_dir.mean()

def rast_info2emb(triangles_hw, bary_coords, faces):
    bary_coords[..., 2] = 1 - (bary_coords[..., 0] + bary_coords[..., 1])
    h, w = torch.where(triangles_hw != -1)
    human_faces = faces[0, triangles_hw[h, w]]
    max_bary = bary_coords.argmax(axis=2)[h, w]
    vert_ids = human_faces[torch.arange(human_faces.shape[0]), max_bary]
    return vert_ids
