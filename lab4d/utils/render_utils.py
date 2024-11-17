# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
from collections import defaultdict

import torch
import torch.nn.functional as F


def sample_cam_rays(hxy, Kinv, near_far, n_depth=64, depth=None, perturb=False):
    """Sample NeRF rays in camera space

    Args:
        hxy: (M,N,3) Homogeneous pixel coordinates on the image plane
        Kinv: (M,3,3) Inverse camera intrinsics
        near_far: (M,2) Location of near/far planes per frame
        n_depth (int): Number of points to sample along each ray
        depth: (M,N,D,1) If provided, use these Z-coordinates for each ray sample
        perturb (bool): If True, use stratified sampling and perturb depth samples
    Returns:
        xyz: (M,N,D,3) Ray points in camera space
        dir: (M,N,D,3) Ray directions in camera space
        delta: (M,N,D,1) Distance between adjacent samples along a ray
        depth: (M,N,D,1) Z-coordinate of each ray sample
    """
    M, N = hxy.shape[:2]
    dir = torch.einsum("mni,mij->mnj", hxy, Kinv.permute(0, 2, 1))  # (M, N, 3)
    dir_norm = torch.norm(dir, dim=-1)  # (M, N)


    if depth is None:
        # get depth
        z_steps = torch.linspace(0, 1, n_depth, device=dir.device)[None]  # (1, D)
        depth = near_far[:, 0:1] * (1 - z_steps) + near_far[:, 1:2] * z_steps  # (M, D)
        depth = depth[:, None, :, None].repeat(1, N, 1, 1)  # (M, N, D, 1)
    else:
        n_depth = depth.shape[2]

    # perturb depth
    if perturb:
        depth_mid = 0.5 * (depth[:, :, :-1] + depth[:, :, 1:])  # (M,N,D-1,1) mid points
        upper = torch.cat([depth_mid, depth[:, :, -1:]], -2)
        lower = torch.cat([depth[:, :, :1], depth_mid], -2)
        perturb_rand = torch.rand(depth.shape, device=dir.device)
        depth = lower + (upper - lower) * perturb_rand

    # get xyz
    xyz = dir.unsqueeze(2) * depth  # (M, N, D, 3)

    # interval between points
    deltas = depth[:, :, 1:] - depth[:, :, :-1]  # (M, N, D-1, 1)
    deltas = torch.cat([deltas, deltas[:, :, -1:]], -2)  # (M, N, D, 1)
    deltas = deltas * dir_norm[..., None, None]  # (M, N, D, 1)

    # normalized direction
    dir = dir / dir_norm.unsqueeze(-1)  # (M, N, 3)
    dir = dir.unsqueeze(2).repeat(1, 1, n_depth, 1)  # (M, N, D, 3)

    # save xyz into .ply file using trimesh
    # import ipdb; ipdb.set_trace()
    # xyz_np = xyz.detach().cpu().numpy().reshape(-1, 3)
    # import trimesh
    # cloud = trimesh.points.PointCloud(xyz_np)
    # cloud.export('sampling_pts.ply')

    return xyz, dir, deltas, depth


def render_pixel(field_dict, deltas):
    """Volume-render neural field outputs along rays

    Args:
        field_dict (Dict): Neural field outputs to render, with keys
            "density" (M,N,D,1), "vis" (M,N,D,1), and arbitrary keys (M,N,D,x)
        deltas: (M,N,D,1) Distance along rays between adjacent samples
    Returns:
        rendered (Dict): Rendered outputs, with arbitrary keys (M,N,x)
    """
    rendered = {}

    weights, transmit = compute_weights(field_dict["density"], deltas)
    rendered = integrate(field_dict, weights)

    # auxiliary outputs
    if "eikonal" in field_dict:
        rendered["eikonal"] = field_dict["eikonal"].mean(dim=(-1, -2))  # (M, N)

    if "delta_skin" in field_dict:
        rendered["delta_skin"] = field_dict["delta_skin"].mean(dim=(-1, -2))

    # visibility loss
    # part of binary cross entropy: -label * log(sigmoid(vis)), where label is transmit
    transmit = transmit[..., None].detach()
    # sharpness = 20  # 0.6->0.88
    # is_visible = torch.sigmoid(sharpness * (transmit - 0.5))
    is_visible = transmit
    vis_loss = -(F.logsigmoid(field_dict["vis"]) * is_visible).mean(-2)
    # normalize by the number of visible points
    vis_loss = vis_loss / is_visible.mean().detach()
    rendered["vis"] = vis_loss

    # mask for gaussian density
    if "gauss_density" in field_dict:
        gauss_weights, _ = compute_weights(field_dict["gauss_density"], deltas)
        rendered["gauss_mask"] = torch.sum(gauss_weights, -1, keepdim=True)
    return rendered


def compute_weights(density, deltas):
    """Compute weight and transmittance for each point along a ray

    Args:
        density (M,N,D,1): Volumetric density of points along rays
        deltas (M,N,D,1): Distance along rays between adjacent samples
    Returns:
        weights (M,N,D): Contribution of each point to the output rendering
        transmit (M,N,D): Transmittance from camera to each point along ray
    """
    density = (deltas * density)[..., 0]
    alpha_p = 1 - torch.exp(-density)  # (M, N, D)
    alpha_p = torch.cat(
        [alpha_p, torch.ones_like(alpha_p[:, :, :1])], dim=-1
    )  # (M, N, D), [a1,a2,a3,...,an,1]

    transmit = torch.cumsum(density, dim=-1)
    transmit = torch.exp(-transmit)  # (M, N, D)
    transmit = torch.cat(
        [torch.ones_like(transmit[:, :, :1]), transmit], dim=-1
    )  # (M, N, D), [1, (1-a1), (1-a1)(1-a2), ..., (1-a1)(1-a2)...(1-an)]

    # aggregate: sum to 1
    # [a1, (1-a1)a2, (1-a1)(1-a2)a3, ..., (1-a1)(1-a2)...(1-an)1]
    weights = alpha_p * transmit  # (M, N, D+1)
    weights = weights[..., :-1]  # (M, N, D), only take the first D weights
    transmit = transmit[..., 1:]  # (M, N, D), only take the first D transmits
    return weights, transmit


def integrate(field_dict, weights):
    """Integrate neural field outputs over rays render = \sum_i w_i^n * value_i

    Args:
        field_dict (Dict): Neural field outputs with arbitrary keys (M,N,D,x)
        weights: (M,N,D) Contribution of each point to the output rendering
    Returns:
        rendered (Dict): Output renderings with arbitrary keys (M,N,x)
    """
    key_skip = [
        "density",
        "vis",
        "flow",
        "eikonal",
        "xy_reproj",
        "xyz_reproj",
        "gauss_density",
    ]
    key_freeze = ["cyc_dist", "xyz_cam", "skin_entropy"]

    rendered = {}
    rendered["mask"] = torch.sum(weights, -1, keepdim=True)
    w_normalized = weights / (rendered["mask"] + 1e-6)

    for k in field_dict:
        if k in key_skip:
            continue
        elif k in key_freeze:
            wt = w_normalized.detach()
        else:
            wt = w_normalized
        rendered[k] = torch.sum(wt.unsqueeze(-1) * field_dict[k], -2)

    # remove too close points from flow rendering
    if "flow" in field_dict:
        w_flow = weights * field_dict["flow"][..., 2]
        w_flow = w_flow / (torch.sum(w_flow, -1, keepdim=True) + 1e-6)
        rendered["flow"] = torch.sum(
            w_flow.unsqueeze(-1) * field_dict["flow"][..., :2], -2
        )
    # normlaize normal
    if "normal" in field_dict:
        rendered["normal"] = F.normalize(rendered["normal"], 2, -1)

    # normalize density over all components
    density_sum = []
    key_list = []
    for k in rendered:
        if "density_" in k:
            density_sum.append(rendered[k])
            key_list.append(k)
    density_sum = torch.cat(density_sum, dim=-1).sum(-1, keepdims=True) + 1e-6
    for k in key_list:
        rendered[k.replace("density_", "mask_")] = rendered[k] / density_sum
        del rendered[k]
    return rendered


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    from https://github.com/kwea123/nerf_pl/
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, n_samples1) where n_samples is "the number of coarse samples per ray - 2"
        weights: (N_rays, n_samples)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[
        denom < eps
    ] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0]
    )
    return samples
