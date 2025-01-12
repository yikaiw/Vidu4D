# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# taken from Rigidmask: https://github.com/gengshan-y/rigidmask/blob/b308b5082d09926e687c55001c20def6b0708021/utils/dydepth.py#L425
import os
import sys

import cv2
import numpy as np
import trimesh

from lab4d.utils.profile_utils import record_function

sys.path.insert(
    0,
    "%s/../third_party/vcnplus/" % os.path.join(os.path.dirname(__file__)),
)

from flowutils.flowlib import warp_flow


@record_function("compute_procrustes")
def compute_procrustes_robust(pts0, pts1):
    """
    analytical solution of R/t from correspondence
    pts0: N x 3
    pts1: N x 3
    """
    num_samples = 2000
    min_samples = 10
    extent = (pts0.max(0) - pts0.min(0)).mean()
    threshold = extent * 0.05

    inliers = []
    samples = []
    idx_array = np.arange(pts0.shape[0])
    for i in range(num_samples):
        sample = np.random.choice(idx_array, size=min_samples, replace=False)
        sol = compute_procrustes(pts0[sample], pts1[sample])

        # evaluate inliers
        R, t = sol
        pts2 = R @ pts0.T + t[:, np.newaxis]
        dist = np.linalg.norm(pts2.T - pts1, 2, axis=1)
        inliers.append((dist < threshold).sum())
        samples.append(sample)

    best_idx = np.argmax(np.sum(inliers, axis=0))
    print("inlier_ratio: ", np.max(inliers) / pts0.shape[0])
    best_sample = samples[best_idx]
    sol = compute_procrustes(pts0[best_sample], pts1[best_sample])
    return sol


@record_function("compute_procrustes")
def compute_procrustes(pts0, pts1):
    """
    analytical solution of R/t from correspondence
    pts0: N x 3
    pts1: N x 3
    """
    if pts0.shape[0] < 10:
        print("Warning: too few points for procrustes. Return identity.")
        return np.eye(3), np.zeros(3)
    pts0_mean = np.mean(pts0, 0)
    pts1_mean = np.mean(pts1, 0)
    pts0_centered = pts0 - pts0_mean
    pts1_centered = pts1 - pts1_mean
    H = pts0_centered.T @ pts1_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = pts1_mean - R @ pts0_mean

    # pts2 = R @ pts0.T + t[:, np.newaxis]
    # pts2 = pts2.T
    # trimesh.Trimesh(pts0).export('tmp/0.obj')
    # trimesh.Trimesh(pts1).export('tmp/1.obj')
    # trimesh.Trimesh(pts2).export('tmp/2.obj')
    return R, t


@record_function("two_frame_registration")
def two_frame_registration(
    depth0, depth1, flow, K0, K1, mask, registration_type="procrustes"
):
    # prepare data
    shape = flow.shape[:2]
    x0, y0 = np.meshgrid(range(shape[1]), range(shape[0]))
    x0 = x0.astype(np.float32)
    y0 = y0.astype(np.float32)
    x1 = x0 + flow[:, :, 0]
    y1 = y0 + flow[:, :, 1]
    hp0 = np.stack((x0, y0, np.ones(x0.shape)), 0).reshape((3, -1))
    hp1 = np.stack((x1, y1, np.ones(x0.shape)), 0).reshape((3, -1))

    # use bg + valid pixels to compute R/t
    # valid_mask = np.logical_and(mask, flow[..., 2] > 0).flatten()
    valid_mask = mask.flatten()
    pts0 = np.linalg.inv(K0) @ hp0 * depth0.flatten()
    depth1_warped = warp_flow(depth1.astype(float), flow[..., :2]).flatten()
    pts1 = np.linalg.inv(K1) @ hp1 * depth1_warped

    if registration_type == "procrustes":
        # Procrustes
        valid_mask = np.logical_and(valid_mask, depth1_warped > 0)
        rmat, trans = compute_procrustes(pts0.T[valid_mask], pts1.T[valid_mask])
        # rmat, trans = compute_procrustes_robust(pts0.T[valid_mask], pts1.T[valid_mask])
    elif registration_type == "pnp":
        # PnP
        _, rvec, trans = cv2.solvePnP(
            pts0.T[valid_mask.flatten(), np.newaxis],
            hp1[:2].T[valid_mask.flatten(), np.newaxis],
            K0,
            0,
            flags=cv2.SOLVEPNP_DLS,
        )
        _, rvec, trans = cv2.solvePnP(
            pts0.T[valid_mask, np.newaxis],
            hp1[:2].T[valid_mask, np.newaxis],
            K0,
            0,
            rvec,
            trans,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        rmat = cv2.Rodrigues(rvec)[0]
        trans = trans[:, 0]
    else:
        raise NotImplementedError

    cam01 = np.eye(4)
    cam01[:3, :3] = rmat
    cam01[:3, 3] = trans
    return cam01
