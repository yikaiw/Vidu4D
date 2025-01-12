# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from lab4d.utils.profile_utils import record_function


@record_function("resize_to_target")
def resize_to_target(flowfw, aspect_ratio=None, is_flow=False):
    h, w = flowfw.shape[:2]
    if aspect_ratio is None:
        factor = np.sqrt(250 * 1000 / (h * w))
        th, tw = int(h * factor), int(w * factor)
    else:
        rh, rw = aspect_ratio[:2]
        factor = np.sqrt(250 * 1000 / (rh * rw))
        th, tw = int(rh * factor), int(rw * factor)

    factor_h = th / h
    factor_w = tw / w

    flowfw_d = cv2.resize(flowfw, (tw, th))

    if is_flow:
        flowfw_d[..., 0] *= factor_w
        flowfw_d[..., 1] *= factor_h
    return flowfw_d


@record_function("reduce_component")
def reduce_component(mask):
    dtype = mask.dtype
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if nb_components > 1:
        max_label, max_size = max(
            [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
            key=lambda x: x[1],
        )
        mask = (output == max_label).astype(int)
    mask = mask.astype(dtype)
    return mask


def robust_rot_align(rot1, rot2):
    """
    align rot1 to rot2 using RANSAC
    """
    in_thresh = 1.0 / 4 * np.pi  # 45 deg
    n_samples = rot2.shape[0]
    rots = rot2[:, :3, :3] @ rot1[:, :3, :3].transpose(0, 2, 1)

    inliers = []
    for i in range(n_samples):
        rots_aligned = rots[i : i + 1] @ rot1[:, :3, :3]
        dist = rots_aligned @ rot2[:, :3, :3].transpose(0, 2, 1)
        dist = R.from_matrix(dist).as_rotvec()
        dist = np.linalg.norm(dist, 2, axis=1)
        inliers.append((dist < in_thresh).sum())

    # Convert rotation vectors back to rotation matrices
    best_rot = rots[np.argmax(inliers)]
    # print(inliers)
    return best_rot
