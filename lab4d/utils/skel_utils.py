# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
from typing import Dict

import cv2
import numpy as np
import torch

from lab4d.utils.geom_utils import so3_to_exp_map
from lab4d.utils.quat_transform import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quaternion_translation_mul,
    quaternion_translation_to_dual_quaternion,
    dual_quaternion_to_quaternion_translation,
)


def get_valid_edges(edges):
    """Get parent and child joint indices for edges with parent > 0

    Args:
        edges (Dict(int, int)): Maps each joint to its parent joint
    Returns:
        idx: (B,) Child joint indices
        parent_idx: (B,) Corresponding parent joint indices
    """
    idx = np.asarray(list(edges.keys()))
    parent_idx = np.asarray(list(edges.values()))
    valid_idx = parent_idx > 0
    idx = idx[valid_idx] - 1
    parent_idx = parent_idx[valid_idx] - 1
    return idx, parent_idx


def rest_joints_to_local(rest_joints, edges):
    """Convert rest joints to local coordinates, where local = current - parent

    Args:
        rest_joints: (B, 3) Joint locations
        edges (Dict(int, int)): Maps each joint to its parent joint
    Returns:
        local_rest_joints: (B, 3) Translations from parent to child joints
    """
    idx, parent_idx = get_valid_edges(edges)
    local_rest_joints = rest_joints.clone()
    local_rest_joints[idx] = rest_joints[idx] - rest_joints[parent_idx]
    return local_rest_joints


def fk_se3(local_rest_joints, so3, edges, to_dq=True):
    """Compute forward kinematics given joint angles on a skeleton

    Args:
        local_rest_joints: (B, 3) Translations from parent to current joints,
            assuming identity rotation in zero configuration
        so3: (..., B, 3) Axis-angles at each joint
        edges (Dict(int, int)): Maps each joint to its parent joint
        to_dq (bool): If True, output link rigid transforms as dual quaternions,
            otherwise output SE(3)
    Returns:
        out: Location of each joint. This is written as dual quaternions
            ((..., B, 4), (..., B, 4)) if to_dq=True, otherwise it is written
            as (..., B, 4, 4) SE(3) matrices.
    """
    assert local_rest_joints.shape == so3.shape
    shape = so3.shape

    # allocate global rtmat
    identity_rt = torch.eye(4, device=so3.device)
    identity_rt = identity_rt.view((1,) * (len(shape) - 2) + (-1, 4, 4))
    identity_rt = identity_rt.expand(*shape[:-1], -1, -1).clone()
    identity_rt_slice = identity_rt[..., 0, :, :].clone()
    local_to_parent = identity_rt.clone()
    global_rt = identity_rt.clone()

    # get local rt transformation: (..., k, 4, 4)
    # first rotate around joint i
    # then translate wrt the relative position of the parent to i
    local_to_parent[..., :3, :3] = so3_to_exp_map(so3)
    local_to_parent[..., :3, 3] = local_rest_joints

    for idx, parent_idx in edges.items():
        if parent_idx > 0:
            parent_to_global = global_rt[..., parent_idx - 1, :, :].clone()
        else:
            parent_to_global = identity_rt_slice
        global_rt[..., idx - 1, :, :] = (
            parent_to_global @ local_to_parent[..., idx - 1, :, :]
        )

    if to_dq:
        global_quat = matrix_to_quaternion(global_rt[..., :3, :3])
        global_dq = quaternion_translation_to_dual_quaternion(
            global_quat, global_rt[..., :3, 3]
        )
        return global_dq
    else:
        return global_rt


def shift_joints_to_bones_dq(dq, edges, shift=None):
    """Compute bone centers and orientations from joint locations

    Args:
        dq: ((..., B, 4), (..., B, 4)) Location of each joint, written as dual
            quaternions
        edges (Dict(int, int)): Maps each joint to its parent joint
    Returns:
        dq: ((..., B, 4), (..., B, 4)) Bone-to-object SE(3) transforms,
            written as dual quaternions
    """
    quat, joints = dual_quaternion_to_quaternion_translation(dq)
    if shift is not None:
        joints += shift.reshape((1,) * (joints[0].ndim - 1) + (3,))
    joints = shift_joints_to_bones(joints, edges)
    dq = quaternion_translation_to_dual_quaternion(quat, joints)
    return dq


def shift_joints_to_bones(joints, edges):
    """Compute bone centers and orientations from joint locations

    Args:
        joints: (..., B, 3) Location of each joint
        edges (Dict(int, int)): Maps each joint to its parent joint
    Returns:
        joints: (..., B, 3) Location of each joint
    """
    idx, parent_idx = get_valid_edges(edges)
    # find the center between each joint and its children
    joint_center = (joints[..., parent_idx, :] + joints[..., idx, :]) / 2
    joints[..., parent_idx, :] = joint_center
    for i in list(set(parent_idx)):
        if np.sum(parent_idx == i) > 1:
            # average over all the children
            joints[..., i, :] = (joint_center[..., parent_idx == i, :]).mean(dim=-2)
    return joints


def get_predefined_skeleton(skel_type):
    """Compute pre-defined skeletons

    Args:
        skel_type (str): Skeleton type ("human" or "quad")
    Returns:
        rest_joints:
    return rest_joints, edges, symm_idx
    """
    # assuming fixed base, i.e.: root is not moving within fk

    # for sanity check
    BOB_PARENT = {key: 0 for key in range(1, 25)}
    BOB_SYMM_IDX = {key: key for key in range(1, 25)}
    BOB_REST_JOINTS = torch.randn(25, 3) * 0.02

    # TODO: annotate the semantic meaning of each joint
    # TODO: map human to QUAD
    HUMAN_PARENT = {
        1: 0,  # spine 1
        13: 0,  # left upper leg
        16: 0,  # right upper leg
        2: 1,  # spine 2
        3: 2,  # spine 3
        4: 3,  # head
        5: 3,  # left shoulder
        9: 3,  # right shoulder
        6: 5,  # left arm
        7: 6,  # left forearm
        8: 7,  # left hand
        10: 9,  # right arm
        11: 10,  # right forearm
        12: 11,  # right hand
        14: 13,  # left lower leg
        15: 14,  # left foot
        17: 16,  # right lower leg
        18: 17,  # right foot
    }

    HUMAN_SYMM_IDX = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 9,
        6: 10,
        7: 11,
        8: 12,
        9: 5,
        10: 6,
        11: 7,
        12: 8,
        13: 16,
        14: 17,
        15: 18,
        16: 13,
        17: 14,
        18: 15,
    }

    HUMAN_REST_JOINTS = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-3.6278e-05, 3.6903e-03, -7.2475e-04],
            [-9.3221e-05, 8.0693e-03, -1.1619e-03],
            [-1.2457e-04, 1.3251e-02, -1.3801e-03],
            [-6.0306e-05, 1.8105e-02, -7.8039e-04],
            [2.2711e-03, 1.6784e-02, -8.8300e-04],
            [7.1616e-03, 1.6918e-02, -1.6573e-03],
            [1.7433e-02, 1.6934e-02, -1.7350e-03],
            [2.7266e-02, 1.6963e-02, -1.7920e-03],
            [-2.4980e-03, 1.6817e-02, -9.5435e-04],
            [-7.4151e-03, 1.6886e-02, -1.9168e-03],
            [-1.7819e-02, 1.6867e-02, -1.7721e-03],
            [-2.7194e-02, 1.6867e-02, -1.6701e-03],
            [3.4517e-03, -2.5785e-03, 4.9599e-04],
            [3.3529e-03, -1.8460e-02, 2.0430e-04],
            [3.3907e-03, -3.4376e-02, -7.4148e-04],
            [-3.4360e-03, -2.6853e-03, 2.9919e-05],
            [-3.3118e-03, -1.8488e-02, 2.1094e-04],
            [-3.3864e-03, -3.4373e-02, -7.9789e-04],
        ]
    )
    HUMAN_REST_JOINTS *= 2.5  # upscale to match the initial obj bound
    HUMAN_REST_JOINTS = adjust_rest_joints(HUMAN_REST_JOINTS)

    QUAD_PARENT = {
        1: 0,  # spine 1
        13: 0,  # tail 1
        18: 0,  # left hip
        22: 0,  # right hip
        2: 1,  # spine 2
        3: 2,  # spine 3
        4: 3,  # spine 4
        5: 3,  # left shoulder
        9: 3,  # right shoulder
        6: 5,  # left elbow
        7: 6,  # left wrist
        8: 7,  # left hand
        10: 9,  # right elbow
        11: 10,  # right wrist
        12: 11,  # right hand
        14: 13,  # tail 2
        15: 14,  # tail 3
        16: 15,  # tail 4
        17: 16,  # tail 5
        19: 18,  # left knee
        20: 19,  # left ankle
        21: 20,  # left foot
        23: 22,  # right knee
        24: 23,  # right ankle
        25: 24,  # right foot
    }

    QUAD_SYMM_IDX = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 9,
        6: 10,
        7: 11,
        8: 12,
        9: 5,
        10: 6,
        11: 7,
        12: 8,
        13: 13,
        14: 14,
        15: 15,
        16: 16,
        17: 17,
        18: 22,
        19: 23,
        20: 24,
        21: 25,
        22: 18,
        23: 19,
        24: 20,
        25: 21,
    }

    QUAD_REST_JOINTS = torch.tensor(
        [
            [0.0000e00, 0.01, 0.03],
            [-9.3610e-05, 1.0187e-03, -2.1873e-02],
            [-5.4921e-05, 1.7428e-03, -9.3399e-03],
            [-8.7874e-05, 2.8378e-03, 4.7383e-03],
            [-6.6505e-05, 1.9184e-02, 1.9050e-02],
            [6.6107e-03, 8.1839e-03, 1.1086e-02],
            [9.1702e-03, -7.7618e-03, 1.0090e-02],
            [1.0476e-02, -2.7165e-02, 6.9399e-03],
            [1.1353e-02, -3.5803e-02, 1.1250e-02],
            [-6.9130e-03, 8.2406e-03, 1.1061e-02],
            [-9.5720e-03, -7.6817e-03, 1.0104e-02],
            [-1.0856e-02, -2.7090e-02, 7.0649e-03],
            [-1.1773e-02, -3.5696e-02, 1.1439e-02],
            [3.2358e-05, 6.6986e-03, -4.5738e-02],
            [9.5675e-05, 3.9485e-03, -5.4802e-02],
            [1.6878e-04, 3.1219e-03, -6.3845e-02],
            [2.2074e-04, 4.3004e-03, -7.3049e-02],
            [2.0674e-04, 6.3312e-03, -8.2086e-02],
            [7.4309e-03, -2.5624e-03, -3.3335e-02],
            [7.9435e-03, -1.7319e-02, -3.6508e-02],
            [8.1728e-03, -2.8493e-02, -3.9845e-02],
            [8.5748e-03, -3.3565e-02, -3.7078e-02],
            [-7.5478e-03, -2.5571e-03, -3.3397e-02],
            [-8.2738e-03, -1.7257e-02, -3.6706e-02],
            [-8.6677e-03, -2.8381e-02, -4.0128e-02],
            [-9.1048e-03, -3.3482e-02, -3.7373e-02],
        ],
        dtype=torch.float32,
    )

    # QUAD_REST_JOINTS = torch.tensor(
    #     [
    #         [0.00, 0.02, -0.01],
    #         [0.0000e00, 2.0328e-03, 1.1140e-02],
    #         [-1.8135e-05, 2.4967e-03, 2.3316e-02],
    #         [-4.9199e-05, 3.4470e-03, 3.5327e-02],
    #         [-1.0905e-04, -2.2856e-03, 4.9982e-02],
    #         [4.1884e-03, 4.2455e-03, 4.5376e-02],
    #         [8.8315e-03, -6.8146e-03, 5.5930e-02],
    #         [1.0025e-02, -2.6900e-02, 4.5117e-02],
    #         [1.1872e-02, -4.5686e-02, 4.8696e-02],
    #         [-4.4176e-03, 4.2007e-03, 4.5323e-02],
    #         [-8.9425e-03, -6.8600e-03, 5.5928e-02],
    #         [-9.6777e-03, -2.7020e-02, 4.5212e-02],
    #         [-1.1257e-02, -4.5812e-02, 4.8885e-02],
    #         [0.0000e00, -1.3309e-03, -5.6403e-04],
    #         [1.4841e-05, -4.9808e-03, -1.0146e-02],
    #         [3.9881e-05, -6.9987e-03, -1.8760e-02],
    #         [7.0156e-05, -6.9221e-03, -2.8164e-02],
    #         [8.1609e-05, -4.9885e-03, -3.7222e-02],
    #         [6.5738e-03, -2.3803e-03, 5.7810e-03],
    #         [8.2078e-03, -2.1237e-02, 7.3598e-03],
    #         [7.9324e-03, -3.2467e-02, 3.1862e-05],
    #         [8.7782e-03, -4.3558e-02, 6.0882e-05],
    #         [-6.5738e-03, -2.3803e-03, 5.7810e-03],
    #         [-8.2102e-03, -2.1242e-02, 7.2932e-03],
    #         [-7.8715e-03, -3.2444e-02, -7.6227e-05],
    #         [-8.6467e-03, -4.3539e-02, -8.7928e-05],
    #     ],
    #     dtype=torch.float32,
    # )

    QUAD_REST_JOINTS = adjust_rest_joints(QUAD_REST_JOINTS)

    if skel_type == "human":
        edges, rest_joints, symm_idx = HUMAN_PARENT, HUMAN_REST_JOINTS, HUMAN_SYMM_IDX
    elif skel_type == "quad":
        edges, rest_joints, symm_idx = QUAD_PARENT, QUAD_REST_JOINTS, QUAD_SYMM_IDX
    else:
        raise ValueError("Unknown skeleton type %s" % skel_type)

    rest_joints = rest_joints[1:] + rest_joints[:1]
    symm_idx = [v - 1 for v in symm_idx.values()]
    return rest_joints, edges, symm_idx


def adjust_human_rest_joints(HUMAN_REST_JOINTS, HUMAN_PARENT, leg_angle=np.pi / 6):
    """Convert human rest joints from GL to CV coordinates and adjust the pose

    Args:
        HUMAN_REST_JOINTS: (B, 3) GL joint locations
        HUMAN_PARENT (Dict(int, int)): Maps each joint to its parent joint
        leg_angle (float): Amount to rotate the leg
    Returns:
        HUMAN_REST_JOINTS: (B, 3) GL joint locations
    """
    local_js = rest_joints_to_local(HUMAN_REST_JOINTS[1:], HUMAN_PARENT)
    rot_lft = cv2.Rodrigues(np.asarray([0, 0, leg_angle]))[0]
    rot_lft = torch.tensor(rot_lft, dtype=torch.float32)
    rot_rht = cv2.Rodrigues(np.asarray([0, 0, -leg_angle]))[0]
    rot_rht = torch.tensor(rot_rht, dtype=torch.float32)
    local_js[0] = rot_lft @ local_js[0]
    local_js[3] = rot_lft @ local_js[3]
    local_js[6] = rot_lft @ local_js[6]
    local_js[1] = rot_rht @ local_js[1]
    local_js[4] = rot_rht @ local_js[4]
    local_js[7] = rot_rht @ local_js[7]
    HUMAN_REST_JOINTS[1:] = fk_se3(
        local_js[None],
        torch.zeros(1, len(local_js), 3),
        HUMAN_PARENT,
        to_dq=False,
    )[0, :, :3, 3]
    HUMAN_REST_JOINTS[0, 1] += 0.5
    HUMAN_REST_JOINTS *= 0.1  # downscale to match the initial obj bound
    HUMAN_REST_JOINTS[:, 1:] *= -1  # GL to CV coordinate
    return HUMAN_REST_JOINTS


def adjust_rest_joints(REST_JOINTS):
    """Convert rest joints from GL to CV coordinates

    Args:
        REST_JOINTS: (B, 3) GL joint locations
    Returns:
        REST_JOINTS: (B, 3) CV joint locations
    """
    REST_JOINTS[:, 1:] *= -1  # GL to CV coordinate
    return REST_JOINTS
