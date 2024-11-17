# Copyright (c) 2023 Jeff Tan, Carnegie Mellon University.
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from lab4d.tests.utils import check_func
from lab4d.utils.geom_utils import so3_to_exp_map
from lab4d.utils.quat_transform import (
    axis_angle_to_quaternion,
    quaternion_translation_mul,
    quaternion_translation_to_se3,
)


def test_construct_eval_batch_hxy(eval_size, len_fid, device):
    """Test efficient implementation of hxy in engine/model.py::construct_eval_batch"""

    def impl1(eval_size, len_fid, device):
        x0, y0 = np.meshgrid(range(eval_size), range(eval_size))
        hxy = np.stack([x0, y0, np.ones_like(x0)], -1).reshape(-1, 3)
        hxy = torch.tensor(hxy, dtype=torch.float32, device=device)
        hxy = hxy[None].repeat(len_fid, 1, 1)
        return hxy

    def impl2(eval_size, len_fid, device):
        eval_range = torch.arange(eval_size, dtype=torch.float32, device=device)
        hxy = torch.cartesian_prod(eval_range, eval_range)
        hxy = torch.stack([hxy[:, 1], hxy[:, 0], torch.ones_like(hxy[:, 0])], -1)
        hxy = hxy[None].expand(len_fid, -1, -1)
        return hxy

    check_func(impl1, impl2, (eval_size, len_fid, device), name="construct_eval_batch")


def test_sample_grid(aabb, grid_size, device):
    """Test efficient implementation of sample_grid in nnutils/nerf.py::sample_grid"""

    def impl1(aabb, grid_size, device):
        ptx = np.linspace(aabb[0][0], aabb[1][0], grid_size).astype(np.float32)
        pty = np.linspace(aabb[0][1], aabb[1][1], grid_size).astype(np.float32)
        ptz = np.linspace(aabb[0][2], aabb[1][2], grid_size).astype(np.float32)
        query_yxz = np.stack(np.meshgrid(pty, ptx, ptz), -1)  # (y,x,z)
        query_yxz = torch.tensor(query_yxz, device=device).view(-1, 3)
        query_xyz = torch.cat(
            [query_yxz[:, 1:2], query_yxz[:, 0:1], query_yxz[:, 2:3]], -1
        )
        return query_xyz

    def impl2(aabb, grid_size, device):
        ptx = torch.linspace(aabb[0][0], aabb[1][0], grid_size, device=device)
        pty = torch.linspace(aabb[0][1], aabb[1][1], grid_size, device=device)
        ptz = torch.linspace(aabb[0][2], aabb[1][2], grid_size, device=device)
        query_xyz = torch.cartesian_prod(ptx, pty, ptz)  # (x,y,z)
        return query_xyz

    check_func(impl1, impl2, (aabb, grid_size, device), name="sample_grid")


def test_pos_embedding_forward(x, in_channels, n_freqs, alpha):
    """Test efficient implementation of nnutils/embedding.py::PosEmbedding"""

    def impl1(x, in_channels, n_freqs, alpha):
        freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs, device=x.device)
        funcs = [torch.sin, torch.cos]
        n_funcs = len(funcs)

        shape = x.shape
        device = x.device
        input_dim = shape[-1]
        output_dim = input_dim * (1 + n_freqs * n_funcs)
        out_shape = shape[:-1] + (output_dim,)

        x = x.reshape(-1, input_dim)
        out = []
        for freq in freq_bands:
            for func in funcs:
                out += [func(freq * x)]
        out = torch.cat(out, -1)

        # Apply the annealing window w = 0.5* (1+cos(pi + pi clip(alpha-j)) )
        if alpha is not None:
            alpha_freq = alpha * n_freqs
            out = out.view(-1, n_freqs, n_funcs, input_dim)
            window = alpha_freq - torch.arange(n_freqs).to(device)
            window = torch.clamp(window, 0.0, 1.0)
            window = 0.5 * (1 + torch.cos(np.pi * window + np.pi))
            window = window.view(1, -1, 1, 1)
            out = window * out

        out = out.view(-1, n_freqs * n_funcs * input_dim)
        out = torch.cat([x, out], -1)
        out = out.view(out_shape)
        return out

    def impl2(x, in_channels, n_freqs, alpha):
        freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs, device=x.device)
        funcs = [torch.sin, torch.cos]
        n_funcs = len(funcs)

        shape = x.shape
        device = x.device
        input_dim = shape[-1]
        output_dim = input_dim * (1 + n_freqs * n_funcs)
        out_shape = shape[:-1] + (output_dim,)

        x = x.reshape(-1, input_dim)
        out = torch.empty(x.shape[0], output_dim, dtype=x.dtype, device=device)
        out[:, :input_dim] = x
        out_bands = out[:, input_dim:].view(-1, n_freqs, n_funcs, input_dim)
        for i, func in enumerate(funcs):
            # (B, nfreqs, input_dim) = (1, nfreqs, 1) * (B, 1, input_dim)
            out_bands[:, :, i] = func(freq_bands[None, :, None] * x[:, None, :])

        # Apply the annealing window w = 0.5* (1+cos(pi + pi clip(alpha-j)) )
        if alpha is not None:
            alpha_freq = alpha * n_freqs
            window = alpha_freq - torch.arange(n_freqs, device=device)
            window = torch.clamp(window, 0.0, 1.0)
            window = 0.5 * (1 + torch.cos(np.pi * window + np.pi))
            window = window.view(1, -1, 1, 1)
            out_bands[:] = window * out_bands

        out = out.view(out_shape)
        return out

    check_func(
        impl1, impl2, (x, in_channels, n_freqs, alpha), name="pos_embedding_forward"
    )


def test_construct_eval_batch_hxy(eval_size, len_fid, device):
    """Test efficient implementation of hxy in engine/model.py::construct_eval_batch"""

    def impl1(eval_size, len_fid, device):
        x0, y0 = np.meshgrid(range(eval_size), range(eval_size))
        hxy = np.stack([x0, y0, np.ones_like(x0)], -1).reshape(-1, 3)
        hxy = torch.tensor(hxy, dtype=torch.float32, device=device)
        hxy = hxy[None].repeat(len_fid, 1, 1)
        return hxy

    def impl2(eval_size, len_fid, device):
        eval_range = torch.arange(eval_size, dtype=torch.float32, device=device)
        hxy = torch.cartesian_prod(eval_range, eval_range)
        hxy = torch.stack([hxy[:, 1], hxy[:, 0], torch.ones_like(hxy[:, 0])], -1)
        hxy = hxy[None].expand(len_fid, -1, -1)
        return hxy

    check_func(impl1, impl2, (eval_size, len_fid, device), name="construct_eval_batch")


def test_matmul(V):
    """Test efficient implementation of matmul in utils/quat_transform.py"""

    def impl1(V):
        return V @ V

    def impl2(V):
        return V @ V

    check_func(impl1, impl2, (V,), name="matmul")


def test_quat_to_matrix(quat):
    """Test quaternion to matrix operation"""
    from lab4d.third_party.nvp import quaternions_to_rotation_matrices
    from lab4d.utils.quat_transform import quaternion_to_matrix

    check_func(
        quaternions_to_rotation_matrices,
        quaternion_to_matrix,
        (quat,),
        name="quat_to_matrix",
    )


def test_fk(local_rest_joints, so3, edges):
    """Test efficient implementation of hxy in engine/model.py::construct_eval_batch"""

    def fk_se3(local_rest_joints, so3, edges):
        """
        Forward kinematics

        Args:
            rest_joints: (b, k, 3), rest joint positions
            so3: (b,..., k, 3), axis-angle of each joint
            edges: (idx,parent_idx)xk, parent index of each joint
        Returns:
            out: ((b,..., k, 4), (b,..., k, 4)) tuple of dual quaternion as rigid transformations from part to root
        """
        shape = so3.shape
        local_rest_joints = local_rest_joints.view(
            (shape[0],) + (1,) * (len(shape) - 3) + (-1, 3)
        )
        local_rest_joints = local_rest_joints.expand(*shape)

        # allocate global rtmat
        identity_rt = torch.eye(4, device=so3.device)
        identity_rt = identity_rt.view((1,) * (len(shape) - 2) + (-1, 4, 4))
        identity_rt = identity_rt.expand(*shape[:-1], -1, -1).clone()
        identity_rt_slice = identity_rt[..., 0, :, :].clone()
        local_rt = identity_rt.clone()
        global_rt = identity_rt.clone()

        # get local rt transformation: (..., k, 4, 4)
        local_rt[..., :3, :3] = so3_to_exp_map(so3)
        local_rt[..., :3, 3] = local_rest_joints

        for idx, parent_idx in edges.items():
            if parent_idx > 0:
                accu_rt = global_rt[..., parent_idx - 1, :, :].clone()
            else:
                accu_rt = identity_rt_slice
            global_rt[..., idx - 1, :, :] = accu_rt @ local_rt[..., idx - 1, :, :]

        return global_rt

    def fk_quat(local_rest_joints, so3, edges):
        """
        Forward kinematics

        Args:
            rest_joints: (b, k, 3), rest joint positions
            so3: (b,..., k, 3), axis-angle of each joint
            edges: (idx,parent_idx)xk, parent index of each joint
        Returns:
            out: ((b,..., k, 4), (b,..., k, 4)) tuple of dual quaternion as rigid transformations from part to root
        """
        shape = so3.shape
        local_rest_joints = local_rest_joints.view(
            (shape[0],) + (1,) * (len(shape) - 3) + (-1, 3)
        )
        local_rest_joints = local_rest_joints.expand(shape)

        # get local dq transformation: (..., k, 4)
        local_qr = axis_angle_to_quaternion(so3)

        # allocate global dq transformation: (..., k, 4)
        global_qr = torch.zeros_like(local_qr)
        global_qr[..., 0] = 1.0
        global_t = torch.zeros_like(so3)
        identity_qr = global_qr[..., 0, :].clone()
        identity_t = global_t[..., 0, :].clone()

        # go through the tree
        for idx, parent_idx in edges.items():
            if parent_idx == 0:
                parent_qr = identity_qr
                parent_t = identity_t
            else:
                parent_qr = global_qr[..., parent_idx - 1, :].clone()
                parent_t = global_t[..., parent_idx - 1, :].clone()
            (
                global_qr[..., idx - 1, :],
                global_t[..., idx - 1, :],
            ) = quaternion_translation_mul(
                (parent_qr, parent_t),
                (local_qr[..., idx - 1, :], local_rest_joints[..., idx - 1, :]),
            )
        global_dq = quaternion_translation_to_se3(global_qr, global_t)
        return global_dq

    check_func(fk_se3, fk_quat, (local_rest_joints, so3, edges), name="fk")


if __name__ == "__main__":
    # test_construct_eval_batch_hxy(64, 2, "cuda")
    # test_sample_grid(((-1.1, -1.4, -1.2), (1.1, 1.3, 1.7)), 3, "cuda")
    # test_pos_embedding_forward(
    #     torch.randn(256, 16, 64, 3, dtype=torch.float32, device="cuda"), 3, 7, 0.75
    # )

    # test_matmul(torch.randn(1, 1, 1, dtype=torch.float32, device="cuda"))

    test_quat_to_matrix(torch.randn(4096, 4, dtype=torch.float32, device="cuda"))

    # local_rest_joints = torch.randn(256, 23, 3, dtype=torch.float32, device="cuda")
    # so3 = torch.randn(256, 23, 3, dtype=torch.float32, device="cuda")
    # edges = {
    #     1: 0,  # left hip
    #     2: 0,  # right hip
    #     3: 0,  # spine
    #     4: 1,  # left knee
    #     5: 2,  # right knee
    #     6: 3,  # spine1
    #     7: 4,  # left ankle
    #     8: 5,  # right ankle
    #     9: 6,  # spine2
    #     10: 7,  # left toe
    #     11: 8,  # right toe
    #     12: 9,  # neck
    #     13: 9,  # left shoulder
    #     14: 9,  # right shoulder
    #     15: 12,  # head
    #     16: 13,  # left elbow
    #     17: 14,  # right elbow
    #     18: 16,  # left wrist
    #     19: 17,  # right wrist
    #     20: 18,  # left hand
    #     21: 19,  # right hand
    #     22: 20,  # left thumb
    #     23: 21,  # right thumb
    # }
    # test_fk(local_rest_joints, so3, edges)
