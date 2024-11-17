# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import torch

from lab4d.utils.geom_utils import hat_map, so3_to_exp_map


@torch.jit.script
def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def so3_exp_map(log_rot, eps=0.0001):
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R


def test_hat_map():
    # Define a test input tensor
    v = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    # Compute the skew-symmetric matrices using the hat_map function
    V = hat_map(v)
    # Verify that the output has the correct shape
    assert V.shape == (3, 3, 3)
    # Verify that the output is correct
    expected_V = torch.tensor(
        [
            [[0, -3, 2], [3, 0, -1], [-2, 1, 0]],
            [[0, -6, 5], [6, 0, -4], [-5, 4, 0]],
            [[0, -9, 8], [9, 0, -7], [-8, 7, 0]],
        ],
        dtype=torch.float32,
    )
    if not torch.allclose(V, expected_V):
        print("Computed output:")
        print(V)
        print("Expected output:")
        print(expected_V)
    assert torch.allclose(V, expected_V)


def test_so3_to_exp_map():
    so3 = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    exp_map = so3_exp_map(so3)
    computed_exp_map = so3_to_exp_map(so3)
    if not torch.allclose(computed_exp_map, exp_map):
        print("Computed output:")
        print(computed_exp_map)
        print("Expected output:")
        print(exp_map)


test_so3_to_exp_map()
test_hat_map()
