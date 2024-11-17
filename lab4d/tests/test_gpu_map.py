# Copyright (c) 2023 Jeff Tan, Carnegie Mellon University.
import numpy as np
import torch

from lab4d.tests.utils import check_func
from lab4d.utils.gpu_utils import gpu_map


def func(arg1, arg2):
    x = torch.ones(arg1, arg2, dtype=torch.int64, device="cuda")
    return int(torch.sum(x))


def test_gpu_map_static(n_elts):
    """Test utils/proc_utils.py::gpu_map_static"""

    def impl1(n_elts):
        return [(i + 1) * (i + 2) for i in range(n_elts)]

    def impl2(n_elts):
        return gpu_map(func, [(x + 1, x + 2) for x in range(n_elts)], method="static")

    check_func(impl1, impl2, (n_elts,), name="gpu_map_static", niters=1)


def test_gpu_map_dynamic(n_elts):
    """Test utils/proc_utils.py::gpu_map_dynamic"""

    def impl1(n_elts):
        return [(i + 1) * (i + 2) for i in range(n_elts)]

    def impl2(n_elts):
        return gpu_map(func, [(x + 1, x + 2) for x in range(n_elts)], method="dynamic")

    check_func(impl1, impl2, (n_elts,), name="gpu_map_dynamic", niters=1)


if __name__ == "__main__":
    test_gpu_map_static(11)
    # test_gpu_map_dynamic(11)
