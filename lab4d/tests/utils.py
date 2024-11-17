# Copyright (c) 2023 Jeff Tan, Carnegie Mellon University.
import time
from statistics import mean, stdev

import numpy as np
import torch


def check_func(func1, func2, args=(), name="", niters=100, rtol=None, atol=None):
    """Verify that both input functions produce identical outputs

    Args:
        func1: First function to test
        func2: Second function to test
        args: Arguments to both functions
        name: Name of this test
        niters: Number of test iterations (default 5)
        rtol: Relative tolerance (by default, selected based on datatype)
        atol: Absolute tolerance (by default, selected based on datatype)
    """
    # Make sure cuda is already loaded
    torch.zeros(1, dtype=torch.float32, device="cuda")

    all_t1 = []
    all_t2 = []
    for i in range(niters):
        torch.cuda.synchronize()
        t1 = time.time()
        out1 = func1(*args)
        torch.cuda.synchronize()
        all_t1.append(time.time() - t1)

        torch.cuda.synchronize()
        t2 = time.time()
        out2 = func2(*args)
        torch.cuda.synchronize()
        all_t2.append(time.time() - t2)

        try:
            assert type(out1) == type(out2)
            if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
                torch.testing.assert_close(out1, out2, rtol=rtol, atol=atol)
            elif isinstance(out1, np.ndarray) and isinstance(out2, np.ndarray):
                np.testing.assert_allclose(out1, out2, rtol=rtol, atol=atol)
            else:
                assert all(
                    elt1 == elt2 for elt1, elt2 in zip(out1, out2)
                ), f"out1={out1} but out2={out2}"
        except Exception as e:
            print(f"Error: {e}")

    all_t1 = all_t1[10:]  # Remove the first few iterations to account for warmup
    all_t2 = all_t2[10:]
    avg_t1 = 1000 * mean(all_t1)  # milliseconds
    avg_t2 = 1000 * mean(all_t2)
    std_t1 = 1000 * stdev(all_t1) if len(all_t1) > 1 else 0
    std_t2 = 1000 * stdev(all_t2) if len(all_t1) > 1 else 0

    print(
        f"Test '{name}' passed:\tavg_t1={avg_t1:.2f}ms,\tavg_t2={avg_t2:.2f}ms,"
        f"\tstd_t1={std_t1:.2f}ms,\tstd_t2={std_t2:.2f}ms"
    )
