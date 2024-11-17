# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os

import torch


def get_local_rank():
    try:
        return int(os.environ["LOCAL_RANK"])
    except:
        # print("LOCAL_RANK not found, set to 0")
        return 0


class DataParallelPassthrough(torch.nn.parallel.DistributedDataParallel):
    """For multi-GPU access, forward attributes to the inner module."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __delattr__(self, name):
        try:
            return super().__delattr__(name)
        except AttributeError:
            return delattr(self.module, name)
