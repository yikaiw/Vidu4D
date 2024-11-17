# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os
import sys

import torch
import torch.backends.cudnn as cudnn
from absl import app

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from lab4d.config import get_config, save_config
from lab4d.engine.train_utils import get_local_rank
from lab4d.utils.profile_utils import record_function

cudnn.benchmark = True


def train_ddp(Trainer):
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    opts = get_config()
    if local_rank == 0:
        save_config()

    try:
        torch.distributed.init_process_group(
            "nccl",
            init_method="env://",
            world_size=opts["ngpu"],
            rank=local_rank,
        )
    except:
        pass
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(1)
    # torch.manual_seed(0)

    trainer = Trainer(opts)
    trainer.train()


def main(_):
    from lab4d.engine.trainer import Trainer

    train_ddp(Trainer)


if __name__ == "__main__":
    app.run(main)
