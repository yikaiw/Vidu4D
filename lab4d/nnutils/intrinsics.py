# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import torch.nn as nn

from lab4d.nnutils.time import TimeMLP


class IntrinsicsMLP(TimeMLP):
    """Encode camera intrinsics over time with an MLP

    Args:
        intrinsics: (N,4) Camera intrinsics (fx, fy, cx, cy)
        frame_info (Dict): Metadata about the frames in a dataset
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in the time embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        time_scale (float): Control the sensitivity to time by scaling.
          Lower values make the module less sensitive to time.
    """

    def __init__(
        self,
        intrinsics,
        frame_info=None,
        D=5,
        W=256,
        num_freq_t=0,
        skips=[],
        activation=nn.ReLU(True),
        time_scale=0.1,
    ):
        if frame_info is None:
            num_frames = len(intrinsics)
            frame_info = {
                "frame_offset": np.asarray([0, num_frames]),
                "frame_mapping": list(range(num_frames)),
                "frame_offset_raw": np.asarray([0, num_frames]),
            }
        # xyz encoding layers
        super().__init__(
            frame_info,
            D=D,
            W=W,
            num_freq_t=num_freq_t,
            skips=skips,
            activation=activation,
            time_scale=time_scale,
        )

        # output layers
        self.focal = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, 2),
        )

        # camera intrinsics: fx,fy,px,py
        self.base_logfocal = nn.Parameter(torch.zeros(self.time_embedding.num_vids, 2))
        self.base_ppoint = nn.Parameter(torch.zeros(self.time_embedding.num_vids, 2))
        self.register_buffer(
            "init_vals", torch.tensor(intrinsics, dtype=torch.float32), persistent=False
        )

    def mlp_init(self):
        """Initialize camera intrinsics from external values"""
        intrinsics = self.init_vals
        frame_offset = self.get_frame_offset()
        self.base_logfocal.data = intrinsics[frame_offset[:-1], :2].log()
        self.base_ppoint.data = intrinsics[frame_offset[:-1], 2:]
        super().mlp_init(termination_loss=1.0)

    def forward(self, t_embed):
        """
        Args:
            t_embed: (..., self.W) Input Fourier time embeddings
        Returns:
            out: (..., 4) Camera intrinsics
        """
        t_feat = super().forward(t_embed)
        focal = self.focal(t_feat).exp()
        return focal

    def get_vals(self, frame_id=None):
        """Compute camera intrinsics at the given frames.

        Args:
            frame_id: (...,) Frame id. If None, compute at all frames
        Returns:
            intrinsics: (..., 4) Output camera intrinsics
        """
        t_embed = self.time_embedding(frame_id)
        focal = self.forward(t_embed)
        if frame_id is None:
            inst_id = self.time_embedding.frame_to_vid
        else:
            inst_id = self.time_embedding.raw_fid_to_vid[frame_id]
        base_focal = self.base_logfocal[inst_id].exp()
        base_ppoint = self.base_ppoint[inst_id]
        focal = focal * base_focal
        # force square pixels
        focal[..., :] = (focal + focal.flip(-1)) / 2
        ppoint = base_ppoint.expand_as(focal)
        intrinsics = torch.cat([focal, ppoint], dim=-1)
        return intrinsics
