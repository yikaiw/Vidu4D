# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lab4d.nnutils.base import BaseMLP
from lab4d.nnutils.embedding import PosEmbedding, TimeEmbedding, get_fourier_embed_dim


class TimeMLP(BaseMLP):
    """MLP that encodes a quantity over time.

    Args:
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
        frame_info,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
        time_scale=1.0,
    ):
        frame_offset = frame_info["frame_offset"]
        # frame_offset_raw = frame_info["frame_offset_raw"]
        if num_freq_t > 0:
            max_ts = (frame_offset[1:] - frame_offset[:-1]).max()
            # scale according to input frequency: num_frames = 64 -> freq = 6
            num_freq_t = np.log2(max_ts / 64) + num_freq_t
            # # scale according to input frequency: num_frames = 512 -> freq = 6
            # num_freq_t = np.log2(max_ts / 512) + num_freq_t
            num_freq_t = int(np.rint(num_freq_t))
            # print("max video len: %d, override num_freq_t to %d" % (max_ts, num_freq_t))

        super().__init__(
            D=D,
            W=W,
            in_channels=W,
            out_channels=W,
            skips=skips,
            activation=activation,
            final_act=True,
        )

        self.time_embedding = TimeEmbedding(
            num_freq_t, frame_info, out_channels=W, time_scale=time_scale
        )

        def loss_fn(y):
            x = self.get_vals()
            return F.mse_loss(x, y)

        self.loss_fn = loss_fn

    def forward(self, t_embed):
        """
        Args:
            t_embed: (..., self.W) Time Fourier embeddings
        Returns:
            out: (..., self.W) Time-dependent features
        """
        t_feat = super().forward(t_embed)
        return t_feat

    def mlp_init(self, loss_fn=None, termination_loss=0.0001):
        """Initialize the time embedding MLP to match external priors.
        `self.init_vals` is defined by the child class, and could be
        (nframes, 4, 4) camera poses or (nframes, 4) camera intrinsics
        """
        if loss_fn is None:
            loss_fn = self.loss_fn

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        i = 0
        while True:
            optimizer.zero_grad()
            loss = loss_fn(self.init_vals)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"iter: {i}, loss: {loss.item():.4f}")
            i += 1
            if loss < termination_loss:
                break

    def compute_distance_to_prior(self):
        """Compute L2-distance from current SE(3) / intrinsics values to
        external priors.

        Returns:
            loss (0,): Mean squared error to priors
        """
        return self.loss_fn(self.init_vals)

    def get_vals(self, frame_id=None):
        """Compute values at the given frames.

        Args:
            frame_id: (...,) Frame id. If None, evaluate at all frames
        Returns:
            pred: Predicted outputs
        """
        t_embed = self.time_embedding(frame_id)
        pred = self.forward(t_embed)
        return pred

    def get_mean_vals(self):
        """Compute the mean embedding over all frames"""
        device = self.parameters().__next__().device
        t_embed = self.time_embedding.get_mean_embedding(device)
        pred = self.forward(t_embed)
        return pred

    def get_frame_offset(self):
        """Return the number of frames before the first frame of each video"""
        return self.time_embedding.frame_offset
