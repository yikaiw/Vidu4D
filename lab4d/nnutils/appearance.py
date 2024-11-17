# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import torch
import torch.nn as nn

from lab4d.nnutils.time import TimeMLP


class AppearanceEmbedding(TimeMLP):
    """Encode global appearance code over time with an MLP

    Args:
        frame_info (Dict): Metadata about the frames in a dataset
        appr_channels (int): Number of channels in appearance codes
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in the time embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        frame_info,
        appr_channels,
        D=2,
        W=64,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
        time_scale=0.1,
    ):
        self.appr_channels = appr_channels
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
        self.output = nn.Linear(W, appr_channels)

    def forward(self, t_embed):
        """
        Args:
            t: (..., self.W) Input time embeddings
        Returns:
            out: (..., appr_channels) Output appearance codes
        """
        t_feat = super().forward(t_embed)
        out = self.output(t_feat)
        return out
