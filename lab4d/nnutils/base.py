# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import torch
import torch.nn as nn

from lab4d.nnutils.embedding import InstEmbedding


class ScaleLayer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.FloatTensor([scale]))

    def forward(self, input):
        return input * self.scale


class BaseMLP(nn.Module):
    """Adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py

    Args:
        D (int): Number of linear layers for density (sigma) encoder
        W (int): Number of hidden units in each MLP layer
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        final_act (bool): If True, apply the activation function to the output
    """

    def __init__(
        self,
        D=8,
        W=256,
        in_channels=63,
        out_channels=3,
        skips=[4],
        activation=nn.ReLU(True),
        final_act=False,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips

        if in_channels == 0:
            return

        # linear layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, activation)
            setattr(self, f"linear_{i+1}", layer)
        if final_act:
            self.linear_final = nn.Sequential(nn.Linear(W, out_channels), activation)
        else:
            self.linear_final = nn.Linear(W, out_channels)

    def forward(self, x):
        """
        Args:
            x: (..., self.in_channels)
        Returns:
            out: (..., self.out_channels)
        """
        out = x
        for i in range(self.D):
            if i in self.skips:
                out = torch.cat([x, out], -1)
            out = getattr(self, f"linear_{i+1}")(out)
        out = self.linear_final(out)
        return out


class CondMLP(BaseMLP):
    """A MLP that accepts both input `x` and condition `c`

    Args:
        num_inst (int): Number of distinct object instances. If --nosingle_inst
            is passed, this is equal to the number of videos, as we assume each
            video captures a different instance. Otherwise, we assume all videos
            capture the same instance and set this to 1.
        D (int): Number of linear layers for density (sigma) encoder
        W (int): Number of hidden units in each MLP layer
        in_channels (int): Number of channels in input `x`
        inst_channels (int): Number of channels in condition `c`
        out_channels (int): Number of output channels
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        final_act (bool): If True, apply the activation function to the output
    """

    def __init__(
        self,
        num_inst,
        D=8,
        W=256,
        in_channels=63,
        inst_channels=32,
        out_channels=3,
        skips=[4],
        activation=nn.ReLU(True),
        final_act=False,
    ):
        super().__init__(
            D=D,
            W=W,
            in_channels=in_channels + inst_channels,
            out_channels=out_channels,
            skips=skips,
            activation=activation,
            final_act=final_act,
        )

        self.inst_embedding = InstEmbedding(num_inst, inst_channels)

    def forward(self, feat, inst_id):
        """
        Args:
            feat: (M, ..., self.in_channels)
            inst_id: (M,) Instance id, or None to use the average instance
        Returns:
            out: (M, ..., self.out_channels)
        """
        if inst_id is None:
            if self.inst_embedding.out_channels > 0:
                inst_code = self.inst_embedding.get_mean_embedding()
                inst_code = inst_code.expand(feat.shape[:-1] + (-1,))
                # print("inst_embedding exists but inst_id is None, using mean inst_code")
            else:
                # empty, falls back to single-instance NeRF
                inst_code = torch.zeros(feat.shape[:-1] + (0,), device=feat.device)
        else:
            inst_code = self.inst_embedding(inst_id)
            inst_code = inst_code.view(
                inst_code.shape[:1] + (1,) * (feat.ndim - 2) + (-1,)
            )
            inst_code = inst_code.expand(feat.shape[:-1] + (-1,))

        feat = torch.cat([feat, inst_code], -1)
        # if both input feature and inst_code are empty, return zeros
        if feat.shape[-1] == 0:
            return feat
        return super().forward(feat)

    @staticmethod
    def get_dim_inst(num_inst, inst_channels):
        if num_inst > 1:
            return inst_channels
        else:
            return 0
