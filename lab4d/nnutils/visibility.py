# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import torch
from torch import nn

from lab4d.nnutils.base import CondMLP
from lab4d.nnutils.embedding import PosEmbedding


class VisField(nn.Module):
    """Predict a visibility score (-inf to +inf) for all 3D points

    Args:
        num_inst (int): Number of distinct object instances. If --nosingle_inst
            is passed, this is equal to the number of videos, as we assume each
            video captures a different instance. Otherwise, we assume all videos
            capture the same instance and set this to 1.
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_xyz (int): Number of frequencies in position embedding
        inst_channels (int): Number of channels in the instance code
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        num_inst,
        D=2,
        W=64,
        num_freq_xyz=10,
        inst_channels=32,
        skips=[4],
        activation=nn.ReLU(True),
    ):
        super().__init__()

        # position and direction embedding
        self.pos_embedding = PosEmbedding(3, num_freq_xyz)

        # xyz encoding layers
        self.basefield = CondMLP(
            num_inst=num_inst,
            D=D,
            W=W,
            in_channels=self.pos_embedding.out_channels,
            inst_channels=inst_channels,
            out_channels=1,
            skips=skips,
            activation=activation,
            final_act=False,
        )

    def forward(self, xyz, inst_id=None):
        """
        Args:
            xyz: (..., 3), xyz coordinates
            inst_id: (...,) instance id, or None to use the average instance
        Returns:
            out: (..., 1), visibility score
        """
        xyz_embed = self.pos_embedding(xyz)
        visibility = self.basefield(xyz_embed, inst_id)
        return visibility
