# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import trimesh
from torch import nn

from lab4d.nnutils.base import BaseMLP
from lab4d.nnutils.embedding import PosEmbedding
from lab4d.nnutils.nerf import NeRF
from lab4d.utils.decorator import train_only_fields
from lab4d.utils.geom_utils import Kmatinv, pinhole_projection


class FeatureNeRF(NeRF):
    """A neural radiance field that renders features in addition to RGB

    Args:
        vid_info (Dict): Dataset metadata from get_data_info()
        D (int): Number of linear layers for density (sigma) encoder
        W (int): Number of hidden units in each MLP layer
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_dir (int): Number of frequencies in direction embedding
        appr_channels (int): Number of channels in the global appearance code
            (captures shadows, lighting, and other environmental effects)
        appr_num_freq_t (int): Number of frequencies in the time embedding of
            the global appearance code
        num_inst (int): Number of distinct object instances. If --nosingle_inst
            is passed, this is equal to the number of videos, as we assume each
            video captures a different instance. Otherwise, we assume all videos
            capture the same instance and set this to 1.
        inst_channels (int): Number of channels in the instance code
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        init_beta (float): Initial value of beta, from Eqn. 3 of VolSDF.
            We transform a learnable signed distance function into density using
            the CDF of the Laplace distribution with zero mean and beta scale.
        init_scale (float): Initial geometry scale factor
        color_act (bool): If True, apply sigmoid to the output RGB
        feature_channels (int): Number of feature field channels
    """

    def __init__(
        self,
        data_info,
        D=8,
        W=256,
        num_freq_xyz=10,
        num_freq_dir=4,
        appr_channels=32,
        appr_num_freq_t=6,
        num_inst=1,
        inst_channels=32,
        skips=[4],
        activation=nn.ReLU(True),
        init_beta=0.1,
        init_scale=0.1,
        color_act=True,
        feature_channels=16,
        opts=None,
    ):
        super().__init__(
            data_info,
            D=D,
            W=W,
            num_freq_xyz=num_freq_xyz,
            num_freq_dir=num_freq_dir,
            appr_channels=appr_channels,
            appr_num_freq_t=appr_num_freq_t,
            num_inst=num_inst,
            inst_channels=inst_channels,
            skips=skips,
            activation=activation,
            init_beta=init_beta,
            init_scale=init_scale,
            color_act=color_act,
            opts=opts,
        )

        self.feat_pos_embedding = PosEmbedding(3, N_freqs=6)  # lower frequency
        self.feature_field = BaseMLP(
            D=5,
            W=128,
            in_channels=self.feat_pos_embedding.out_channels,
            out_channels=feature_channels,
            final_act=False,
        )

        sigma = torch.tensor([1.0])
        self.logsigma = nn.Parameter(sigma.log())

    def query_field(self, samples_dict, flow_thresh=None):
        """Render outputs from a neural radiance field.

        Args:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,SE(3)), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,2), and
                "feature" (M,N,16)
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            feat_dict (Dict): Neural field outputs. Keys: "rgb" (M,N,D,3),
                "density" (M,N,D,1), "density_{fg,bg}" (M,N,D,1), "vis" (M,N,D,1),
                "cyc_dist" (M,N,D,1), "xyz" (M,N,D,3), "xyz_cam" (M,N,D,3),
                "depth" (M,1,D,1) TODO
            deltas: (M,N,D,1) Distance along rays between adjacent samples
            aux_dict (Dict): Auxiliary neural field outputs. Keys: TODO
        """
        feat_dict, deltas, aux_dict = super(FeatureNeRF, self).query_field(
            samples_dict, flow_thresh=flow_thresh
        )
        xyz = feat_dict["xyz"]
        field2cam = samples_dict["field2cam"]
        Kinv = samples_dict["Kinv"]
        frame_id = samples_dict["frame_id"]
        inst_id = samples_dict["inst_id"]

        # import ipdb; ipdb.set_trace()
        # samples feature field
        if not self.opts["two_branch"]:
            feat_field_dict = self.compute_feat(xyz)
            feat_dict.update(feat_field_dict)
        # global matching
        if "feature" in samples_dict and "feature" in feat_dict:

            feature = feat_dict["feature"]
            xyz_matches = self.global_match(samples_dict["feature"], feature, xyz)
            xy_reproj, xyz_reproj = self.forward_project(
                xyz_matches,
                field2cam,
                Kinv,
                frame_id,
                inst_id,
                samples_dict=samples_dict,
            )
            # import ipdb; ipdb.set_trace()
            # print("xyz_reproj mean", xyz_reproj.mean(dim=1))
            aux_dict["xyz_matches"] = xyz_matches
            aux_dict["xyz_reproj"] = xyz_reproj
            aux_dict["xy_reproj"] = xy_reproj
        return feat_dict, deltas, aux_dict

    @train_only_fields
    def compute_feat(self, xyz):
        """Render feature field

        Args:
            xyz: (M,N,D,3) Points in field coordinates
        Returns:
            feat_field_dict: Feature field. Keys: "feature" (M,N,D,16)
        """
        feat_field_dict = {}
        xyz_embed = self.feat_pos_embedding(xyz)
        feature = self.feature_field(xyz_embed)
        feature = feature / feature.norm(dim=-1, keepdim=True)
        feat_field_dict["feature"] = feature
        return feat_field_dict

    def global_match(
        self,
        feat_px,
        feat_canonical,
        xyz_canonical,
        num_candidates=1024,
        num_grad=128,
    ):
        """Match pixel features to canonical features, which combats local
        minima in differentiable rendering optimization

        Args:
            feat: (M,N,feature_channels) Pixel features
            feat_canonical: (M,N,D,feature_channels) Canonical features
            xyz_canonical: (M,N,D,3) Canonical points
        Returns:
            xyz_matched: (M,N,3) Matched xyz
        """
        shape = feat_px.shape
        feat_px = feat_px.view(-1, shape[-1])  # (M*N, feature_channels)
        feat_canonical = feat_canonical.view(-1, shape[-1])  # (M*N*D, feature_channels)
        xyz_canonical = xyz_canonical.view(-1, 3)  # (M*N*D, 3)

        # sample canonical points
        num_candidates = min(num_candidates, feat_canonical.shape[0])
        idx = torch.randperm(feat_canonical.shape[0])[:num_candidates]
        feat_canonical = feat_canonical[idx]  # (num_candidates, feature_channels)
        xyz_canonical = xyz_canonical[idx]  # (num_candidates, 3)

        # compute similarity
        score = torch.matmul(feat_px, feat_canonical.t())  # (M*N, num_candidates)

        # # find top K candidates
        # num_grad = min(num_grad, score.shape[1])
        # score, idx = torch.topk(score, num_grad, dim=1, largest=True)
        # score = score * self.logsigma.exp()  # temperature

        # # soft argmin
        # prob = torch.softmax(score, dim=1)
        # xyz_matched = torch.sum(prob.unsqueeze(-1) * xyz_canonical[idx], dim=1)

        # use all candidates
        score = score * self.logsigma.exp()  # temperature
        prob = torch.softmax(score, dim=1)
        xyz_matched = torch.sum(prob.unsqueeze(-1) * xyz_canonical, dim=1)

        xyz_matched = xyz_matched.view(shape[:-1] + (-1,))
        # import ipdb; ipdb.set_trace()
        return xyz_matched

    def forward_project(self, xyz, field2cam, Kinv, frame_id, inst_id, samples_dict={}):
        """Project xyz to image plane

        Args:
            xyz: (M,N,3) Points in field coordinates
            Kinv: (M,3,3) Inverse of camera intrinsics
            field2cam: (M,1,1,4,4) Field to camera transformation
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance

        Returns:
            xy: (M,N,2) Points in image plane
        """
        # TODO: make the format consistent
        xyz = xyz[:, :, None]
        # inst_id = inst_id[..., :1]
        # transform xyz to camera coordinates
        xyz_cam = self.forward_warp(
            xyz, field2cam, frame_id, inst_id, samples_dict=samples_dict
        )
        xyz_cam = xyz_cam[:, :, 0]

        # project
        Kmat = Kmatinv(Kinv)
        xy_reproj = pinhole_projection(Kmat, xyz_cam)[..., :2]
        return xy_reproj, xyz_cam
