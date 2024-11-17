# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import trimesh
from torch import nn
from torch.nn import functional as F

from lab4d.nnutils.base import CondMLP
from lab4d.nnutils.embedding import PosEmbedding, TimeEmbedding
from lab4d.utils.quat_transform import (
    dual_quaternion_to_quaternion_translation,
    quaternion_to_matrix,
)
from lab4d.utils.transforms import get_bone_coords
from lab4d.utils.vis_utils import get_colormap


class SkinningField(nn.Module):
    """Attaches 3D geometry to bones (either bag of bones or skeleton)

    Args:
        num_coords (int): Number of bones
        frame_info (FrameInfo): Metadata about the frames in a dataset
        num_inst (int): Number of distinct object instances. If --nosingle_inst
            is passed, this is equal to the number of videos, as we assume each
            video captures a different instance. Otherwise, we assume all videos
            capture the same instance and set this to 1.
        D (int): Number of linear layers for delta skinning field
        W (int): Number of hidden units in each MLP layer
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_t (int): Number of frequencies in time Fourier embedding
        inst_channels (int): Number of channels in the instance code
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        init_scale (float): Initial scale factor / variance of Gaussian bones
        delta_skin (bool): Whether to apply a learned MLP delta on top of
            Gaussian skinning weights
        symm_idx (Dict(int, int) or None): If given, enforce bone symmetry by
            regularizing scale and bone length between left/right body parts.
            This is a dict mapping body parts from left->right and right->left
    """

    def __init__(
        self,
        num_coords,
        frame_info,
        num_inst,
        D=2,
        W=64,
        num_freq_xyz=0,
        num_freq_t=6,
        inst_channels=32,
        skips=[4],
        activation=nn.ReLU(True),
        init_scale=0.03,
        delta_skin=True,
        symm_idx=None,
    ):
        super().__init__()

        # 3D gaussians
        gaussians = init_scale * torch.ones(
            num_coords, 3
        )  # scale of bone skinning field
        self.log_gauss = nn.Parameter(torch.log(gaussians))
        self.num_coords = num_coords

        if delta_skin:
            # position and direction embedding
            self.pos_embedding = PosEmbedding(3 * num_coords, num_freq_xyz)
            self.time_embedding = TimeEmbedding(num_freq_t, frame_info)

            # xyz encoding layers
            self.delta_field = CondMLP(
                num_inst=num_inst,
                D=D,
                W=W,
                in_channels=self.pos_embedding.out_channels
                + self.time_embedding.out_channels,
                inst_channels=inst_channels,
                out_channels=num_coords,
                skips=skips,
                activation=activation,
                final_act=False,
            )

        self.symm_idx = symm_idx

    def forward(self, xyz, bone2obj, frame_id, inst_id):
        """Compute Gaussian skinning weights, modulated with time and
        instance dependent delta skinning weights

        Args:
            xyz: (M,N,D,3) Points in object canonical space
            bone2obj: ((M,N,D,B,4), (M,N,D,B,4)) Bone-to-object SE(3)
                transforms, written as dual quaternions
            frame_id: (M,) Frame id. If None, compute for all frames
            inst_id: (M,) Instance id. If None, compute for the mean instance
        Returns:
            skin: (M,N,D,B) Skinning weights from each point to each bone
                (unnormalized)
        """
        # gaussian weights (-inf, 0),  (M, N, D, K, 3)
        xyz_bone = self.get_gauss_bone_coords(xyz, bone2obj)
        dist2 = xyz_bone.pow(2).sum(dim=-1)

        if hasattr(self, "delta_field"):
            # modulate with t/inst
            xyz_embed = self.pos_embedding(xyz_bone.reshape(xyz.shape[:-1] + (-1,)))
            if frame_id is None:
                t_embed = self.time_embedding.get_mean_embedding(xyz.device)
            else:
                t_embed = self.time_embedding(frame_id)
            t_embed = t_embed.reshape(-1, 1, 1, t_embed.shape[-1])
            t_embed = t_embed.expand(xyz.shape[:-1] + (-1,))
            xyzt_embed = torch.cat([xyz_embed, t_embed], dim=-1)
            delta = self.delta_field(xyzt_embed, inst_id)
            delta = F.relu(delta) * 0.1
            skin = -(dist2 + delta)
        else:
            skin = -dist2
            delta = None

        return skin, delta

    def get_gauss_bone_coords(self, xyz, bone2obj):
        """Transform points from object canonical space to Gaussian bone coords,
        and apply Gaussian scale factor

        Args:
            xyz: (M,N,D,3) Points in object canonical space
            bone2obj: ((M,N,D,B,4), (M,N,D,B,4)) Bone-to-object SE(3)
                transforms, written as dual quaternions
        Returns:
            xyz_bone: (M,N,D,B,3) Points in bone space
        """
        ndim_xyz = len(xyz.shape)
        xyz_bone = get_bone_coords(xyz, bone2obj)
        gauss = self.get_gauss()
        gauss = gauss.view((1,) * (ndim_xyz - 1) + (-1, 3))
        xyz_bone = xyz_bone / gauss
        return xyz_bone

    def get_gauss(self):
        """Compute scale factor / variance of each Gaussian bone

        Returns:
            gauss: (B,3) Per bone xyz scale factor
        """
        log_gauss = self.log_gauss
        if self.symm_idx is not None:
            log_gauss = (log_gauss[self.symm_idx] + log_gauss) / 2
        return log_gauss.exp()

    def draw_gaussian(self, articulation, edges):
        """Visualize Gaussian bones as a mesh

        Args:
            articulation: ((B,4), (B,4)) Bone-to-object SE(3) transforms,
                written as dual quaternions
            edges (Dict(int, int) or None): If given, a mapping from each joint
                to its parent joint on an articulated skeleton
        """
        with torch.no_grad():
            meshes = []
            gaussians = self.get_gauss().cpu().numpy()

            qr, trans = dual_quaternion_to_quaternion_translation(articulation)
            articulation = np.eye(4, 4)[None].repeat(len(qr), axis=0)
            articulation[:, :3, :3] = quaternion_to_matrix(qr).cpu().numpy()
            articulation[:, :3, 3] = trans.cpu().numpy()

            # add bone center / joints
            sph = trimesh.creation.uv_sphere(radius=1, count=[4, 4])
            colormap = get_colormap(self.num_coords, repeat=sph.vertices.shape[0])
            for k, gauss in enumerate(gaussians):
                ellips = sph.copy()
                # make it smaller for visualization
                ellips.vertices *= 5e-3
                # ellips.vertices *= gauss[None]
                ellips.apply_transform(articulation[k])
                meshes.append(ellips)

            # add edges if any
            if edges is not None:
                # rad = gaussians.mean() * 0.1
                rad = 5e-4
                for idx, parent_idx in edges.items():
                    if parent_idx == 0:
                        continue
                    parent_center = articulation[parent_idx - 1][:3, 3]
                    child_center = articulation[idx - 1][:3, 3]
                    cyl = np.stack([parent_center, child_center], 0)
                    cyl = trimesh.creation.cylinder(rad, segment=cyl, sections=3)
                    meshes.append(cyl)

            meshes = trimesh.util.concatenate(meshes)
            colormap_pad = np.ones((meshes.vertices.shape[0] - colormap.shape[0], 3))
            colormap = np.concatenate([colormap, 192 * colormap_pad], 0)
            meshes.visual.vertex_colors = colormap
            return meshes
