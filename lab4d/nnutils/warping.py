# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import torch
import trimesh
from torch import nn
from torch.nn import functional as F

from lab4d.nnutils.base import CondMLP
from lab4d.nnutils.embedding import PosEmbedding, TimeEmbedding
from lab4d.nnutils.pose import ArticulationFlatMLP, ArticulationSkelMLP
from lab4d.nnutils.skinning import SkinningField
from lab4d.third_party.nvp import NVP
from lab4d.utils.geom_utils import dual_quaternion_skinning, marching_cubes, extend_aabb
from lab4d.utils.quat_transform import dual_quaternion_inverse, dual_quaternion_mul
from lab4d.utils.transforms import get_xyz_bone_distance, get_bone_coords
from lab4d.utils.loss_utils import entropy_loss, cross_entropy_skin_loss

from lab4d.utils.quat_transform import (
    dual_quaternion_to_quaternion_translation,
    quaternion_to_matrix,
    quaternion_translation_inverse,
    axis_angle_to_quaternion
)

def create_warp(fg_motion, data_info):
    """Construct a warping field.

    Args:
        fg_motion (str): Foreground motion type ("rigid", "dense", "bob",
            "skel-{human,quad}", or "comp_skel-{human,quad}_{bob,dense}")
        data_info (Dict): Dataset metadata from get_data_info()
    Returns:
        warp: An IdentityWarp, DenseWarp, SkinningWarp, or ComposedWarp
    """
    # joint angles initialization
    frame_info = data_info["frame_info"]
    if "joint_angles" in data_info.keys():
        joint_angles = data_info["joint_angles"]
    else:
        joint_angles = None

    if fg_motion == "rigid":
        warp = IdentityWarp(frame_info)
    elif fg_motion == "dense":
        warp = DenseWarp(frame_info)
    elif fg_motion == "denseSE3":
        warp = DenseWarpSE3(frame_info)
    elif fg_motion == "bob":
        warp = SkinningWarp(frame_info)
    elif fg_motion == "bob-nosoft":
        warp = SkinningWarp(frame_info, delta_skin=False)
    elif fg_motion == "bob-lesssoft":
        warp = SkinningWarp(frame_info, delta_skin=False, num_freq_xyz=10,num_freq_t=6)
    elif fg_motion == "bob-sc":
        warp = SkinningWarp(frame_info, delta_skin=False, num_se3=100)
    elif fg_motion.startswith("skel-"):
        warp = SkinningWarp(
            frame_info,
            skel_type=fg_motion.split("-")[1],
            joint_angles=joint_angles,
        )
    elif fg_motion.startswith("comp"):
        warp = ComposedWarp(
            data_info,
            frame_info,
            warp_type=fg_motion,
            joint_angles=joint_angles,
        )
    else:
        raise NotImplementedError
    return warp


class IdentityWarp(nn.Module):
    """A rigid warp with no deformation.

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_t (int): Number of frequencies in time embedding
    """

    def __init__(self, frame_info, num_freq_xyz=10, num_freq_t=6):
        super().__init__()
        self.num_frames = frame_info["frame_offset"][-1]
        self.num_inst = len(frame_info["frame_offset"]) - 1

    def forward(
        self, xyz, frame_id, inst_id, backward=False, samples_dict={}, return_aux=False
    ):
        """
        Args:
            xyz: (M,N,D,3) Points in object canonical space
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            backward (bool): Forward (=> deformed) or backward (=> canonical)
            samples_dict (Dict): Only used for SkeletonWarp
        Returns:
            xyz: (M,N,D,3) Warped xyz coordinates
        """
        warp_dict = {}
        if return_aux:
            return xyz, warp_dict
        else:
            return xyz

    def get_template_vis(self, aabb):
        """Visualize Gaussian density and SDF as meshes.

        Args:
            aabb: (2,3) Axis-aligned bounding box
        Returns:
            mesh_gauss (Trimesh): Gaussian density mesh
            mesh_sdf (Trimesh): SDF mesh
        """
        mesh = trimesh.Trimesh()
        return mesh, mesh


class DenseWarp(IdentityWarp):
    """Predict dense translation fields, using separate MLPs for forward and
    backward warping. Used by DNeRF.

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_t (int): Number of frequencies in time embedding
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
    """

    def __init__(self, frame_info, num_freq_xyz=6, num_freq_t=6, D=6, W=256):
        super().__init__(
            frame_info=frame_info, num_freq_xyz=num_freq_xyz, num_freq_t=num_freq_t
        )

        self.pos_embedding = PosEmbedding(3, num_freq_xyz)
        self.time_embedding = TimeEmbedding(num_freq_t, frame_info)

        self.forward_map = CondMLP(
            self.num_inst,
            D=D,
            W=W,
            in_channels=self.pos_embedding.out_channels
            + self.time_embedding.out_channels,
            out_channels=3,
        )

        self.backward_map = CondMLP(
            self.num_inst,
            D=D,
            W=W,
            in_channels=self.pos_embedding.out_channels
            + self.time_embedding.out_channels,
            out_channels=3,
        )

    def forward(
        self, xyz, frame_id, inst_id, backward=False, samples_dict={}, return_aux=False
    ):
        """
        Args:
            xyz: (M,N,D,3) Points in object canonical space
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            backward (bool): Forward (=> deformed) or backward (=> canonical)
            samples_dict (Dict): Only used for SkeletonWarp
        Returns:
            xyz: (M,N,D,3) Warped xyz coordinates
        """
        xyz_embed = self.pos_embedding(xyz)
        t_embed = self.time_embedding(frame_id)
        t_embed = t_embed.reshape(-1, 1, 1, t_embed.shape[-1])
        t_embed = t_embed.expand(xyz.shape[:-1] + (-1,))
        embed = torch.cat([xyz_embed, t_embed], dim=-1)
        if backward:
            motion = self.backward_map(embed, inst_id)
        else:
            motion = self.forward_map(embed, inst_id)
        out = xyz + motion * 0.1  # control the scale
        warp_dict = {}
        if return_aux:
            return out, warp_dict
        else:
            return out

class DenseWarpSE3(IdentityWarp):
    """for 4DGS.

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_t (int): Number of frequencies in time embedding
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
    """

    def __init__(self, frame_info, num_freq_xyz=6, num_freq_t=6, D=6, W=256):
        super().__init__(
            frame_info=frame_info, num_freq_xyz=num_freq_xyz, num_freq_t=num_freq_t
        )

        self.pos_embedding = PosEmbedding(3, num_freq_xyz)
        self.time_embedding = TimeEmbedding(num_freq_t, frame_info)
        self.trans_scaling = torch.nn.Parameter(torch.tensor([0.1]))

        self.forward_map_trans = CondMLP(
            self.num_inst,
            D=D,
            W=W // 2,
            in_channels=self.pos_embedding.out_channels
            + self.time_embedding.out_channels,
            out_channels=3,
        )

        self.forward_map_rot = CondMLP(
            self.num_inst,
            D=D,
            W=W // 2,
            in_channels=self.pos_embedding.out_channels
            + self.time_embedding.out_channels,
            out_channels=3,
        )

        # self.backward_map = CondMLP(
        #     self.num_inst,
        #     D=D,
        #     W=W,
        #     in_channels=self.pos_embedding.out_channels
        #     + self.time_embedding.out_channels,
        #     out_channels=3,
        # )


    def forward(
        self, xyz, frame_id, inst_id, backward=False, samples_dict={}, return_aux=False, return_qt=False,
    ):
        """
        Args:
            xyz: (M,N,D,3) Points in object canonical space
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            backward (bool): Forward (=> deformed) or backward (=> canonical)
            samples_dict (Dict): Only used for SkeletonWarp
        Returns:
            xyz: (M,N,D,3) Warped xyz coordinates
        """
        xyz_embed = self.pos_embedding(xyz)
        t_embed = self.time_embedding(frame_id)
        t_embed = t_embed.reshape(-1, 1, 1, t_embed.shape[-1])
        t_embed = t_embed.expand(xyz.shape[:-1] + (-1,))
        embed = torch.cat([xyz_embed, t_embed], dim=-1)

        trans = self.forward_map_trans(embed, inst_id) * self.trans_scaling
        so3 = self.forward_map_rot(embed, inst_id)
        qr = axis_angle_to_quaternion(so3)
        if backward:
            qr, trans = quaternion_translation_inverse(qr, trans)

        if return_qt:
            out = (qr, trans)
        else:
            out = xyz + trans  # control the scale
        warp_dict = {}
        if return_aux:
            return out, warp_dict
        else:
            return out

class NVPWarp(IdentityWarp):
    """Predict dense translation fields, using a single invertible MLP for
    forward and backward warping. Used by RealNVP, CaDeX.

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_t (int): Number of frequencies in time embedding
        D (int): Number of hidden layers
    """

    def __init__(self, frame_info, num_freq_xyz=6, num_freq_t=6, D=2):
        super().__init__(
            frame_info=frame_info, num_freq_xyz=num_freq_xyz, num_freq_t=num_freq_t
        )
        self.time_embedding = TimeEmbedding(num_freq_t, frame_info)

        self.map = NVP(
            n_layers=D,
            feature_dims=self.time_embedding.out_channels,
            hidden_size=[32, 16, 16, 8, 8],
            proj_dims=32,
            code_proj_hidden_size=[32, 32, 32],
            proj_type="simple",
            block_normalize=False,
            normalization=False,
        )

    def forward(
        self, xyz, frame_id, inst_id, backward=False, samples_dict={}, return_aux=False
    ):
        """
        Args:
            xyz: (M,N,D,3) Points in object canonical space
            frame_id: (M) Frame id. If None, warp for all frames
            inst_id: (M) Instance id. If None, warp for the average instance
            backward (bool): Forward (=> deformed) or backward (=> canonical)
            samples_dict (Dict): Only used for SkeletonWarp
        Returns:
            out: (..., 3) Warped xyz coordinates
        """
        t_embed = self.time_embedding(frame_id)
        t_embed = t_embed.reshape(-1, 1, 1, t_embed.shape[-1])
        t_embed = t_embed.expand(xyz.shape[:-1] + (-1,))  # (M, N, D, x)
        t_embed = t_embed[:, 0]  # (M, D, x) vs (M, N, D, 3)
        if backward:
            out = self.map.inverse(t_embed, xyz)
        else:
            out = self.map.forward(t_embed, xyz)
        warp_dict = {}
        if return_aux:
            return out, warp_dict
        else:
            return out


class SkinningWarp(IdentityWarp):
    """Neural blend skinning warping field. Supports bag of bones (bob) or skeleton

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        skel_type (str): Skeleton type ("flat", "skel-human", or "skel-quad")
        joint_angles: (B, 3) If provided, initial joint angles
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_t (int): Number of frequencies in time embedding
        num_se3 (int): Number of bones
        init_gauss_scale (float): Initial scale/variance of the Gaussian bones
        init_beta (float): Initial transparency for bone rendering
    """

    def __init__(
        self,
        frame_info,
        skel_type="flat",
        joint_angles=None,
        num_freq_xyz=10,
        num_freq_t=6,
        num_se3=25,
        init_gauss_scale=0.03,
        init_beta=0.01,
        delta_skin=True,
    ):
        super().__init__(
            frame_info=frame_info, num_freq_xyz=num_freq_xyz, num_freq_t=num_freq_t
        )
        if skel_type == "flat":
            if num_se3 <50:
                self.articulation = ArticulationFlatMLP(frame_info, num_se3)
            else:
                self.articulation = ArticulationFlatMLP(frame_info, num_se3, D=2, W=32)
            symm_idx = None
        else:
            self.articulation = ArticulationSkelMLP(frame_info, skel_type, joint_angles)
            num_se3 = self.articulation.num_se3
            symm_idx = self.articulation.symm_idx

        self.skinning_model = SkinningField(
            num_se3,
            frame_info,
            self.num_inst,
            init_scale=init_gauss_scale,
            symm_idx=symm_idx,
            delta_skin=delta_skin,
        )

        # beta: transparency for bone rendering
        beta = torch.tensor([init_beta])
        self.logibeta = nn.Parameter(-beta.log())  # beta: transparency

    def forward(
        self, xyz, frame_id, inst_id, backward=False, samples_dict={}, return_aux=False, return_qt=False,
    ):
        """Warp points according to a skinning field and articulated bones

        Args:
            xyz: (M,N,D,3) Points in object canonical space
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the mean instance
            backward (bool): Forward (=> deformed) or backward (=> canonical)
            samples_dict: Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            out: (M,N,D,3) Warped xyz coordinates
        """
        # import ipdb; ipdb.set_trace()
        # print("xyz", xyz.shape, "frame_id", frame_id.shape)
        # compute part-to-object space transformations
        if "rest_articulation" in samples_dict and "t_articulation" in samples_dict:
            rest_articulation = samples_dict["rest_articulation"]
            t_articulation = samples_dict["t_articulation"]
        else:
            (
                t_articulation,
                rest_articulation,
            ) = self.articulation.get_vals_and_mean(frame_id)

        # compute per bone se3
        if backward:
            se3 = dual_quaternion_mul(
                rest_articulation, dual_quaternion_inverse(t_articulation)
            )
            articulation = t_articulation
        else:
            se3 = dual_quaternion_mul(
                t_articulation, dual_quaternion_inverse(rest_articulation)
            )
            articulation = rest_articulation
            frame_id = None
        # import ipdb; ipdb.set_trace()
        articulation = (
            articulation[0][:, None, None].expand(xyz.shape[:3] + (-1, -1)),
            articulation[1][:, None, None].expand(xyz.shape[:3] + (-1, -1)),
        )

        # skinning weights
        skin, delta_skin = self.skinning_model(xyz, articulation, frame_id, inst_id)
        skin_prob = skin.softmax(-1)

        out = dual_quaternion_skinning(se3, xyz, skin_prob, return_qt=return_qt)
        # if return_qt:
        #     out = dual_quaternion_skinning_return_qt(se3, xyz, skin_prob)
        # else:
        #     # left-multiply per-point se3
        #     out = dual_quaternion_skinning(se3, xyz, skin_prob)


        warp_dict = {}
        warp_dict["skin_entropy"] = cross_entropy_skin_loss(skin)[..., None]
        if delta_skin is not None:
            # (M, N, D, 1)
            warp_dict["delta_skin"] = delta_skin.pow(2).mean(-1, keepdims=True)
        if return_aux:
            return out, warp_dict
        else:
            return out

    def get_gauss_sdf(self, xyz, bias=0.0):
        """Calculate signed distance to Gaussian bones

        Args:
            xyz: (N, 3) Points in object canonical space
            bias (float): Value to add to signed distances, to control the
                surface to be different from the skinning weights
        Returns:
            sdf: (N,) Signed distance to Gaussian bones, -inf to +inf
        """
        density = self.get_gauss_density(xyz)
        density = density.clamp(1e-6, 1 - 1e-6)
        sdf = -density.logit()  # (N,), -inf, inf
        # control the surface to be different from skinning weights
        sdf = sdf + bias
        return sdf

    def get_gauss_density(self, xyz, bone2obj=None):
        """Sample volumetric density at Gaussian bones

        Args:
            xyz: (M,3) Points in object canonical space
            bone2obj: ((M,B,4), (M,B,4)) Bone-to-object SE(3) transforms,
                written as dual quaternions
        Returns:
            density: (M,3) Volumetric density of bones at each point
        """
        if bone2obj is None:
            bone2obj = self.articulation.get_mean_vals()  # 1,K,4,4

        dist2 = get_xyz_bone_distance(xyz, bone2obj)  # N,K
        dist2 = dist2 / (0.01) ** 2  # assuming spheres of radius 0.01

        # # gauss bones
        # xyz = xyz[:, None, None]  # (N,1,1,3)
        # bone2obj = (
        #     bone2obj[0][None, None].repeat(xyz.shape[0], 1, 1, 1, 1),
        #     bone2obj[1][None, None].repeat(xyz.shape[0], 1, 1, 1, 1),
        # )  # (N,1,1,K,4)
        # dist2 = -self.skinning_model.forward(
        #     xyz, bone2obj, None, None, normalize=False
        # )[0][:, 0, 0]

        score = (-0.5 * dist2).exp()  # (N,K)

        # hard selection
        density = score.max(-1)[0]  # (N,)

        density = density[..., None]
        return density

    def get_template_vis(self, aabb):
        """Visualize Gaussian density and SDF as meshes.

        Args:
            aabb: (2,3) Axis-aligned bounding box
        Returns:
            mesh_gauss (Trimesh): Gaussian density mesh
            mesh_sdf (Trimesh): SDF mesh
        """
        articulation = self.articulation.get_mean_vals()  # (1,K,4,4)
        articulation = (articulation[0][0], articulation[1][0])
        mesh_gauss = self.skinning_model.draw_gaussian(
            articulation, self.articulation.edges
        )

        sdf_func = lambda xyz: self.get_gauss_sdf(xyz)
        mesh_sdf = marching_cubes(sdf_func, aabb, level=0.005)
        return mesh_gauss, mesh_sdf


class ComposedWarp(SkinningWarp):
    """Compose a skeleton warp and soft warp. The skeleton warp handles large
    articulations and the soft warp handles fine details (e.g. ears)

    Args:
        vid_info (Dict): Dataset metadata from get_data_info()
        frame_info (FrameInfo): Metadata about the frames in a dataset
        warp_type (str): Type of warp: "comp_skel-{human,quad}_{bob,dense}"
        joint_angles: (B, 3) If provided, initial joint angles
    """

    def __init__(
        self,
        data_info,
        frame_info,
        warp_type,
        joint_angles=None,
    ):
        # e.g., comp_skel-human_dense, limited to skel+another type of field
        type_list = warp_type.split("_")[1:]
        assert len(type_list) == 2
        assert type_list[0] in ["skel-human", "skel-quad"]
        assert type_list[1] in ["bob", "dense"]
        if type_list[1] == "bob":
            raise NotImplementedError

        super().__init__(
            frame_info,
            skel_type=type_list[0].split("-")[1],
            joint_angles=joint_angles,
        )
        # self.post_warp = DenseWarp(frame_info, D=2, W=64)
        # self.post_warp = DenseWarp(frame_info, D=2, W=128)
        self.post_warp = DenseWarp(frame_info, D=2, W=256)
        # self.post_warp = NVPWarp(frame_info)

    def forward(
        self, xyz, frame_id, inst_id, backward=False, samples_dict={}, return_aux=False
    ):
        """Warp points according to a skinning field and articulated bones

        Args:
            xyz: (M,N,D,3) Points in object canonical space
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the mean instance
            backward (bool): Forward (=> deformed) or backward (=> canonical)
            samples_dict: Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            out: (M,N,D,3) Warped xyz coordinates
        """
        # if forward, and has frame_id
        if not backward and frame_id is not None:
            xyz = self.post_warp.forward(
                xyz, frame_id, inst_id, backward=False, samples_dict=samples_dict
            )

        out, warp_dict = super().forward(
            xyz,
            frame_id,
            inst_id,
            backward=backward,
            samples_dict=samples_dict,
            return_aux=True,
        )

        if backward and frame_id is not None:
            out = self.post_warp.forward(
                out, frame_id, inst_id, backward=True, samples_dict=samples_dict
            )
        if return_aux:
            return out, warp_dict
        else:
            return out

    def compute_post_warp_dist2(self, xyz, frame_id, inst_id):
        """Compute L2 distance between points before soft deformation and points
        after soft deformation

        Args:
            xyz: (M, ..., 3) Points after skeleton articulation
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the mean instance
        Returns:
            dist2: (M, ...) Squared soft deformation distance
        """
        xyz_t = self.post_warp.forward(xyz, frame_id, inst_id, backward=False)
        dist2 = (xyz_t - xyz).pow(2).sum(-1)

        # additional cycle consistency regularization for soft deformation
        if isinstance(self.post_warp, DenseWarp):
            xyz_back = self.post_warp.forward(xyz_t, frame_id, inst_id, backward=True)
            dist2 = (dist2 + (xyz_t - xyz_back).pow(2).sum(-1)) * 0.5
        return dist2
