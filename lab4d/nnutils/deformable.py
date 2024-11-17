# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import trimesh
from torch import nn
from torch.nn import functional as F
from pysdf import SDF
import trimesh.transformations as tf

from lab4d.nnutils.feature import FeatureNeRF
from lab4d.nnutils.warping import SkinningWarp, create_warp
from lab4d.utils.decorator import train_only_fields
from lab4d.utils.geom_utils import extend_aabb
from lab4d.utils.loss_utils import align_vectors
from lab4d.engine.train_utils import get_local_rank


class Deformable(FeatureNeRF):
    """A dynamic neural radiance field

    Args:
        fg_motion (str): Foreground motion type ("rigid", "dense", "bob",
            "skel-{human,quad}", or "comp_skel-{human,quad}_{bob,dense}")
        data_info (Dict): Dataset metadata from get_data_info()
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
        skips (List(int): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        init_beta (float): Initial value of beta, from Eqn. 3 of VolSDF.
            We transform a learnable signed distance function into density using
            the CDF of the Laplace distribution with zero mean and beta scale.
        init_scale (float): Initial geometry scale factor.
        color_act (bool): If True, apply sigmoid to the output RGB
        feature_channels (int): Number of feature field channels
    """

    def __init__(
        self,
        fg_motion,
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
            feature_channels=feature_channels,
            opts=opts,
        )

        self.warp = create_warp(fg_motion, data_info)
        self.fg_motion = fg_motion

    def init_proxy(self, geom_path, init_scale):
        """Initialize proxy geometry as a sphere

        Args:
            geom_path (str): Unused
            init_scale (float): Unused
        """
        geom_path = "/pfs/mt-1oY5F7/liuguangce/program-vidu/4dcode-8.3/objs for lab4d-neus/monster/tmpbrv8we0o.obj"
        init_scale = 0.1
        mesh = trimesh.load(geom_path)
        rotation_matrix = tf.euler_matrix(-np.pi, 0, 0)
        mesh.apply_transform(rotation_matrix)
        mesh.vertices = mesh.vertices * init_scale
        self.init_geometry = mesh
        self.proxy_geometry = trimesh.creation.uv_sphere(radius=0.12, count=[4, 4])

    def get_init_sdf_fn(self, mode='sphere'):
        """Initialize signed distance function as a skeleton or sphere

        Returns:
            sdf_fn_torch (Function): Signed distance function
        """
        if mode=='sphere':
            sdf_fn_numpy = SDF(self.proxy_geometry.vertices, self.proxy_geometry.faces)
        else:
            sdf_fn_numpy = SDF(self.init_geometry.vertices, self.init_geometry.faces)

        def sdf_fn_torch(pts):
            sdf = -sdf_fn_numpy(pts.cpu().numpy())[:, None]  # negative inside
            sdf = torch.tensor(sdf, device=pts.device, dtype=pts.dtype)
            return sdf

        def sdf_fn_torch_sphere(pts):
            radius = 0.1
            # l2 distance to a unit sphere
            dis = (pts).pow(2).sum(-1, keepdim=True)
            sdf = torch.sqrt(dis) - radius  # negative inside, postive outside
            return sdf

        @torch.no_grad()
        def sdf_fn_torch_skel(pts):
            sdf = self.warp.get_gauss_sdf(pts)
            return sdf

        # if "skel-" in self.fg_motion:
        #     return sdf_fn_torch_skel
        # else:
        if mode=='sphere':
            return sdf_fn_torch_sphere
        else:
            return sdf_fn_torch
    
    def backward_warp(
        self, xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict={}
    ):
        """Warp points from camera space to object canonical space. This
        requires "un-articulating" the object from observed time-t to rest.

        Args:
            xyz_cam: (M,N,D,3) Points along rays in camera space
            dir_cam: (M,N,D,3) Ray directions in camera space
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance.
            samples_dict (Dict): Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            xyz: (M,N,D,3) Points along rays in object canonical space
            dir: (M,N,D,3) Ray directions in object canonical space
            xyz_t: (M,N,D,3) Points along rays in object time-t space.
        """
        xyz_t, dir = self.cam_to_field(xyz_cam, dir_cam, field2cam)
        xyz, warp_dict = self.warp(
            xyz_t,
            frame_id,
            inst_id,
            backward=True,
            samples_dict=samples_dict,
            return_aux=True,
        )

        # TODO: apply se3 to dir
        backwarp_dict = {"xyz": xyz, "dir": dir, "xyz_t": xyz_t}
        backwarp_dict.update(warp_dict)
        return backwarp_dict

    def forward_warp(self, xyz, field2cam, frame_id, inst_id, samples_dict={}):
        """Warp points from object canonical space to camera space. This
        requires "re-articulating" the object from rest to observed time-t.

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            samples_dict (Dict): Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            xyz_cam: (M,N,D,3) Points along rays in camera space
        """

        xyz_next = self.warp(xyz, frame_id, inst_id, samples_dict=samples_dict)
        xyz_cam = self.field_to_cam(xyz_next, field2cam)
        return xyz_cam

    @train_only_fields
    def cycle_loss(self, xyz, xyz_t, frame_id, inst_id, samples_dict={}):
        """Enforce cycle consistency between points in object canonical space,
        and points warped from canonical space, backward to time-t space, then
        forward to canonical space again

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space
            xyz_t: (M,N,D,3) Points along rays in object time-t space
            frame_id: (M,) Frame id. If None, render at all frames
            inst_id: (M,) Instance id. If None, render for the average instance
            samples_dict (Dict): Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            cyc_dict (Dict): Cycle consistency loss. Keys: "cyc_dist" (M,N,D,1)
        """
        cyc_dict = super().cycle_loss(xyz, xyz_t, frame_id, inst_id, samples_dict)

        xyz_cycled, warp_dict = self.warp(
            xyz, frame_id, inst_id, samples_dict=samples_dict, return_aux=True
        )
        cyc_dist = (xyz_cycled - xyz_t).norm(2, -1, keepdim=True)
        cyc_dict["cyc_dist"] = cyc_dist
        cyc_dict.update(warp_dict)
        return cyc_dict

    def gauss_skin_consistency_loss(self, nsample=2048):
        """Enforce consistency between the NeRF's SDF and the SDF of Gaussian bones

        Args:
            nsample (int): Number of samples to take from both distance fields
        Returns:
            loss: (0,) Skinning consistency loss
        """
        pts = self.sample_points_aabb(nsample, extend_factor=0.25)

        # match the gauss density to the reconstructed density   
        # 其实就是pts在权值最大的bone处的高斯分布的密度
        density_gauss = self.warp.get_gauss_density(pts)  # (N,1) # 这是规范空间的骨骼高斯的概率密度
        with torch.no_grad():
            density = self.forward(pts, inst_id=None, get_density=True)
            density = density / self.logibeta.exp()  # (0,1) # 这是每一帧利用mlp得到的密度
       
        # binary cross entropy loss to align gauss density to the reconstructed density
        # weight the loss such that:
        # wp lp = wn ln
        # wp lp + wn ln = lp + ln
        weight_pos = 0.5 / (1e-6 + density.mean())
        weight_neg = 0.5 / (1e-6 + 1 - density).mean()
        weight = density * weight_pos + (1 - density) * weight_neg
        # loss = ((density_gauss - density).pow(2) * weight.detach()).mean()
        loss = F.binary_cross_entropy(
            density_gauss, density.detach(), weight=weight.detach()
        )

        # if get_local_rank() == 0:
        #     is_inside = density > 0.5
        #     mesh = trimesh.Trimesh(vertices=pts[is_inside[..., 0]].detach().cpu())
        #     mesh.export("tmp/0.obj")

        #     is_inside = density_gauss > 0.5
        #     mesh = trimesh.Trimesh(vertices=pts[is_inside[..., 0]].detach().cpu())
        #     mesh.export("tmp/1.obj")
        return loss

    def soft_deform_loss(self, nsample=1024):
        """Minimize soft deformation so it doesn't overpower the skeleton.
        Compute L2 distance of points before and after soft deformation

        Args:
            nsample (int): Number of samples to take from both distance fields
        Returns:
            loss: (0,) Soft deformation loss
        """
        device = next(self.parameters()).device
        pts = self.sample_points_aabb(nsample, extend_factor=1.0)
        frame_id = torch.randint(0, self.num_frames, (nsample,), device=device)
        inst_id = torch.randint(0, self.num_inst, (nsample,), device=device)
        dist2 = self.warp.compute_post_warp_dist2(pts[:, None, None], frame_id, inst_id)
        return dist2.mean()

    def get_samples(self, Kinv, batch):
        """Compute time-dependent camera and articulation parameters.

        Args:
            Kinv: (N,3,3) Inverse of camera matrix
            Batch (Dict): Batch of inputs. Keys: "dataid", "frameid_sub",
                "crop2raw", "feature", "hxy", and "frameid"
        Returns:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,SE(3)), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,2),
                "feature" (M,N,16), "rest_articulation" ((M,B,4), (M,B,4)), and
                "t_articulation" ((M,B,4), (M,B,4))
        """
        samples_dict = super().get_samples(Kinv, batch)

        if isinstance(self.warp, SkinningWarp):
            # cache the articulation values
            # mainly to avoid multiple fk computation
            # (M,K,4)x2, # (M,K,4)x2
            inst_id = samples_dict["inst_id"]
            frame_id = samples_dict["frame_id"]
            if "joint_so3" in batch.keys():
                override_so3 = batch["joint_so3"]
                samples_dict[
                    "rest_articulation"
                ] = self.warp.articulation.get_mean_vals()
                samples_dict["t_articulation"] = self.warp.articulation.get_vals(
                    frame_id, override_so3=override_so3
                )
            else:
                (
                    samples_dict["t_articulation"],
                    samples_dict["rest_articulation"],
                ) = self.warp.articulation.get_vals_and_mean(frame_id)

        return samples_dict

    def mlp_init(self):
        """For skeleton fields, initialize bone lengths and rest joint angles
        from an external skeleton
        """
        super().mlp_init()
        if self.fg_motion.startswith("skel"):
            if hasattr(self.warp.articulation, "init_vals"):
                self.warp.articulation.mlp_init()

    def query_field(self, samples_dict, flow_thresh=None):
        """Render outputs from a neural radiance field.

        Args:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,SE(3)), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,2), and
                "feature" (M,N,16), "rest_articulation" ((M,B,4), (M,B,4)),
                and "t_articulation" ((M,B,4), (M,B,4))
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            feat_dict (Dict): Neural field outputs. Keys: "rgb" (M,N,D,3),
                "density" (M,N,D,1), "density_{fg,bg}" (M,N,D,1), "vis" (M,N,D,1),
                "cyc_dist" (M,N,D,1), "xyz" (M,N,D,3), "xyz_cam" (M,N,D,3),
                "depth" (M,1,D,1) TODO
            deltas: (M,N,D,1) Distance along rays between adjacent samples
            aux_dict (Dict): Auxiliary neural field outputs. Keys: TODO
        """
        feat_dict, deltas, aux_dict = super().query_field(
            samples_dict, flow_thresh=flow_thresh
        )
        if not self.opts["two_branch"]:
        # xyz = feat_dict["xyz"].detach()  # don't backprop to cam/dfm fields
            xyz = feat_dict["xyz"]
            gauss_field = self.compute_gauss_density(xyz, samples_dict)
            feat_dict.update(gauss_field)
        # import ipdb; ipdb.set_trace()
        return feat_dict, deltas, aux_dict

    def compute_gauss_density(self, xyz, samples_dict):
        """If this is a SkinningWarp, compute density from Gaussian bones

        Args:
            xyz: (M,N,D,3) Points in object canonical space
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,SE(3)), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,2), and
                "feature" (M,N,16), "rest_articulation" ((M,B,4), (M,B,4)),
                and "t_articulation" ((M,B,4), (M,B,4))
        Returns:
            gauss_field (Dict): Density. Keys: "gauss_density" (M,N,D,1)
        """
        gauss_field = {}
        if isinstance(self.warp, SkinningWarp):
            shape = xyz.shape[:-1]
            if "rest_articulation" in samples_dict:
                rest_articulation = (
                    samples_dict["rest_articulation"][0][:1],
                    samples_dict["rest_articulation"][1][:1],
                )
            xyz = xyz.view(-1, 3)
            gauss_density = self.warp.get_gauss_density(xyz, bone2obj=rest_articulation)
            # gauss_density = gauss_density * 100  # [0,100] heuristic value
            gauss_density = gauss_density * self.warp.logibeta.exp()
            gauss_field["gauss_density"] = gauss_density.view(shape + (1,))

        return gauss_field
