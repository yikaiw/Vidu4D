# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import imp
from re import T
from altair import renderers
import numpy as np
from regex import R
from sympy import re
import torch
import torch.nn.functional as F
import trimesh
from pysdf import SDF
from torch import nn
import pytorch3d.ops as ops
from torchvision.utils import save_image
from lab4d.nnutils.appearance import AppearanceEmbedding
from lab4d.nnutils.base import CondMLP
from lab4d.nnutils.embedding import PosEmbedding
from lab4d.nnutils.pose import CameraMLP
from lab4d.nnutils.visibility import VisField
from lab4d.utils.decorator import train_only_fields
from lab4d.utils.geom_utils import (
    Kmatinv,
    apply_se3mat,
    extend_aabb,
    get_near_far,
    marching_cubes,
    pinhole_projection,
    check_inside_aabb,
)
from lab4d.utils.loss_utils import align_vectors
from lab4d.utils.quat_transform import (
    quaternion_apply,
    quaternion_translation_inverse,
    quaternion_translation_mul,
    quaternion_translation_to_se3,
    dual_quaternion_to_quaternion_translation,
    quaternion_translation_apply,
    quaternion_to_matrix,
    quaternion_mul
)
import pytorch3d
from lab4d.utils.render_utils import sample_cam_rays, sample_pdf, compute_weights
from lab4d.utils.torch_utils import compute_gradient
from lab4d.nnutils.warping import SkinningWarp, create_warp
from quaternion import quaternion_conjugate as _quaternion_conjugate_cuda
from gs.scene import GaussianModel
from gs.gaussian_renderer import render
from gs.scene.cameras import Camera, KCamera

from gs.arguments import PipelineParams
from argparse import ArgumentParser



class DictAsMemberAccess:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

class DeformableGaussian(GaussianModel):
    """A static neural radiance field with an MLP backbone.

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
        init_scale (float): Initial geometry scale factor.
        color_act (bool): If True, apply sigmoid to the output RGB
    """

    def __init__(
        self,
        data_info,
        fg_motion,
        num_inst=1,
        init_scale=0.1,
        opts=None
    ):
        rtmat = data_info["rtmat"]
        frame_info = data_info["frame_info"]
        frame_offset = data_info["frame_info"]["frame_offset"]
        frame_offset_raw = data_info["frame_info"]["frame_offset_raw"]
        geom_path = data_info["geom_path"]

        super().__init__(sh_degree=opts['sh_degree'])
        
        fg_motion = fg_motion[3:] # remove "gs-"
        self.warp = create_warp(fg_motion, data_info)
        self.fg_motion = fg_motion
        # dataset info
        self.opts = opts
        self.frame_offset = frame_offset
        self.frame_offset_raw = frame_offset_raw
        self.num_frames = frame_offset[-1]
        self.num_inst = num_inst

        sigma = torch.tensor([1.0])
        self.logsigma = nn.Parameter(sigma.log())
        beta = torch.tensor([0.1])
        self.logibeta = nn.Parameter(-beta.log())

        self.logdir = "%s/%s-%s" % (opts["logroot"], opts["seqname"], opts["logname"])
        
        # self.pos_embedding = PosEmbedding(3, num_freq_xyz)
        # self.dir_embedding = PosEmbedding(3, num_freq_dir)
        # camera pose: field to camera
        rtmat[..., :3, 3] *= init_scale
        self.camera_mlp = CameraMLP(rtmat, frame_info=frame_info)

        # visibility mlp
        # self.vis_mlp = VisField(self.num_inst)

        # load initial mesh
        self.init_proxy(init_scale=init_scale)

        # non-parameters are not synchronized
        self.register_buffer(
            "near_far", torch.zeros(frame_offset_raw[-1], 2), persistent=False
        )
        self.register_buffer("aabb", torch.zeros(2, 3))
        # self.near_far = torch.zeros(frame_offset_raw[-1], 2, requires_grad=False).cuda()
        # self.aabb = torch.zeros(2, 3, requires_grad=False).cuda()
        self.update_aabb(beta=0)

        self.pipline = {'compute_cov3D_python': False, 'debug': False, 'convert_SHs_python': False}

        bg_color = [0 for i in range(16+2)]
        if self.opts["gs_learnable_bg"]:
            self.learnable_bkgd = nn.Parameter(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda", requires_grad=True))

        self.background_feat = torch.tensor([0 for i in range(3)], dtype=torch.float32, device="cuda")




        # self._xyz_c = self._xyz
        # self._rotation_c = self._rotation

        # feature_channels = 16
        # self._regist_feat = torch.zeros(self._xyz.shape[0], feature_channels, dtype=torch.float32, device="cuda")

        parser = ArgumentParser(description="Training script parameters")
        self.pipeline = PipelineParams(parser)
        self.pipeline.debug = opts["debug_cuda"]
        self.training_setup(DictAsMemberAccess(self.opts))

    @property
    def get_rotation(self):
        if hasattr(self,"_override_rotation"):
            return self.rotation_activation(self._override_rotation)
        else:
            return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        # import ipdb; ipdb.set_trace()
        if hasattr(self,"_override_xyz"):
            return self._override_xyz
        else:
            return self._xyz

    def render_view(self, view, override_xyz=None, override_rotation=None, override_color=None, override_bkgd=None):

        if override_xyz is not None:
            assert len(override_xyz.shape) == 2
            self._override_xyz = override_xyz
            self._override_rotation = override_rotation

        bkgd = self.background if override_bkgd is None else override_bkgd
        # assert bkgd.shape[-1] == 3 or bkgd.shape[-1] == override_color.shape[-1] + 3
        rendered = render(view, self, self.pipeline, bkgd, override_color=override_color)
        if self.opts["gs_learnable_bg"]:
            bkgd_rgb = self.learnable_bkgd[:, None, None].repeat(1, rendered['render'].shape[1], rendered['render'].shape[2])
            rendered['render'][:3] = rendered['render'][:3] + (1 - rendered["acc"]) * bkgd_rgb
        
        #    return {"render": rendered_image,
            # "viewspace_points": screenspace_points,
            # "visibility_filter" : radii > 0,
            # "radii": radii}

        if override_xyz is not None:
            del self._override_xyz
        if override_rotation is not None:
            del self._override_rotation
        # print("Check check grad")
        return rendered

    # def forward(self, xyz, dir=None, frame_id=None, inst_id=None, get_density=True):
    #     """
    #     Args:
    #         xyz: (M,N,D,3) Points along ray in object canonical space
    #         dir: (M,N,D,3) Ray direction in object canonical space
    #         frame_id: (M,) Frame id. If None, render at all frames
    #         inst_id: (M,) Instance id. If None, render for the average instance
    #     Returns:
    #         rgb: (M,N,D,3) Rendered RGB
    #         sigma: (M,N,D,1) If get_density=True, return density. Otherwise
    #             return signed distance (negative inside)
    #     """
    #     if frame_id is not None:
    #         assert frame_id.ndim == 1
    #     if inst_id is not None:
    #         assert inst_id.ndim == 1
    #     xyz_embed = self.pos_embedding(xyz)
    #     xyz_feat = self.basefield(xyz_embed, inst_id)

    #     sdf = self.sdf(xyz_feat)  # negative inside, positive outside
    #     if get_density:
    #         ibeta = self.logibeta.exp()
    #         # density = torch.sigmoid(-sdf * ibeta) * ibeta  # neus
    #         density = (
    #             0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibeta)
    #         ) * ibeta  # volsdf
    #         out = density
    #     else:
    #         out = sdf

    #     if dir is not None:
    #         dir_embed = self.dir_embedding(dir)
    #         if self.appr_channels > 0:
    #             appr_embed = self.appr_embedding.get_vals(frame_id)
    #             appr_embed = appr_embed[:, None, None].expand(
    #                 dir_embed.shape[:-1] + (appr_embed.shape[-1],)
    #             )
    #             appr_embed = torch.cat([dir_embed, appr_embed], -1)
    #         else:
    #             appr_embed = dir_embed

    #         xyz_embed = self.pos_embedding_color(xyz)
    #         xyz_feat = xyz_feat + self.colorfield(xyz_embed, inst_id)

    #         rgb = self.rgb(torch.cat([xyz_feat, appr_embed], -1))
    #         if self.color_act:
    #             rgb = rgb.sigmoid()
    #         out = rgb, out
    #     return out



    # def geometry_init(self, sdf_fn, nsample=256):
    #     """Initialize SDF using tsdf-fused geometry if radius is not given.
    #     Otherwise, initialize sdf using a unit sphere

    #     Args:
    #         sdf_fn (Function): Maps vertices to signed distances
    #         nsample (int): Number of samples
    #     """
    #     device = next(self.parameters()).device
    #     # setup optimizer
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    #     # optimize
    #     for i in range(500):
    #         optimizer.zero_grad()

    #         # sample points and gt sdf
    #         inst_id = torch.randint(0, self.num_inst, (nsample,), device=device)

    #         # sample points
    #         pts = self.sample_points_aabb(nsample, extend_factor=0.25)

    #         # get sdf from proxy geometry
    #         sdf_gt = sdf_fn(pts)

    #         # evaluate sdf loss
    #         sdf = self.forward(pts, inst_id=inst_id, get_density=False)
    #         scale = align_vectors(sdf, sdf_gt)
    #         sdf_loss = (sdf * scale.detach() - sdf_gt).pow(2).mean()

    #         # evaluate visibility loss
    #         vis = self.vis_mlp(pts, inst_id=inst_id)
    #         vis_loss = -F.logsigmoid(vis).mean()
    #         vis_loss = vis_loss * 0.01

    #         # evaluate eikonal loss
    #         eikonal_loss = self.compute_eikonal(pts[:, None, None], inst_id=inst_id)
    #         eikonal_loss = eikonal_loss[eikonal_loss > 0].mean()
    #         eikonal_loss = eikonal_loss * 1e-4

    #         total_loss = sdf_loss + vis_loss + eikonal_loss
    #         total_loss.backward()
    #         optimizer.step()
    #         if i % 100 == 0:
    #             print(f"iter {i}, loss {total_loss.item()}")

    def update_proxy(self):
        """Extract proxy geometry using marching cubes"""
        # mesh = self.extract_canonical_mesh(level=0.005)
        # if mesh is not None:
        #     self.proxy_geometry = mesh
        # create a trimesh.mesh from the vertices from self._xyz.detach().to("cpu")
        self.proxy_geometry = trimesh.Trimesh(vertices=self._xyz.detach().to("cpu"))

    @torch.no_grad()
    def extract_canonical_mesh(
        self,
        grid_size=64,
        level=0.0,
        inst_id=None,
        use_visibility=True,
        use_extend_aabb=True,
    ):
        """Extract canonical mesh using marching cubes

        Args:
            grid_size (int): Marching cubes resolution
            level (float): Contour value to search for isosurfaces on the signed
                distance function
            inst_id: (M,) Instance id. If None, extract for the average instance
            use_visibility (bool): If True, use visibility mlp to mask out invisible
            1region.
            use_extend_aabb (bool): If True, extend aabb by 50% to get a loose proxy.
              Used at training time.
        Returns:
            mesh (Trimesh): Extracted mesh
        """
        raise NotImplementedError
        if inst_id is not None:
            inst_id = torch.tensor([inst_id], device=next(self.parameters()).device)
        sdf_func = lambda xyz: self.forward(xyz, inst_id=inst_id, get_density=False)
        vis_func = lambda xyz: self.vis_mlp(xyz, inst_id=inst_id) > 0
        if use_extend_aabb:
            aabb = extend_aabb(self.aabb, factor=0.5)
        else:
            aabb = self.aabb
        

        mesh = marching_cubes(
            sdf_func,
            aabb,
            visibility_func=vis_func if use_visibility else None,
            grid_size=grid_size,
            level=level,
            apply_connected_component=True if self.category == "fg" else False,
        )
        return mesh
            
    def init_proxy(self, init_scale=0.2): # refactored!
        """Initialize the geometry from a mesh

        Args:
            geom_path (str): Initial shape mesh
            init_scale (float): Geometry scale factor
        """
        from gs.utils.sh_utils import SH2RGB
        from gs.scene.gaussian_model import BasicPointCloud
        from gs.scene.dataset_readers import fetchPly, storePly
        import os

        if not self.opts["gs_init_mesh"] == "":
            xyz, color, feature = load_mesh_as_pcd_trimesh(self.opts["gs_init_mesh"], 200000, return_feat=True)
            feature_tensor = torch.from_numpy(feature).cuda()  # 先转为张量并移动到GPU
            feature_tensor = feature_tensor.requires_grad_(True)  # 设置梯度追踪
            self._regist_feat = nn.Parameter(feature_tensor)  # 包装为模型参数
            # self._regist_feat = nn.Parameter(data=torch.from_numpy(feature)).contiguous().requires_grad_(True).cuda()
            # self._regist_feat = torch.tensor(feature, dtype=torch.float32, device="cuda").requires_grad_(True)
            # import ipdb;ipdb.set_trace()
            assert feature.shape[0] == color.shape[0]
            pcd = BasicPointCloud(xyz, color[:, :3] / 255, None)
            # scene = Scene(dataset, gaussians)

        else:
            if not self.opts["gs_init_ply"] == "":
                ply_path = self.opts["gs_init_ply"]
            else:
                ply_path = os.path.join("%s/%s-%s" % (self.opts["logroot"], self.opts["seqname"], self.opts["logname"]), "init_points3d.ply")

                # Since this data set has no colmap data, we start with random points
                num_pts = 100_000
                print(f"Generating random point cloud ({num_pts})...")
            
                # We create random points inside the bounds of the synthetic Blender scenes
                # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

                xyz = np.random.random((num_pts, 3)) * init_scale
                shs = np.random.random((num_pts, 3)) / 255.0
                pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

                storePly(ply_path, xyz, SH2RGB(shs) * 255)

            try:
                pcd = fetchPly(ply_path)
            except:
                pcd = None

        # import ipdb; ipdb.set_trace()
        radius = np.linalg.norm(xyz - np.mean(xyz, axis=0, keepdims=True), axis=-1)
        radius_scale = 1.1
        self.cameras_extent = np.max(radius) * radius_scale
        print("GS scene radius =", self.cameras_extent)

        self.create_from_pcd(pcd, self.cameras_extent)
        self.proxy_geometry = trimesh.Trimesh(vertices=self._xyz.detach().to("cpu"))


    def update_aabb(self, beta=0.9):
        """Update axis-aligned bounding box by interpolating with the current
        proxy geometry's bounds

        Args:
            beta (float): Interpolation factor between previous/current values
        """
        device = self.aabb.device
        pts = self.proxy_geometry.vertices
        pts = torch.tensor(pts, dtype=torch.float32, device=device)
        beta = torch.tensor(beta, device=device)
        if pts is not None:
            # pts = torch.tensor(pts, dtype=torch.float32, device=device)
            # import ipdb; ipdb.set_trace()
            # aabb = torch.stack([[pts.min(0)[0][0], pts.min(0)[0][1], pts.min(0)[0][2]], [pts.max(0)[0][0], pts.max(0)[0][1], pts.max(0)[0][2]]], 0)
            aabb = torch.stack([pts.min(0)[0], pts.max(0)[0]], 0)
            self.aabb = self.aabb * beta + aabb * (1 - beta)



    def update_near_far(self, beta=0.9): # refactored!
        """Update near-far bounds by interpolating with the current near-far bounds

        Args:
            beta (float): Interpolation factor between previous/current values
        """
        # get camera
        device = next(self.parameters()).device
        with torch.no_grad():
            quat, trans = self.camera_mlp.get_vals()  # (B, 4, 4)
            rtmat = quaternion_translation_to_se3(quat, trans)

        verts = self.proxy_geometry.vertices
        if verts is not None:
            proxy_pts = torch.tensor(verts, dtype=torch.float32, device=device)
            near_far = get_near_far(proxy_pts, rtmat).to(device)
            frame_mapping = self.camera_mlp.time_embedding.frame_mapping
            self.near_far.data[frame_mapping] = self.near_far.data[
                frame_mapping
            ] * beta + near_far * (1 - beta)

    def sample_points_aabb(self, nsample): # overrided!
        """Sample points within axis-aligned bounding box

        Args:
            nsample (int): Number of samples
            extend_factor (float): Extend aabb along each side by factor of
                the previous size
        Returns:
            pts: (nsample, 3) Sampled points
        """
        num_candidates = min(nsample, self._xyz.shape[0])
        idx = torch.randperm(self._xyz.shape[0])[:num_candidates]
        pts = self._xyz[idx].detach() # 需要detach吧
        return pts

    def visibility_decay_loss(self, nsample=512):
        """Encourage visibility to be low at random points within the aabb. The
        effect is that invisible / occluded points are assigned -inf visibility

        Args:
            nsample (int): Number of points to sample
        Returns:
            loss: (0,) Visibility decay loss
        """
        # sample random points
        device = next(self.parameters()).device
        return torch.tensor(0.0, device=device, requires_grad=True)
        pts = self.sample_points_aabb(nsample)
        inst_id = torch.randint(0, self.num_inst, (nsample,), device=device)

        # evaluate loss
        vis = self.vis_mlp(pts, inst_id=inst_id)
        loss = -F.logsigmoid(-vis).mean()
        return loss

    # def compute_eikonal(self, xyz, inst_id=None, sample_ratio=16):
    #     """Compute eikonal loss

    #     Args:
    #         xyz: (M,N,D,3) Input coordinates in canonical space
    #         inst_id: (M,) Instance id, or None to use the average instance
    #         sample_ratio (int): Fraction to subsample to make it more efficient
    #     Returns:
    #         eikonal_loss: (M,N,D,1) Squared magnitude of SDF gradient
    #     """
    #     M, N, D, _ = xyz.shape
    #     xyz = xyz.reshape(-1, D, 3)
    #     sample_size = xyz.shape[0] // sample_ratio
    #     if sample_size < 1:
    #         sample_size = 1
    #     if inst_id is not None:
    #         inst_id = inst_id[:, None].expand(-1, N)
    #         inst_id = inst_id.reshape(-1)
    #     eikonal_loss = torch.zeros_like(xyz[..., 0])

    #     # subsample to make it more efficient
    #     if M * N > sample_size:
    #         probs = torch.ones(M * N)
    #         rand_inds = torch.multinomial(probs, sample_size, replacement=False)
    #         xyz = xyz[rand_inds]
    #         if inst_id is not None:
    #             inst_id = inst_id[rand_inds]
    #     else:
    #         rand_inds = Ellipsis

    #     xyz = xyz.detach()
    #     inst_id = inst_id.detach() if inst_id is not None else None
    #     fn_sdf = lambda x: self.forward(x, inst_id=inst_id, get_density=False)
    #     g = compute_gradient(fn_sdf, xyz)[..., 0]

    #     eikonal_loss[rand_inds] = (g.norm(2, dim=-1) - 1) ** 2
    #     eikonal_loss = eikonal_loss.reshape(M, N, D, 1)
    #     return eikonal_loss

    # def compute_normal(self, xyz_cam, dir_cam, field2cam, frame_id=None, inst_id=None, samples_dict={}):
    #     """Compute eikonal loss and normals in camera space

    #     Args:
    #         xyz_cam: (M,N,D,3) Points along rays in camera space
    #         dir_cam: (M,N,D,3) Ray directions in camera space
    #         field2cam: (M,SE(3)) Object-to-camera SE(3) transform
    #         frame_id: (M,) Frame id to query articulations, or None to use all frames
    #         inst_id: (M,) Instance id, or None to use the average instance
    #         samples_dict (Dict): Time-dependent bone articulations. Keys:
    #             "rest_articulation": ((M,B,4), (M,B,4)) and
    #             "t_articulation": ((M,B,4), (M,B,4))
    #     Returns:
    #         normal: (M,N,D,3) Normal vector field in camera space
    #     """
    #     M, N, D, _ = xyz_cam.shape

    #     def fn_sdf(xyz_cam):
    #         xyz = self.backward_warp(
    #             xyz_cam,
    #             dir_cam,
    #             field2cam,
    #             frame_id=frame_id,
    #             inst_id=inst_id,
    #             samples_dict=samples_dict,
    #         )["xyz"]
    #         sdf = self.forward(xyz, inst_id=inst_id, get_density=False)
    #         return sdf

    #     g = compute_gradient(fn_sdf, xyz_cam)[..., 0]

    #     eikonal = (g.norm(2, dim=-1, keepdim=True) - 1) ** 2
    #     normal = torch.nn.functional.normalize(g, dim=-1)

    #     # Multiply by [1, -1, -1] to match normal conventions from ECON
    #     # https://github.com/YuliangXiu/ECON/blob/d98e9cbc96c31ecaa696267a072cdd5ef78d14b8/apps/infer.py#L257
    #     normal = normal * torch.tensor([1, -1, -1], device="cuda")

    #     return eikonal, normal

    @torch.no_grad()
    def get_valid_idx(self, xyz, xyz_t=None, vis_score=None, samples_dict={}):
        """Return a mask of valid points by thresholding visibility score

        Args:
            xyz: (M,N,D,3) Points in object canonical space to query
            xyz_t: (M,N,D,3) Points in object time t space to query
            vis_score: (M,N,D,1) Predicted visibility score, not used
        Returns:
            valid_idx: (M,N,D) Visibility mask, bool
        """
        # check whether the point is inside the aabb
        aabb = extend_aabb(self.aabb)
        # (M,N,D), whether the point is inside the aabb
        inside_aabb = check_inside_aabb(xyz, aabb)

        # valid_idx = inside_aabb & (vis_score[..., 0] > -5)
        valid_idx = inside_aabb

        if xyz_t is not None and "t_articulation" in samples_dict.keys():
            # for time t points, we set aabb based on articulation
            t_bones = dual_quaternion_to_quaternion_translation(
                samples_dict["t_articulation"]
            )[1][0]
            t_aabb = torch.stack([t_bones.min(0)[0], t_bones.max(0)[0]], 0)
            t_aabb = extend_aabb(t_aabb, factor=1.0)
            inside_aabb = check_inside_aabb(xyz_t, t_aabb)
            valid_idx = valid_idx & inside_aabb

        # temporally disable visibility mask
        if self.category == "bg":
            valid_idx = None

        return valid_idx

    # @torch.no_grad()
    # def importance_sampling(
    #     self,
    #     hxy,
    #     Kinv,
    #     near_far,
    #     field2cam,
    #     frame_id,
    #     inst_id,
    #     samples_dict,
    #     n_depth=64,
    # ):
    #     """
    #     importance sampling coarse
    #     """
    #     # sample camera space rays
    #     xyz_cam, dir_cam, deltas, depth = sample_cam_rays(
    #         hxy, Kinv, near_far, perturb=False, n_depth=n_depth // 2
    #     )  # (M, N, D, x)

    #     # backward warping
    #     xyz = self.backward_warp(
    #         xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict=samples_dict
    #     )["xyz"]

    #     # get pdf
    #     density = self.forward(
    #         xyz,
    #         dir=None,
    #         frame_id=frame_id,
    #         inst_id=inst_id,
    #     )  # (M, N, D, x)
    #     weights, _ = compute_weights(density, deltas)  # (M, N, D, x)

    #     depth_mid = 0.5 * (depth[:, :, :-1] + depth[:, :, 1:])  # (M, N, D-1)
    #     is_det = not self.training
    #     depth_mid = depth_mid.view(-1, n_depth // 2 - 1)
    #     weights = weights.view(-1, n_depth // 2)

    #     depth_ = sample_pdf(
    #         depth_mid, weights[:, 1:-1], n_depth // 2, det=is_det
    #     ).detach()
    #     depth_ = depth_.reshape(depth.shape)
    #     # detach so that grad doesn't propogate to weights_sampled from here

    #     depth, _ = torch.sort(torch.cat([depth, depth_], -2), -2)  # (M, N, D)

    #     # sample camera space rays
    #     xyz_cam, dir_cam, deltas, depth = sample_cam_rays(
    #         hxy, Kinv, near_far, depth=depth, perturb=False
    #     )

    #     return xyz_cam, dir_cam, deltas, depth

    # def compute_jacobian(self, xyz, xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict):
    #     """Compute eikonal and normal fields from Jacobian of SDF

    #     Args:
    #         xyz: (M,N,D,3) Points along rays in object canonical space. Only for training
    #         xyz_cam: (M,N,D,3) Points along rays in camera space. Only for rendering
    #         dir_cam: (M,N,D,3) Ray directions in camera space. Only for rendering
    #         field2cam: (M,SE(3)) Object-to-camera SE(3) transform. Only for rendering
    #         frame_id: (M,) Frame id to query articulations, or None to use all frames.
    #             Only for rendering
    #         inst_id: (M,) Instance id. If None, compute for the average instance
    #         samples_dict (Dict): Time-dependent bone articulations. Only for rendering. Keys:
    #             "rest_articulation": ((M,B,4), (M,B,4)) and
    #             "t_articulation": ((M,B,4), (M,B,4))
    #     Returns:
    #         jacob_dict (Dict): Jacobian fields. Keys: "eikonal" (M,N,D,1). Only when
    #             rendering, "normal" (M,N,D,3)
    #     """
    #     jacob_dict = {}
    #     if self.training:
    #         # For efficiency, compute subsampled eikonal loss in canonical space
    #         jacob_dict["eikonal"] = self.compute_eikonal(xyz, inst_id=inst_id)
    #     else:
    #         # For rendering, compute full eikonal loss and normals in camera space
    #         jacob_dict["eikonal"], jacob_dict["normal"] = self.compute_normal(
    #             xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict
    #         )
    #     return jacob_dict

    # def query_nerf(self, xyz, dir, frame_id, inst_id, valid_idx=None):
    #     """Neural radiance field rendering

    #     Args:
    #         xyz: (M,N,D,3) Points along rays in object canonical space
    #         dir: (M,N,D,3) Ray directions in object canonical space
    #         frame_id: (M,) Frame id. If None, render at all frames
    #         inst_id: (M,) Instance id. If None, render for the average instance
    #         valid_idx: (M,N,D) Mask of whether each point is visible to camera
    #     Returns:
    #         field_dict (Dict): Field outputs. Keys: "rgb" (M,N,D,3),
    #             "density" (M,N,D,1), and "density_{fg,bg}" (M,N,D,1)
    #     """
    #     if valid_idx is not None:
    #         if valid_idx.sum() == 0:
    #             field_dict = {
    #                 "rgb": torch.zeros(valid_idx.shape + (3,), device=xyz.device),
    #                 "density": torch.zeros(valid_idx.shape + (1,), device=xyz.device),
    #                 "density_%s"
    #                 % self.category: torch.zeros(
    #                     valid_idx.shape + (1,), device=xyz.device
    #                 ),
    #             }
    #             return field_dict
    #         # reshape
    #         shape = xyz.shape
    #         xyz = xyz[valid_idx][:, None, None]  # MND,1,1,3
    #         dir = dir[valid_idx][:, None, None]
    #         frame_id = frame_id[:, None, None].expand(shape[:3])[valid_idx]
    #         inst_id = inst_id[:, None, None].expand(shape[:3])[valid_idx]

    #     rgb, density = self.forward(
    #         xyz,
    #         dir=dir,
    #         frame_id=frame_id,
    #         inst_id=inst_id,
    #     )  # (M, N, D, x)

    #     # reshape
    #     field_dict = {
    #         "rgb": rgb,
    #         "density": density,
    #         "density_%s" % self.category: density,
    #     }

    #     if valid_idx is not None:
    #         for k, v in field_dict.items():
    #             tmpv = torch.zeros(valid_idx.shape + (v.shape[-1],), device=v.device)
    #             tmpv[valid_idx] = v.view(-1, v.shape[-1])
    #             field_dict[k] = tmpv
    #     return field_dict

    @staticmethod
    def cam_to_field(xyz_cam, field2cam, rotation_cam=None):
        """Transform rays from camera SE(3) to object SE(3)

        Args:
            xyz_cam: (M,N,D,3) Points along rays in camera SE(3)
            dir_cam: (M,N,D,3) Ray directions in camera SE(3)
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
        Returns:
            xyz: (M,N,D,3) Points along rays in object SE(3)
            dir: (M,N,D,3) Ray directions in object SE(3)
        """
        # warp camera space points to canonical space
        # scene/object space rays # (M,1,1,4,4) * (M,N,D,3) = (M,N,D,3)
        # assert field2cam is not None
        shape = xyz_cam.shape
        cam2field = quaternion_translation_inverse(field2cam[0], field2cam[1])
        cam2field = (
            cam2field[0][:, None, None].expand(shape[:-1] + (4,)).clone(),
            cam2field[1][:, None, None].expand(shape[:-1] + (3,)).clone(),
        )
        xyz = apply_se3mat(cam2field, xyz_cam)
        cam2field = (cam2field[0], torch.zeros_like(cam2field[1]))
        if rotation_cam is not None:
            dir = apply_se3mat(cam2field, rotation_cam)
            return xyz, dir
        else:
            return xyz

    def field_to_cam(self, xyz, rotation, field2cam):
        """Transform points from object SE(3) to camera SE(3)

        Args:
            xyz: (M,N,D,3) Points in object SE(3)
            field2cam: (M,SE(3)) Object to camera SE(3) transform
        Returns:
            xyz_cam: (M,N,D,3) Points in camera SE(3)
        """
        # transform from canonical to next frame camera space
        # (M,1,1,3,4) @ (M,N,D,3) = (M,N,D,3)
        shape = xyz.shape
        field2cam = (
            field2cam[0][:, None, None].expand(shape[:-1] + (4,)).clone(),
            field2cam[1][:, None, None].expand(shape[:-1] + (3,)).clone(),
        )
        xyz_cam_next = apply_se3mat(field2cam, xyz)
        import ipdb; ipdb.set_trace()

        # TODO rotation_cam_next = apply_se3mat(field2cam, rotation)
        return xyz_cam_next, rotation_cam_next

    @staticmethod
    def flip_pair(tensor):
        """Flip the tensor along the pair dimension

        Args:
            tensor: (M*2, ...) Inputs [x0, x1, x2, x3, ..., x_{2k}, x_{2k+1}]

        Returns:
            tensor: (M*2, ...) Outputs [x1, x0, x3, x2, ..., x_{2k+1}, x_{2k}]
        """
        if torch.is_tensor(tensor):
            if len(tensor) < 2:
                return tensor
            return tensor.view(tensor.shape[0] // 2, 2, -1).flip(1).view(tensor.shape)
        elif isinstance(tensor, tuple):
            return tuple([DeformableGaussian.flip_pair(t) for t in tensor])
        elif isinstance(tensor, dict):
            return {k: DeformableGaussian.flip_pair(v) for k, v in tensor.items()}

    def cam_prior_loss(self):
        """Encourage camera transforms over time to match external priors.

        Returns:
            loss: (0,) Mean squared error of camera SE(3) transforms to priors
        """
        loss = self.camera_mlp.compute_distance_to_prior()
        return loss

    # @train_only_fields
    def compute_flow(
        self,
        xyz_cam_t,
        frame_id,
        inst_id,
        field2cam,
        Kinv,
        samples_dict,
        flow_thresh=None,
        visibility_filter=None,
    ):
        """计算渲染图片next帧时用到的所有点的场景流,然后投影

        Args:
            hxy: (M,N,D,3) Homogeneous pixel coordinates on the image plane
            xyz: (M,N,D,3) Canonical field coordinates
            Kinv: (M,3,3) Inverse of camera intrinsics
            flow_thresh (float): Threshold for flow magnitude

        Returns:
            flow: (M,N,D,2) Optical flow proposal
        """
        # import ipdb; ipdb.set_trace()
        Kmat = Kmatinv(Kinv)

        # ones = torch.ones((xyz.shape[0], xyz.shape[1], xyz.shape[2], 1), device=xyz.device)
        # xyz_cam = self.field_to_cam(xyz, field2cam)
        xy_cam = torch.matmul(Kmat[:, None, None], xyz_cam_t[..., None])[..., 0]
        xy_cam = xy_cam / xy_cam[..., 2:] 
        xy_cam = xy_cam[..., :2]
        

        # flip the frame id
        xyz_cam_t_next = self.flip_pair(xyz_cam_t)
        frame_id_next = self.flip_pair(frame_id)
        field2cam_next = (self.flip_pair(field2cam[0]), self.flip_pair(field2cam[1]))
        Kinv_next = self.flip_pair(Kinv)
        samples_dict_next = self.flip_pair(samples_dict)
        Kmat_next = Kmatinv(Kinv_next)

        # xyz_cam_next, rotation_next, _ = self.forward_warp(xyz, None, frame_id_next, inst_id, samples_dict=samples_dict_next)
        # xyz_cam_next, rotation_next, _ = self.forward_warp(self._xyz, None, frame_id_next, inst_id, samples_dict=samples_dict_next)
        # xyz_cam_next = self.field_to_cam(xyz_next, field2cam_next)
        # xy_cam_next = torch.matmul(Kmat_next[:, None, None], xyz_cam_next[..., None])[..., 0]
        xy_cam_next = torch.matmul(Kmat_next[:, None, None], xyz_cam_t_next[..., None])[..., 0]
        xy_cam_next = xy_cam_next / xy_cam_next[..., 2:]
        xy_cam_next = xy_cam_next[..., :2]

        # save xyz_cam use trimesh
        # trimesh.Trimesh(vertices=xyz_cam_t[0,:,0].detach().cpu().numpy()).export("tmp/xyz_cam_t.obj")
        # trimesh.Trimesh(vertices=xyz_cam_t_next[0,:,0].detach().cpu().numpy()).export("tmp/xyz_cam_t_next.obj")
        
        # trimesh.Trimesh(vertices=xyz_cam_next[0,:,0].detach().cpu().numpy()).export("tmp/xyz_next_cam.obj")

        flow = xy_cam_next - xy_cam
        
        # # 假设pair是t和t+1 t帧和t+1的位置差就是场景流， 因为要的是t帧的场景流，所以只关心t帧参与渲染的点
        # scene_flow = xyz_next - computed_xyz_t

        # # project scene flow into cam coords, pts stay the same
        # scene_flow_cam = self.field_to_cam(scene_flow, field2cam_next)


        # # project scene flow into 2D using Kmat_next
        # projected_flow = self.project_scene_flow(scene_flow=scene_flow_cam, Kmat_next=Kmat_next)
        # import ipdb;ipdb.set_trace()  # 检查是否和原生lab4d一致
        return flow
        # extend flow to all points for rendering
        # extend_flow = torch.zeros(self._xyz.shape[0], 3, device=self._xyz.device)
        # extend_flow[visibility_filter] = projected_flow
        
        # if flow_thresh is not None:
        #     flow_thresh = float(flow_thresh)
        #     xyz_valid = xyz_valid & (flow.norm(dim=-1, keepdim=True) < flow_thresh)

        # render flow
        gs_camera = self.get_gs_camera(field2cam, Kinv, samples_dict_next["H"], samples_dict_next["W"])
        rendered = self.render(gs_camera, override_xyz, override_color=projected_flow)

        import ipdb;ipdb.set_trace() # concat这玩意儿干嘛的
        # flow = torch.cat([flow, xyz_valid.float()], dim=-1)

        flow_dict = {"flow": rendered['render']}
        return flow_dict
#--------------------------from deformable.py-----------------
    def project_scene_flow(self, scene_flow, Kmat_next):
        """
        Project 3D scene flow into 2D using intrinsic matrix Kmat_next.

        Args:
            scene_flow: (M,N,D,3) Scene flow in camera coordinates
            Kmat_next: (M,3,3) Intrinsic matrix of the next camera

        Returns:
            projected_flow: (M,N,D,2) Projected scene flow in 2D
        """
        # Reshape scene flow and Kmat_next
        shape = scene_flow.shape
        scene_flow = scene_flow.view(-1, 3)
        Kmat_next = Kmat_next.view(-1, 3, 3)

        # Project scene flow using Kmat_next 
        projected_flow = torch.matmul(Kmat_next[:, None, None], scene_flow[..., None])[..., 0]
        projected_flow = projected_flow[..., 0:2] / projected_flow[..., 2:]

        # # Reshape projected flow back to original shape
        # projected_flow = projected_flow[].view(*shape[:-1], 2)

        return projected_flow

    def get_gs_Kcamera(self, Kinvs, Hs, Ws):
        batch_size = Kinvs.shape[0]
        gs_cameras = []
        for i in range(batch_size):
            Kinv = Kinvs[i]
            Kmat = Kmatinv(Kinv[None, ...])[0]
            H = Hs[i]
            W = Ws[i]


            # hxy[0]=torch.tensor([0,0,1])
            # hxy[1]=torch.tensor([0,W,1])
            # hxy[2]=torch.tensor([H,0,1])
            # hxy[3]=torch.tensor([H,W,1])
            # dir = hxy @ Kinv.permute(1, 0)
            # xyrange = dir.max(axis=0)[0] - dir.min(axis=0)[0]

            # FoVx = 2 * torch.atan(W * Kinv[0, 0] / 2)
            # FoVy = 2 * torch.atan(H * Kinv[1, 1] / 2)
            # dx = Kinv[0, 2]
            # dy = Kinv[1, 2]

            l = Kinv[0, 2] 
            r = Kinv[0, 2] + Kinv[0, 0] * W
            b = Kinv[1, 2]
            t = Kinv[1, 2] + Kinv[1, 1] * H
            gs_camera = KCamera(H=H, W=W, left=l, right=r, top=t, bottom=b)
            # if self.opts["force_center_cam"]:
            #     try:
            #         assert l == -r and t == -b ; "Only symmetric frustums are supported"
            #     except:
            #         print("lrbt", l, r, b, t)
            #         import ipdb; ipdb.set_trace()
            gs_cameras.append(gs_camera)
        # import ipdb;ipdb.set_trace()
        return gs_cameras


    def get_gs_camera(self, field2cams, Kinvs, Hs, Ws, zero_render=False, reverse_pose=False, DEBUG=False):
        # solve Camera, RT is W2C

        batch_size = field2cams[0].shape[0]
        gs_cameras = []
        for i in range(batch_size):
            q = field2cams[0][i]
            R = quaternion_to_matrix(q)
            T = field2cams[1][i]
            Kinv = Kinvs[i]
            Kmat = Kmatinv(Kinv[None, ...])[0]
            # W = (2 * Kmat[0, 2]).to(int)
            # H = (2 * Kmat[1, 2]).to(int)
            H = Hs[i]
            W = Ws[i]
            
            # wrong
            FoVx = 2 * torch.arctan(Kmat[0, 2] / Kmat[0, 0])
            FoVy = 2 * torch.arctan(Kmat[1, 2] / Kmat[1, 1])

            # correct
            hxy = torch.zeros([4, 3]).cuda()
            hxy[0]=torch.tensor([0,0,1])
            hxy[1]=torch.tensor([0,W,1])
            hxy[2]=torch.tensor([H,0,1])
            hxy[3]=torch.tensor([H,W,1])
            dir = hxy @ Kinv.permute(1, 0)
            xyrange = dir.max(axis=0)[0] - dir.min(axis=0)[0]
            FoVx = 2 * torch.atan(xyrange[0] / 2)
            FoVy = 2 * torch.atan(xyrange[1] / 2)


            # FoVx /= 2
            # FoVy /= 2


            # import ipdb; ipdb.set_trace()
            # "train_res"
            # FoVx = 2 * torch.arctan(self.opts["eval_res"]/2 / Kmat[0, 0])
            # FoVy = 2 * torch.arctan(self.opts["eval_res"]/2 / Kmat[1, 1])

            # print("FoVx, FoVy", FoVx * 180 / np.pi, FoVy * 180 / np.pi)
            # print("R,T=", R.data, T.data)
            # print("Kmat=", Kmat.data)
            # import ipdb; ipdb.set_trace()
            # H = int(2 * Kmat[0,2])
            # W = int(2 * Kmat[1,2])
            
            dummy_image = torch.zeros([3, H, W])
            gt_alpha_mask = None
            uid = None

            if zero_render:
                R = torch.eye(3)
                T = torch.zeros(3)
            
            if reverse_pose:
                # import ipdb; ipdb.set_trace()
                R = R.T
                T = -R @ T

            gs_camera = Camera(None, R, T, FoVx, FoVy, dummy_image, gt_alpha_mask, None, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0)
            gs_cameras.append(gs_camera)

            # import ipdb; ipdb.set_trace()
        return gs_cameras

    @staticmethod
    def apply_qt_to_gaussian(xyz, rotation, q, t, bs, inplace=False):
        """return apply rotation and translation to the self canonical gaussian"""
        shape = xyz.shape
        xyz = xyz.view(bs, -1, 3)
        xyz = quaternion_translation_apply(q, t, xyz)
        xyz = xyz.view(*shape)
        if rotation is not None:
            rotation = rotation.view(bs, -1, 4)
            rotation = quaternion_mul(q, rotation)
            # rotation = rotation + q
        else:
            rotation = None

        return xyz,rotation

    def query_field(self, samples_dict, flow_thresh=None):
        """ 

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
        # import ipdb;ipdb.set_trace()
        Kinv = samples_dict["Kinv"]  # (M,3,3)
        field2cam = samples_dict["field2cam"]  # (M,SE(3))
        frame_id = samples_dict["frame_id"]  # (M,)
        inst_id = samples_dict["inst_id"]  # (M,)
        near_far = samples_dict["near_far"]  # (M,2)
        hxy = samples_dict["hxy"]  # (M,N,2)
        H = samples_dict["H"]  # (M,)
        W = samples_dict["W"]  # (M,)

        batch_size = Kinv.shape[0]

        # --------- bypass visibility ------------
        # # visibility
        # vis_score = self.vis_mlp(xyz, inst_id=inst_id)  # (M, N, D, 1)

        # # compute valid_indices to speed up querying fields
        # if self.training:
        #     valid_idx = None
        # else:
        #     valid_idx = self.get_valid_idx(xyz, xyz_t, vis_score, samples_dict)

        # # NeRF
        # feat_dict = self.query_nerf(xyz, dir, frame_id, inst_id, valid_idx=valid_idx)

        # # visibility
        # feat_dict["vis"] = vis_score
        # ---------------------------------

        # test cam if you like

        # from test_gs import test_gs_cam
        # test_gs_cam(self, T_init=torch.tensor([0, 0, 1.0]))
        # test_gs_cam(self, T_init=torch.tensor([0, 0, 1.0]), inverse_cam=True)
        # test_gs_cam(self, T_init=field2cam[1][0].cpu(), R_init=quaternion_to_matrix(field2cam[0][0]).cpu())
        #------------------------------
        # import ipdb ; ipdb.set_trace()
        # compute gauss_density for render

        feat_dict, deltas, aux_dict = ({}, {}, {})
        gauss_field = self.compute_gauss_density(self._xyz, samples_dict)
        aux_dict.update(gauss_field)
        gauss_density = gauss_field["gauss_density"]

        xyz = self._xyz
        rotation = self._rotation
        regist_feat = self._regist_feat
        xyz_reshape = xyz.reshape(1, xyz.shape[0], 1, xyz.shape[1]).expand(frame_id.shape[0], -1, -1, -1) # (B, num_gs, 1, 3)
        rotation_reshape = rotation.reshape(1, rotation.shape[0], 1, rotation.shape[1]).expand(frame_id.shape[0], -1, -1, -1) # to align with nerf
        regist_feat_reshape = regist_feat.reshape(1, regist_feat.shape[0], 1, regist_feat.shape[1]).expand(frame_id.shape[0], -1, -1, -1) # to align with nerf
        # gauss_density_reshape = gauss_density.reshape(1, gauss_density.shape[0], 1, gauss_density.shape[1]).expand(frame_id.shape[0], -1, -1, -1) # to align with nerf
        xyz_reshape = xyz_reshape.contiguous()
        rotation_reshape = rotation_reshape.contiguous()
        regist_feat_reshape = regist_feat_reshape.contiguous()
        # gauss_density_reshape = gauss_density_reshape.contiguous()

        if "no_warp" in samples_dict.keys():
            xyz_cam_t, rotation_t, qt_forward = (xyz_reshape, rotation_reshape[:, :, 0, :], None)
        else:
            xyz_cam_t, rotation_t, qt_forward = self.forward_warp(xyz_reshape, rotation_reshape, frame_id, inst_id, samples_dict, cache_aux_dict=True)

        # compute normal
        scaling = self.get_scaling
        values, shortest_axis = torch.min(scaling, dim=1)
        shortest_axis = shortest_axis[None, ...].repeat(batch_size, 1)

        # # 生成用于索引的批次和序列号
        # rotation_mat = quaternion_to_matrix(rotation_t)
        # batch_indices = torch.arange(rotation_mat.size(0)).unsqueeze(1).expand(-1, rotation_mat.size(1))
        # sequence_indices = torch.arange(rotation_mat.size(1)).unsqueeze(0).expand(rotation_mat.size(0), -1)
        # # 使用高级索引从每个矩阵中选择列
        # normal = rotation_mat[batch_indices, sequence_indices, :, shortest_axis]


        axis = torch.zeros_like(xyz)
        batch_indices = torch.arange(axis.size(0))
        axis[batch_indices, shortest_axis] = 1
        axis = axis[None, ...].repeat(batch_size, 1, 1, 1)
        normal = quaternion_apply(rotation_t, axis)
        q, t = samples_dict["field2cam"]
        q = q[:, None].repeat(1, xyz_cam_t.shape[1], 1)
        t = t[:, None].repeat(1, xyz_cam_t.shape[1], 1)
        normal = quaternion_apply(q, normal)



        # calculate flow first, wait for rendering together
        if not "is_gen3d" in samples_dict.keys() and not "no_warp" in samples_dict.keys():
            # flow
            # 
            pointwise_flow = self.compute_flow(
                xyz_cam_t,
                frame_id,
                inst_id,
                field2cam,
                Kinv,
                samples_dict,
                flow_thresh=flow_thresh,
            )
        else:
            pointwise_flow = torch.ones_like(xyz_cam_t[..., :2])
        # concat self._regii_feat and pointwise_flow 3+16+2

        # gs_camera = self.get_gs_camera(field2cam, Kinv, H, W, zero_render=True, reverse_pose=False)
        gs_camera = self.get_gs_Kcamera(Kinv, H, W)

        rendered = {}
        # import ipdb;ipdb.set_trace()
        
        for i in range(batch_size):
            assert xyz_cam_t.shape[2] == 1
            # rendered_i = self.render_view(gs_camera[i], override_xyz=xyz_t[i,:,0], override_rotation=rotation_t[i])
            flow_scale = torch.max(pointwise_flow.max(), torch.abs(pointwise_flow.min())) + 1e-6
            pointwise_flow_scaled = pointwise_flow / flow_scale
            # pointwise_flow_scaled = torch.zeros_like(pointwise_flow_scaled)

            concated_feat = torch.cat([self._regist_feat, pointwise_flow_scaled[i, :, 0, :], normal[i], gauss_density], dim=-1)
            concated_feat = None
            rendered_i = self.render_view(gs_camera[i], override_xyz=xyz_cam_t[i,:,0], 
                                            override_rotation=rotation_t[i],
                                            override_color=concated_feat,
                                            override_bkgd=self.background_feat,
                                            )
                                        #  override_color=self._regist_feat,
            rgb_rendered = rendered_i["render"][:3]

            feature_rendered = rendered_i["render"][3:19]
            flow_rendered = rendered_i["render"][19:21] * flow_scale
            normal_rendered = rendered_i["render"][21:24]
            gauss_mask = rendered_i["render"][24:25]
            # flow_image = torch.cat([flow_rendered, torch.zeros_like(flow_rendered[:1])], dim=0)
            # save_image(rendered_i["depth"], f"tmp/depth{i}.jpg")
            # save_image(rendered_i["acc"], f"tmp/acc{i}.jpg")
            # save_image(rgb_rendered, f"tmp/rendered{i}.jpg")
            # save_image(feature_rendered[:3], f"tmp/feat{i}.jpg")
            # save_image(flow_image , f"tmp/flow{i}.jpg")


            rendered_i["feature"] = feature_rendered
            rendered_i["render"] = rgb_rendered # overwrite previous concated rendered
            rendered_i["flow"] = flow_rendered
            rendered_i["normal"] = normal_rendered
            rendered_i["gauss_mask"] = gauss_mask

            # if i == 2:
            # self._xyz.data = xyz_cam_t[i,:,0]
            # self.save_ply(f"gs_ply/warp_t_{i}_camcoord.ply")
                # break

            #  # 检查一下对不对
            for k,v in rendered_i.items():

                if not k in ["viewspace_points", "visibility_filter", "radii"]:
                    value = v[None, ...].permute(0, 2, 3, 1)
                    if i == 0:
                        rendered[k] = value
                    else:
                        rendered[k] = torch.cat([rendered[k], value], dim=0)
                else:
                    if i == 0:
                        rendered[k] = [v]
                    else:
                        rendered[k].append(v)
        
        
        self._viewspace_points_batch = rendered["viewspace_points"]
        self._visibility_filter_batch = rendered["visibility_filter"]
        self._radii_batch = rendered["radii"]

        # rendered = torch.stack(rendered, 0)
                    
        # solve visible pts
        #    return {"render": rendered_image,
            # "viewspace_points": screenspace_points,
            # "visibility_filter" : radii > 0,
            # "radii": radii}


        # feat_dict.update(rendered)
        # num_pixels = H[0] * W[0]
        # feat_dict["rendered"] = rendered['render'].reshape(batch_size, num_pixels, 1, -1)
        # feat_dict["depth"] = rendered['depth'].reshape(batch_size, num_pixels, 1, -1)
        # feat_dict["mask"] = rendered['acc'].reshape(batch_size, num_pixels, 1, -1)
        # feat_dict["feature"] = rendered['feature'].reshape(batch_size, num_pixels, 1, -1)
        # feat_dict["flow"] = rendered['flow'].reshape(batch_size, num_pixels, 1, -1)

        feat_dict["rend_dist"] = rendered['rend_dist']
        feat_dict["rend_normal"] = rendered['rend_normal']
        feat_dict["surf_normal"] = rendered['surf_normal']

        feat_dict["rendered"] = rendered['render']
        feat_dict["surf_depth"] = rendered['surf_depth']
        feat_dict["render_depth_median"] = rendered['render_depth_median']
        feat_dict["render_depth_expected"] = rendered['render_depth_expected']
        feat_dict["mask"] = rendered['acc']
        feat_dict["feature"] = rendered['feature']
        feat_dict["flow"] = rendered['flow']
        feat_dict["normal"] = rendered['normal']
        feat_dict["xyz"] = xyz_reshape
        feat_dict["xyz_cam"] = xyz_cam_t
        feat_dict["eikonal"] = torch.zeros_like(rendered['render'], device=rendered['render'].device, requires_grad=xyz_cam_t.requires_grad)
        feat_dict["gauss_mask"] = rendered['gauss_mask']
    
        aux_dict["feature"] = rendered['feature']
        aux_dict["gauss_mask"] = rendered['gauss_mask']

        if not "no_warp" in samples_dict.keys():
        # cycle loss gen3d的时候应该也用上吧
            cyc_dict = self.cycle_loss(frame_id, inst_id, samples_dict=samples_dict, precomputed_xyz_cam=xyz_cam_t, _xyz_reshaped=xyz_reshape, qt_forward=qt_forward)
            aux_dict.update(cyc_dict)
            # 没有相机空间到正规空间这一步
            # for k in cyc_dict.keys():
            #     if k in backwarp_dict.keys():
            #         # 'skin_entropy', 'delta_skin'
            #         feat_dict[k] = (cyc_dict[k] + backwarp_dict[k]) / 2
            #     else:
            #         # 'cyc_dist'
            #         feat_dict[k] = cyc_dict[k]

            # # jacobian
            # jacob_dict = self.compute_jacobian(
            #     xyz, xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict
            # )
            # feat_dict.update(jacob_dict）
        
        # canonical point
        # feat_dict["xyz"] = xyz
        # feat_dict["xyz_cam"] = xyz_cam
    
        

        # 拿到和2D特征对应的3D点，这个点是在canonical space的
        # 然后把这个点变到相机空间中，并投到相机平面拿到 xy_reproj, xyz_reproj
        # xy_reproj与输入的hxy算配准误差来优化相机warping
        
        # xyz = feat_dict["xyz"].detach()  # don't backprop to cam/dfm fields
        # if self.training:

        
        if "feature" in samples_dict and "feature" in feat_dict : 
            
            # 这个浮在空中的点得单独前向吧【备选，找K近邻代替它,然后就不需要再前向了】
            xyz_matches = self.global_match(samples_dict["feature"], regist_feat_reshape, xyz_reshape, num_grad=8) # 拿到每个像素通过特征匹配得到的正规空间的3D点
            # 将这个正规空间的点warp的时间t然后投影到相机平面
            xy_reproj, xyz_reproj = self.forward_project(
                xyz_matches,
                field2cam,
                Kinv,
                frame_id,
                inst_id,
                samples_dict=samples_dict,
            )

            aux_dict["xyz_matches"] = xyz_matches # 每个像素通过特征匹配得到的正规空间的3D点
            aux_dict["xyz_reproj"] = xyz_reproj # 每个像素通过特征匹配得到的t时刻相机坐标系下的3D点
            if samples_dict["feature"].ndim == 4:
                aux_dict["xy_reproj"] = xy_reproj.reshape(samples_dict["feature"].shape[0], samples_dict["feature"].shape[1], samples_dict["feature"].shape[2], 2)
            else:
                aux_dict["xy_reproj"] = xy_reproj.reshape(samples_dict["feature"].shape[0], samples_dict["feature"].shape[1], 2) # 每个像素通过特征匹配得到的t时刻相机坐标系下的3D点投影到相机平面
            # import ipdb;ipdb.set_trace()
            # feat_dict, deltas是用于渲染图片的，好像不需要了
        


        # import ipdb;ipdb.set_trace()

        # feat_regist 3d
        # find

        # visualize mesh results
        if hxy.shape[0] < 15 and self.opts["debug"]:
            trimesh.Trimesh(vertices=feat_dict["xyz_cam"][0,:,0].detach().cpu().numpy()).export("tmp/loss_reproj_xyz_cam.obj")
            trimesh.Trimesh(vertices=aux_dict["xyz_reproj"][0,:,:].detach().cpu().numpy()).export("tmp/loss_reproj_xyz_reproj.obj")
            hxy_1 = hxy.clone().detach()
            hxy_1[..., :2] /= 255.0
            hxy_reproj = torch.concat([aux_dict["xy_reproj"].clone().detach(), torch.ones_like(aux_dict["xy_reproj"][:, :, :, 0:1])], dim=-1)
            hxy_reproj[..., :2] /= 255.0
            trimesh.Trimesh(vertices=hxy_1[0].reshape(-1,3).detach().cpu().numpy()).export("tmp/loss_reproj_hxy.obj")
            trimesh.Trimesh(vertices=hxy_reproj[0].reshape(-1,3).detach().cpu().numpy()).export("tmp/loss_reproj_hxy_reproj.obj")

        return feat_dict, deltas, aux_dict

    def backward_warp(
        self, frame_id, inst_id, samples_dict={}
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
        # xyz_t, dir = self.cam_to_field(xyz_cam, dir_cam, field2cam)
        raise NotImplementedError
        # not check yet

        xyz_reshape = self._xyz.reshape(1, xyz.shape[0], 1, xyz.shape[1]).expand(frame_id.shape[0], -1, -1, -1)
        rotation_reshape = self._rotation.reshape(1, rotation.shape[0], 1, rotation.shape[1]).expand(frame_id.shape[0], -1, -1, -1) # to align with nerf
        xyz_reshape = xyz_reshape.contiguous()
        rotation_reshape = rotation_reshape.contiguous()

        qt, warp_dict = self.warp(
            xyz_reshape,
            frame_id,
            inst_id,
            backward=True,
            samples_dict=samples_dict,
            return_aux=True,
            return_qt=True,
        )

        q, t = qt
        import ipdb;ipdb.set_trace()
        xyz, rotation = self.apply_qt_to_gaussian(xyz_reshape, rotation_reshape, q, t, frame_id.shape[0])
        # TODO: apply se3 to dir
        backwarp_dict = {"xyz": xyz, "rotation": rotation}
        backwarp_dict.update(warp_dict)
        return backwarp_dict

    def forward_warp(self, xyz, rotation, frame_id, inst_id, samples_dict={}, cache_aux_dict=False):
        """Warp self._xyz_c and self._rotation_c to time t and set self._xyz and self._rotation to it
            camera space

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

        # xyz_reshape_old = xyz.reshape(1, xyz.shape[0], 1, xyz.shape[1]).repeat(frame_id.shape[0], 1, 1, 1) # to align with nerf
        try:
            assert xyz.shape[0] == frame_id.shape[0] and xyz.shape[2] == 1 and xyz.shape[3] == 3 and len(xyz.shape) == 4    
        except:
            import ipdb;ipdb.set_trace()

        # print("point in canonical mean = ", xyz.mean(dim=1))
        qt, aux_dict_t = self.warp(xyz, frame_id, inst_id, samples_dict=samples_dict, return_qt=True, return_aux=True)
        q, t = qt
        xyz_t, rotation_t = self.apply_qt_to_gaussian(xyz, rotation, q, t, frame_id.shape[0])
        # print("point in obj timet mean = ", xyz_t.mean(dim=1))

        if cache_aux_dict:
            self._aux_dict = aux_dict_t
        
        # to cam space
        q, t = samples_dict["field2cam"]
        q = q[:, None].repeat(1, xyz_t.shape[1], 1)
        t = t[:, None].repeat(1, xyz_t.shape[1], 1)
        xyz_cam_t, rotation_cam_t = self.apply_qt_to_gaussian(xyz_t, rotation_t, q, t, frame_id.shape[0])
        # xyz_cam_t, rotation_cam_t = self.field_to_cam(xyz_t, rotation_t, samples_dict["field2cam"])
        # print("point in cam timet mean = ", xyz_cam_t.mean(dim=1))

        return xyz_cam_t, rotation_cam_t, qt

    def global_match(
        self,
        feat_px,
        feat_canonical,
        xyz_canonical,
        num_candidates=2048,
        num_grad=None,
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
        idx = torch.tensor([i for i in range(xyz_canonical.shape[0])], device=score.device)
        if num_grad is not None:
            num_grad = min(num_grad, score.shape[1])
            score, idx = torch.topk(score, num_grad, dim=1, largest=True)
        # score = score * self.logsigma.exp()  # temperature

        # # soft argmin
        # prob = torch.softmax(score, dim=1)
        # xyz_matched = torch.sum(prob.unsqueeze(-1) * xyz_canonical[idx], dim=1)

        # use all candidates
        score = score * self.logsigma.exp()  # temperature
        prob = torch.softmax(score, dim=1)
        xyz_matched = torch.sum(prob.unsqueeze(-1) * xyz_canonical[idx], dim=1)

        xyz_matched = xyz_matched.view(shape[:-1] + (-1,))
        return xyz_matched

    def forward_project(self, xyz, field2cam, Kinv, frame_id, inst_id, samples_dict={}):
        """Project xyz in canonical to image plane

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
        xyz = xyz.reshape(frame_id.shape[0], -1, 1, 3)
        # inst_id = inst_id[..., :1]
        # transform xyz to camera coordinates
        xyz_cam, _, _ = self.forward_warp(
            xyz, None, frame_id, inst_id, samples_dict=samples_dict
        )
        xyz_cam = xyz_cam[:, :, 0]

        # project
        Kmat = Kmatinv(Kinv)
        xy_reproj = pinhole_projection(Kmat, xyz_cam)[..., :2]
        return xy_reproj, xyz_cam


    @train_only_fields
    def cycle_loss(self, frame_id, inst_id, samples_dict, _xyz_reshaped=None, precomputed_xyz_cam=None, qt_forward=None):
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
        cyc_dist = torch.zeros_like(precomputed_xyz_cam[..., :1])
        delta_skin = torch.zeros_like(precomputed_xyz_cam[..., :1])
        skin_entropy = torch.zeros_like(precomputed_xyz_cam[..., :1])
        cyc_dict = {
            "cyc_dist": cyc_dist,
            "delta_skin": delta_skin,
            "skin_entropy": skin_entropy,
        }
        # import ipdb;ipdb.set_trace()
        precomputed_xyz_obj_t = self.cam_to_field(precomputed_xyz_cam, field2cam=samples_dict["field2cam"])
        qt, warp_dict = self.warp(
            precomputed_xyz_obj_t,
            frame_id,
            inst_id,
            backward=True,
            samples_dict=samples_dict,
            return_aux=True,
            return_qt=True,
        )
        q, t = qt
        # TODO compute delta_skin and skin_entropy
        cyc_dict.update(warp_dict)

        # traditional way
        xyz_cycled, rotation_cycled = self.apply_qt_to_gaussian(precomputed_xyz_obj_t, None, q, t, frame_id.shape[0])
        if self.opts["debug"]:
            import trimesh
            trimesh.Trimesh(vertices=xyz_cycled[0,:,0].detach().cpu().numpy()).export("tmp/loss_cyc_xyz_cycled.obj")
            trimesh.Trimesh(vertices=self._xyz.detach().cpu().numpy()).export("tmp/loss_cyc_xyz_canonical.obj")

        cyc_dist = (xyz_cycled - _xyz_reshaped).norm(2, -1, keepdim=True)
        
        # another way to calculate cyc_dist
        # qf, tf = qt_forward
        # qf_inv = _quaternion_conjugate_cuda(qf)
        # tf_inv = -tf
        # cyc_dist = (qf_inv - q).norm(2, -1, keepdim=True) + (tf_inv - t).norm(2, -1, keepdim=True)
        # print("TODO:balance cyc_dist", cyc_dist.mean())

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
        dummy_loss = torch.tensor(0.0, device=self.device)
        return dummy_loss
        
        pts = self.sample_points_aabb(nsample)

        # match the gauss density to the reconstructed density
        density_gauss = self.warp.get_gauss_density(pts)  # (N,1)
        with torch.no_grad():
            density = self.get_opacity[pts]
            density = density / self.logibeta.exp()  # (0,1)

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
        pts = self.sample_points_aabb(nsample)
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
        device = next(self.parameters()).device
        hxy = batch["hxy"]
        frame_id = batch["frameid"]
        inst_id = batch["dataid"]
        H = batch["H"]
        W = batch["W"]

        # get camera pose: (1) read from batch (obj only), (2) read from mlp,
        # (3) read from mlp and apply delta to first frame
        if "field2cam" in batch.keys():
            # quaternion_translation representation, (N, 7)
            field2cam = (batch["field2cam"][..., :4], batch["field2cam"][..., 4:])
            field2cam = (field2cam[0], field2cam[1] * self.logscale.exp())
        else:
            field2cam = self.camera_mlp.get_vals(frame_id)

        # compute near-far
        xmin, ymin, zmin, xmax, ymax, zmax = self.aabb.flatten() 

        corners = [(xmin, ymin, zmin), (xmax, ymin, zmin), 
                (xmin, ymax, zmin), (xmax, ymax, zmin), 
                (xmin, ymin, zmax), (xmax, ymin, zmax), 
                (xmin, ymax, zmax), (xmax, ymax, zmax)]
        if self.training:

            if self.opts['use_wide_near_far']:
                # try:
                # corners = trimesh.bounds.corners(self.proxy_geometry.bounds)
                corners = torch.tensor(corners, dtype=torch.float32, device=device)
                field2cam_mat = quaternion_translation_to_se3(field2cam[0], field2cam[1])
                near_far = get_near_far(corners, field2cam_mat, tol_fac=1.5)
                # except:
                #     near_far = self.near_far.to(device)
                #     near_far = near_far[batch["frameid"]]
            else:
                near_far = self.near_far.to(device)
                near_far = near_far[batch["frameid"]]
        else:
            # corners = trimesh.bounds.corners(self.proxy_geometry.bounds)
            corners = torch.tensor(corners, dtype=torch.float32, device=device)
            field2cam_mat = quaternion_translation_to_se3(field2cam[0], field2cam[1])
            near_far = get_near_far(corners, field2cam_mat, tol_fac=1.5)

        # auxiliary outputs
        samples_dict = {}
        samples_dict["Kinv"] = Kinv
        samples_dict["field2cam"] = field2cam
        samples_dict["frame_id"] = frame_id
        samples_dict["inst_id"] = inst_id
        samples_dict["near_far"] = near_far
        samples_dict["hxy"] = hxy

        samples_dict["H"] = H
        samples_dict["W"] = W
        

        if "feature" in batch.keys():
            samples_dict["feature"] = batch["feature"]
        if "is_gen3d" in batch.keys():
            samples_dict["is_gen3d"] = batch["is_gen3d"]

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
        self.camera_mlp.mlp_init()
        self.update_near_far(beta=0)
        # sdf_fn_torch = self.get_init_sdf_fn()

        # self.geometry_init(sdf_fn_torch)
        if self.fg_motion.startswith("skel"):
            if hasattr(self.warp.articulation, "init_vals"):
                self.warp.articulation.mlp_init()

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


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.

    reference: https://github.com/mikedh/trimesh/issues/507#issuecomment-514973337
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def load_mesh_as_pcd_trimesh(mesh_file, num_points, return_feat=False):
    mesh = as_mesh(trimesh.load_mesh(mesh_file))
    n = num_points
    points = []
    while n > 0:
        p, _ = trimesh.sample.sample_surface_even(mesh, n)
        n -= p.shape[0]
        if n >= 0:
            points.append(p)
        else:
            points.append(p[:n])
    if len(points) > 1:
        points = np.concatenate(points, axis=0)
    else:
        points = points[0]
    # points = torch.from_numpy(points.astype(np.float32))
    # np.random.rand(*x.shape)

    from scipy.spatial import cKDTree
    # Build a KD-tree of the vertices of the mesh
    tree = cKDTree(mesh.vertices)

    # Find the nearest vertex to each sampled point
    _, indices = tree.query(points)

    # Get the colors of the nearest vertices
    colors = mesh.visual.vertex_colors[indices]

    if return_feat:
        feature = np.load(mesh_file.replace("geo.obj", "feat.npy"), allow_pickle=True)
        return points, colors, feature[indices]
    else:
        return points, colors#np.random.rand(*points.shape)


