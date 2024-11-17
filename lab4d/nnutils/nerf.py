# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
from re import T
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pysdf import SDF
import trimesh.transformations as tf
from torch import nn
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
)
from lab4d.utils.render_utils import sample_cam_rays, sample_pdf, compute_weights
from lab4d.utils.torch_utils import compute_gradient

class NeRF(nn.Module):
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
        D=5,
        W=128,
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
        opts=None
    ):
        rtmat = data_info["rtmat"]
        frame_info = data_info["frame_info"]
        frame_offset = data_info["frame_info"]["frame_offset"]
        frame_offset_raw = data_info["frame_info"]["frame_offset_raw"]
        geom_path = data_info["geom_path"]

        super().__init__()

        # dataset info
        self.opts = opts
        self.frame_offset = frame_offset
        self.frame_offset_raw = frame_offset_raw
        self.num_frames = frame_offset[-1]
        self.num_inst = num_inst

        # position and direction embedding
        self.pos_embedding = PosEmbedding(3, num_freq_xyz)
        self.dir_embedding = PosEmbedding(3, num_freq_dir)

        # xyz encoding layers
        # TODO: add option to replace with instNGP
        self.basefield = CondMLP(
            num_inst=self.num_inst,
            D=D,
            W=W,
            in_channels=self.pos_embedding.out_channels,
            inst_channels=inst_channels,
            out_channels=W,
            skips=skips,
            activation=activation,
            final_act=True,
        )

        # color
        self.pos_embedding_color = PosEmbedding(3, num_freq_xyz + 2)
        self.colorfield = CondMLP(
            num_inst=self.num_inst,
            D=2,
            W=W,
            in_channels=self.pos_embedding_color.out_channels,
            inst_channels=inst_channels,
            out_channels=W,
            skips=skips,
            activation=activation,
            final_act=True,
        )
        self.color_act = color_act
        if self.opts["rgb_dirfree"]:
            num_freq_dir = -1
        if self.opts["rgb_timefree"]: #
            appr_channels = 0
            print("rgbtimefree\n"*10)
        # non-directional appearance code (shadow, lighting etc.)
        if appr_channels > 0:
            self.appr_embedding = AppearanceEmbedding(
                frame_info, appr_channels, num_freq_t=appr_num_freq_t
            )
        self.appr_channels = appr_channels

        # output layers
        self.sdf = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
            nn.Linear(W + self.dir_embedding.out_channels + self.appr_channels, W // 2),
            activation,
            nn.Linear(W // 2, 3),
        )

        self.in_channels = 6
        self.out_channels = 4

        beta = torch.tensor([init_beta])
        self.logibeta = nn.Parameter(-beta.log())  # beta: transparency

        scale = torch.tensor([init_scale])  # scale of the field
        self.logscale = nn.Parameter(scale.log())

        # camera pose: field to camera
        rtmat[..., :3, 3] *= init_scale
        self.camera_mlp = CameraMLP(rtmat, frame_info=frame_info)

        # visibility mlp
        self.vis_mlp = VisField(self.num_inst)

        # load initial mesh

        self.init_proxy(geom_path, init_scale)
        self.register_buffer("aabb", torch.zeros(2, 3))
        self.update_aabb(beta=0)

        # non-parameters are not synchronized
        self.register_buffer(
            "near_far", torch.zeros(frame_offset_raw[-1], 2), persistent=False
        )

    def forward(self, xyz, dir=None, frame_id=None, inst_id=None, get_density=True):
        """
        Args:
            xyz: (M,N,D,3) Points along ray in object canonical space
            dir: (M,N,D,3) Ray direction in object canonical space
            frame_id: (M,) Frame id. If None, render at all frames
            inst_id: (M,) Instance id. If None, render for the average instance
        Returns:
            rgb: (M,N,D,3) Rendered RGB
            sigma: (M,N,D,1) If get_density=True, return density. Otherwise
                return signed distance (negative inside)
        """

        if frame_id is not None:
            assert frame_id.ndim == 1
        if inst_id is not None:
            assert inst_id.ndim == 1
        xyz_embed = self.pos_embedding(xyz)
        xyz_feat = self.basefield(xyz_embed, inst_id)

        sdf = self.sdf(xyz_feat)  # negative inside, positive outside
        if get_density:
            ibeta = self.logibeta.exp()
            # density = torch.sigmoid(-sdf * ibeta) * ibeta  # neus
            density = (
                0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibeta)
            ) * ibeta  # volsdf
            out = density
        else:
            out = sdf

        if dir is not None:
            dir_embed = self.dir_embedding(dir)
            if self.appr_channels > 0:
                appr_embed = self.appr_embedding.get_vals(frame_id)
                appr_embed = appr_embed[:, None, None].expand(
                    dir_embed.shape[:-1] + (appr_embed.shape[-1],)
                )
                appr_embed = torch.cat([dir_embed, appr_embed], -1)
            else:
                appr_embed = dir_embed

            xyz_embed = self.pos_embedding_color(xyz)
            xyz_feat = xyz_feat + self.colorfield(xyz_embed, inst_id)

            rgb = self.rgb(torch.cat([xyz_feat, appr_embed], -1))
            if self.color_act:
                rgb = rgb.sigmoid()
            out = rgb, out
        return out

    def get_init_sdf_fn(self):
        """Initialize signed distance function from mesh geometry

        Returns:
            sdf_fn_torch (Function): Signed distance function
        """
        sdf_fn_numpy = SDF(self.proxy_geometry.vertices, self.proxy_geometry.faces)

        def sdf_fn_torch(pts):
            sdf = -sdf_fn_numpy(pts.cpu().numpy())[:, None]  # negative inside
            sdf = torch.tensor(sdf, device=pts.device, dtype=pts.dtype)
            return sdf

        return sdf_fn_torch

    def mlp_init(self):
        """Initialize camera transforms and geometry from external priors"""
        self.camera_mlp.mlp_init()
        self.update_near_far(beta=0)
        sdf_fn_torch = self.get_init_sdf_fn()
        self.sdf_fn_torch = self.get_init_sdf_fn('other')

        self.geometry_init(sdf_fn_torch)

    def init_proxy(self, geom_path, init_scale):
        """Initialize the geometry from a mesh

        Args:
            geom_path (str): Initial shape mesh
            init_scale (float): Geometry scale factor
        """
        geom_path = "/pfs/mt-1oY5F7/liuguangce/program-vidu/4dcode-8.3/objs for lab4d-neus/pokemon/tmp4y7ghih_.obj"
        init_scale = 0.005
        mesh = trimesh.load(geom_path)
        rotation_matrix = tf.euler_matrix(-np.pi, 0, 0)
        mesh.apply_transform(rotation_matrix)
        mesh.vertices = mesh.vertices * init_scale
        self.proxy_geometry = mesh

    def geometry_init(self, sdf_fn, nsample=5000):
        """Initialize SDF using tsdf-fused geometry if radius is not given.
        Otherwise, initialize sdf using a unit sphere

        Args:
            sdf_fn (Function): Maps vertices to signed distances
            nsample (int): Number of samples
        """
        device = next(self.parameters()).device
        # setup optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # optimize
        for i in range(1000):
            optimizer.zero_grad()

            # sample points and gt sdf
            inst_id = torch.randint(0, self.num_inst, (nsample,), device=device)

            if not self.opts["lab4d_init_mesh"] == "":
                mesh = trimesh.load(self.opts["lab4d_init_mesh"])
                pts = trimesh.sample.sample_surface_even(mesh, nsample)
                # 不对劲
                raise NotImplementedError
            elif False:
                with np.load('sdf_pc.npz', allow_pickle=True) as data:
                    # 从加载的数据中获取保存的数组
                    # arr1 = data['volume_contour']
                    pts = data['slice_contour']
                    # randomly pick nsample points
                    pts = pts[np.random.choice(pts.shape[0], nsample, replace=False)]
                    pts = torch.tensor(pts, dtype=torch.float32, device="cuda") * 0.0001
                    sdf_gt = torch.zeros_like(pts[:, 0:1])
                    inst_id = torch.randint(0, self.num_inst, (pts.shape[0],), device=device)
            else:
                # sample points
                # import pdb;pdb.set_trace()
                # pts_mesh = torch.tensor(self.proxy_geometry.sample(50000, return_index=False), dtype=torch.float32, device=device)
                # print("max is :",self.proxy_geometry.vertices.max())
                pts = self.sample_points_aabb(nsample, extend_factor=0.25)
                # pts = torch.cat((pts_mesh,pts_rand),dim = 0)
                # get sdf from proxy geometry
                sdf_gt = sdf_fn(pts)
            # import pdb;pdb.set_trace()
            # evaluate sdf loss
            sdf = self.forward(pts, inst_id=inst_id, get_density=False)
            # scale = align_vectors(sdf, sdf_gt)
            # print("scale is ",scale)
            # scale = 1.0
            sdf_loss = (sdf - sdf_gt).pow(2).mean()
            # sdf_loss = (sdf * scale.detach() - sdf_gt).pow(2).mean()

            # evaluate visibility loss
            vis = self.vis_mlp(pts, inst_id=inst_id)
            vis_loss = -F.logsigmoid(vis).mean()
            vis_loss = vis_loss * 0.01

            # evaluate eikonal loss
            eikonal_loss = self.compute_eikonal(pts[:, None, None], inst_id=inst_id)
            eikonal_loss = eikonal_loss[eikonal_loss > 0].mean()
            eikonal_loss = eikonal_loss * 1e-5

            total_loss = sdf_loss + vis_loss + eikonal_loss
            total_loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"iter {i}, scale  ,loss {total_loss.item()}")

    def update_proxy(self):
        """Extract proxy geometry using marching cubes
            xyz: (M,N,D,3) Points along rays in object canonical space
            dir: (M,N,D,3) Ray directions in object canonical space
            frame_id: (M,) Frame id. If None, render at all frames
            inst_id: (M,) Instance id. If None, render for the average instance"""
        mesh = self.extract_canonical_mesh(level=0.005)
        print("scale is :", self.logscale)
        # save mesh
        # mesh.export("proxy.obj")
        # import ipdb; ipdb.set_trace()
        xyz = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
        def sdf_fn(xyz):
            return self.forward(xyz, get_density=False)
        g = compute_gradient(sdf_fn, xyz)[..., 0]
        frame_id = torch.tensor([10], device="cuda")
        inst_id = torch.tensor([0], device="cuda")
        # import ipdb;ipdb.set_trace() # HW是否正确
        with torch.no_grad():
            colors = self.query_nerf(xyz[None, None, :, :], g[None, None, :, :], frame_id, inst_id)["rgb"].reshape(-1, 3).detach().cpu().numpy()
            
            xyz_embed = self.feat_pos_embedding(xyz[None, None, :, :])
            feature = self.feature_field(xyz_embed)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            feature = feature.reshape(-1, 16).detach().cpu().numpy()

        # mesh.vertex_attributes['feature'] = feature
        mesh.metadata['feature'] = feature
        mesh.visual.vertex_colors = colors
        # import ipdb;ipdb.set_trace()
        
        if mesh is not None:
            self.proxy_geometry = mesh

    @torch.no_grad()
    def extract_canonical_mesh(
        self,
        grid_size=256,
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
              region.
            use_extend_aabb (bool): If True, extend aabb by 50% to get a loose proxy.
              Used at training time.
        Returns:
            mesh (Trimesh): Extracted mesh
        """
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

    def update_aabb(self, beta=0.9):
        """Update axis-aligned bounding box by interpolating with the current
        proxy geometry's bounds

        Args:
            beta (float): Interpolation factor between previous/current values
        """
        device = self.aabb.device
        bounds = self.proxy_geometry.bounds
        if bounds is not None:
            aabb = torch.tensor(bounds, dtype=torch.float32, device=device)
            self.aabb = self.aabb * beta + aabb * (1 - beta)

    def update_near_far(self, beta=0.9):
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

    def sample_points_aabb(self, nsample, extend_factor=1.0):
        """Sample points within axis-aligned bounding box

        Args:
            nsample (int): Number of samples
            extend_factor (float): Extend aabb along each side by factor of
                the previous size
        Returns:
            pts: (nsample, 3) Sampled points
        """
        device = next(self.parameters()).device
        aabb = extend_aabb(self.aabb, factor=extend_factor)
        pts = (
            torch.rand(nsample, 3, dtype=torch.float32, device=device)
            * (aabb[1:] - aabb[:1])
            + aabb[:1]
        )
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
        pts = self.sample_points_aabb(nsample)
        inst_id = torch.randint(0, self.num_inst, (nsample,), device=device)

        # evaluate loss
        vis = self.vis_mlp(pts, inst_id=inst_id)
        loss = -F.logsigmoid(-vis).mean()
        return loss

    def compute_eikonal(self, xyz, inst_id=None, sample_ratio=16):
        """Compute eikonal loss

        Args:
            xyz: (M,N,D,3) Input coordinates in canonical space
            inst_id: (M,) Instance id, or None to use the average instance
            sample_ratio (int): Fraction to subsample to make it more efficient
        Returns:
            eikonal_loss: (M,N,D,1) Squared magnitude of SDF gradient
        """
        M, N, D, _ = xyz.shape
        xyz = xyz.reshape(-1, D, 3)
        sample_size = xyz.shape[0] // sample_ratio
        if sample_size < 1:
            sample_size = 1
        if inst_id is not None:
            inst_id = inst_id[:, None].expand(-1, N)
            inst_id = inst_id.reshape(-1)
        eikonal_loss = torch.zeros_like(xyz[..., 0])

        # subsample to make it more efficient
        if M * N > sample_size:
            probs = torch.ones(M * N)
            rand_inds = torch.multinomial(probs, sample_size, replacement=False)
            xyz = xyz[rand_inds]
            if inst_id is not None:
                inst_id = inst_id[rand_inds]
        else:
            rand_inds = Ellipsis

        xyz = xyz.detach()
        inst_id = inst_id.detach() if inst_id is not None else None
        fn_sdf = lambda x: self.forward(x, inst_id=inst_id, get_density=False)
        g = compute_gradient(fn_sdf, xyz)[..., 0]

        eikonal_loss[rand_inds] = (g.norm(2, dim=-1) - 1) ** 2
        eikonal_loss = eikonal_loss.reshape(M, N, D, 1)
        return eikonal_loss

    def compute_normal(self, xyz_cam, dir_cam, field2cam, frame_id=None, inst_id=None, samples_dict={}):
        """Compute eikonal loss and normals in camera space

        Args:
            xyz_cam: (M,N,D,3) Points along rays in camera space
            dir_cam: (M,N,D,3) Ray directions in camera space
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
            frame_id: (M,) Frame id to query articulations, or None to use all frames
            inst_id: (M,) Instance id, or None to use the average instance
            samples_dict (Dict): Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            normal: (M,N,D,3) Normal vector field in camera space
        """
        M, N, D, _ = xyz_cam.shape

        def fn_sdf(xyz_cam):
            xyz = self.backward_warp(
                xyz_cam,
                dir_cam,
                field2cam,
                frame_id=frame_id,
                inst_id=inst_id,
                samples_dict=samples_dict,
            )["xyz"]
            sdf = self.forward(xyz, inst_id=inst_id, get_density=False)
            return sdf

        g = compute_gradient(fn_sdf, xyz_cam)[..., 0]

        eikonal = (g.norm(2, dim=-1, keepdim=True) - 1) ** 2
        normal = torch.nn.functional.normalize(g, dim=-1)

        # Multiply by [1, -1, -1] to match normal conventions from ECON
        # https://github.com/YuliangXiu/ECON/blob/d98e9cbc96c31ecaa696267a072cdd5ef78d14b8/apps/infer.py#L257
        normal = normal * torch.tensor([1, -1, -1], device="cuda")

        return eikonal, normal

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

    def get_samples(self, Kinv, batch):
        """Compute time-dependent camera and articulation parameters.

        Args:
            Kinv: (N,3,3) Inverse of camera matrix
            batch (Dict): Batch of inputs. Keys: "dataid", "frameid_sub",
                "crop2raw", "feature", "hxy", and "frameid"
        Returns:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,SE(3)), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,2), and
                "feature" (M,N,16).
        """

        device = next(self.parameters()).device
        hxy = batch["hxy"]
        frame_id = batch["frameid"]
        inst_id = batch["dataid"]
        # get camera pose: (1) read from batch (obj only), (2) read from mlp,
        # (3) read from mlp and apply delta to first frame
        if "field2cam" in batch.keys():
            # quaternion_translation representation, (N, 7)
            field2cam = (batch["field2cam"][..., :4], batch["field2cam"][..., 4:])
            field2cam = (field2cam[0], field2cam[1] * self.logscale.exp())
        else:
            field2cam = self.camera_mlp.get_vals(frame_id)

        # compute near-far
        if self.training:
            if self.opts['use_wide_near_far']:
                # try:
                corners = trimesh.bounds.corners(self.proxy_geometry.bounds)
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
            corners = trimesh.bounds.corners(self.proxy_geometry.bounds)
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
        if "feature" in batch.keys():
            samples_dict["feature"] = batch["feature"]
        if "is_gen3d" in batch.keys():
            samples_dict["is_gen3d"] = batch["is_gen3d"]
        return samples_dict

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
                "depth" (M,1,D,1)
            deltas: (M,N,D,1) Distance along rays between adjacent samples
            aux_dict (Dict): Auxiliary neural field outputs. Used in Deformable
        """
        Kinv = samples_dict["Kinv"]  # (M,3,3)
        field2cam = samples_dict["field2cam"]  # (M,SE(3))
        frame_id = samples_dict["frame_id"]  # (M,)
        inst_id = samples_dict["inst_id"]  # (M,)
        near_far = samples_dict["near_far"]  # (M,2)
        hxy = samples_dict["hxy"]  # (M,N,2)

        # sample camera space rays
        if not self.training:
            # importance sampling
            xyz_cam, dir_cam, deltas, depth = self.importance_sampling(
                hxy,
                Kinv,
                near_far,
                field2cam,
                frame_id,
                inst_id,
                samples_dict,
                n_depth=128,
            )

        else:
            if "guided_depth" in samples_dict.keys():
                depth_guidance = samples_dict["guided_depth"]
            else:
                depth_guidance = None
            xyz_cam, dir_cam, deltas, depth = sample_cam_rays(
                hxy, Kinv, near_far, perturb=False, depth=depth_guidance
            )  # (M, N, D, x)
        
        # import ipdb;ipdb.set_trace()
        # Kmat = Kmatinv(Kinv[None, ...])[0][0]
        # FoVx = 2 * torch.arctan(Kmat[0, 2] / Kmat[0, 0])
        # FoVy = 2 * torch.arctan(Kmat[1, 2] / Kmat[1, 1])
        # print("FoVx, FoVy", FoVx * 180 / np.pi, FoVy * 180 / np.pi)
        # Kmat = Kmatinv(Kinv[None, ...])[0][1]
        # FoVx = 2 * torch.arctan(Kmat[0, 2] / Kmat[0, 0])
        # FoVy = 2 * torch.arctan(Kmat[1, 2] / Kmat[1, 1])
        # print("FoVx, FoVy", FoVx * 180 / np.pi, FoVy * 180 / np.pi)

        if "no_warp" in samples_dict.keys():
            xyz, dir = self.cam_to_field(xyz_cam, dir_cam, field2cam)
            xyz_t = xyz
        else:
            # backward warping
            backwarp_dict = self.backward_warp(
                xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict=samples_dict
            )
            xyz = backwarp_dict["xyz"]
            dir = backwarp_dict["dir"]
            xyz_t = backwarp_dict["xyz_t"]

        # visibility
        vis_score = self.vis_mlp(xyz, inst_id=inst_id)  # (M, N, D, 1)

        # compute valid_indices to speed up querying fields
        if self.training:
            valid_idx = None
        else:
            valid_idx = self.get_valid_idx(xyz, xyz_t, vis_score, samples_dict)

        # NeRF
        feat_dict = self.query_nerf(xyz, dir, frame_id, inst_id, valid_idx=valid_idx)

        # visibility
        feat_dict["vis"] = vis_score
        if not "is_gen3d" in samples_dict.keys():
            if not self.opts["two_branch"]:
                # flow
                flow_dict = self.compute_flow(
                    hxy,
                    xyz,
                    frame_id,
                    inst_id,
                    field2cam,
                    Kinv,
                    samples_dict,
                    flow_thresh=flow_thresh,
                )
                feat_dict.update(flow_dict)

                # cycle loss
                cyc_dict = self.cycle_loss(
                    xyz, xyz_t, frame_id, inst_id, samples_dict=samples_dict
                )
                for k in cyc_dict.keys():
                    if k in backwarp_dict.keys():
                        # 'skin_entropy', 'delta_skin'
                        feat_dict[k] = (cyc_dict[k] + backwarp_dict[k]) / 2
                    else:
                        # 'cyc_dist'
                        feat_dict[k] = cyc_dict[k]

            # jacobian
            jacob_dict = self.compute_jacobian(
                xyz, xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict
            )
            feat_dict.update(jacob_dict)

        # canonical point
        feat_dict["xyz"] = xyz
        feat_dict["xyz_cam"] = xyz_cam

        # depth
        feat_dict["depth"] = depth / self.logscale.exp()  # world scale

        # auxiliary outputs
        aux_dict = {}
        return feat_dict, deltas, aux_dict

    @torch.no_grad()
    def importance_sampling(
        self,
        hxy,
        Kinv,
        near_far,
        field2cam,
        frame_id,
        inst_id,
        samples_dict,
        n_depth=64,
    ):
        """
        importance sampling coarse
        """
        # sample camera space rays
        xyz_cam, dir_cam, deltas, depth = sample_cam_rays(
            hxy, Kinv, near_far, perturb=False, n_depth=n_depth // 2
        )  # (M, N, D, x)

        # backward warping
        xyz = self.backward_warp(
            xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict=samples_dict
        )["xyz"]

        # get pdf
        density = self.forward(
            xyz,
            dir=None,
            frame_id=frame_id,
            inst_id=inst_id,
        )  # (M, N, D, x)
        weights, _ = compute_weights(density, deltas)  # (M, N, D, x)

        depth_mid = 0.5 * (depth[:, :, :-1] + depth[:, :, 1:])  # (M, N, D-1)
        is_det = not self.training
        depth_mid = depth_mid.view(-1, n_depth // 2 - 1)
        weights = weights.view(-1, n_depth // 2)

        depth_ = sample_pdf(
            depth_mid, weights[:, 1:-1], n_depth // 2, det=is_det
        ).detach()
        depth_ = depth_.reshape(depth.shape)
        # detach so that grad doesn't propogate to weights_sampled from here

        depth, _ = torch.sort(torch.cat([depth, depth_], -2), -2)  # (M, N, D)

        # sample camera space rays
        xyz_cam, dir_cam, deltas, depth = sample_cam_rays(
            hxy, Kinv, near_far, depth=depth, perturb=False
        )

        return xyz_cam, dir_cam, deltas, depth

    def compute_jacobian(self, xyz, xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict):
        """Compute eikonal and normal fields from Jacobian of SDF

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space. Only for training
            xyz_cam: (M,N,D,3) Points along rays in camera space. Only for rendering
            dir_cam: (M,N,D,3) Ray directions in camera space. Only for rendering
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform. Only for rendering
            frame_id: (M,) Frame id to query articulations, or None to use all frames.
                Only for rendering
            inst_id: (M,) Instance id. If None, compute for the average instance
            samples_dict (Dict): Time-dependent bone articulations. Only for rendering. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            jacob_dict (Dict): Jacobian fields. Keys: "eikonal" (M,N,D,1). Only when
                rendering, "normal" (M,N,D,3)
        """
        jacob_dict = {}
        if self.training and not self.opts["two_branch"]:
            # For efficiency, compute subsampled eikonal loss in canonical space
            jacob_dict["eikonal"] = self.compute_eikonal(xyz, inst_id=inst_id)
        else:
            # For rendering, compute full eikonal loss and normals in camera space
            jacob_dict["eikonal"], jacob_dict["normal"] = self.compute_normal(
                xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict
            )
        return jacob_dict

    def query_nerf(self, xyz, dir, frame_id, inst_id, valid_idx=None):
        """Neural radiance field rendering

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space
            dir: (M,N,D,3) Ray directions in object canonical space
            frame_id: (M,) Frame id. If None, render at all frames
            inst_id: (M,) Instance id. If None, render for the average instance
            valid_idx: (M,N,D) Mask of whether each point is visible to camera
        Returns:
            field_dict (Dict): Field outputs. Keys: "rgb" (M,N,D,3),
                "density" (M,N,D,1), and "density_{fg,bg}" (M,N,D,1)
        """
        if valid_idx is not None:
            if valid_idx.sum() == 0:
                field_dict = {
                    "rgb": torch.zeros(valid_idx.shape + (3,), device=xyz.device),
                    "density": torch.zeros(valid_idx.shape + (1,), device=xyz.device),
                    "density_%s"
                    % self.category: torch.zeros(
                        valid_idx.shape + (1,), device=xyz.device
                    ),
                }
                return field_dict
            # reshape
            shape = xyz.shape
            xyz = xyz[valid_idx][:, None, None]  # MND,1,1,3
            dir = dir[valid_idx][:, None, None]
            frame_id = frame_id[:, None, None].expand(shape[:3])[valid_idx]
            inst_id = inst_id[:, None, None].expand(shape[:3])[valid_idx]

        rgb, density = self.forward(
            xyz,
            dir=dir,
            frame_id=frame_id,
            inst_id=inst_id,
        )  # (M, N, D, x)

        # reshape
        field_dict = {
            "rgb": rgb,
            "density": density,
            "density_%s" % self.category: density,
        }

        if valid_idx is not None:
            for k, v in field_dict.items():
                tmpv = torch.zeros(valid_idx.shape + (v.shape[-1],), device=v.device)
                tmpv[valid_idx] = v.view(-1, v.shape[-1])
                field_dict[k] = tmpv
        return field_dict

    @staticmethod
    def cam_to_field(xyz_cam, dir_cam, field2cam):
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
        shape = xyz_cam.shape
        cam2field = quaternion_translation_inverse(field2cam[0], field2cam[1])
        cam2field = (
            cam2field[0][:, None, None].expand(shape[:-1] + (4,)).clone(),
            cam2field[1][:, None, None].expand(shape[:-1] + (3,)).clone(),
        )
        xyz = apply_se3mat(cam2field, xyz_cam)
        cam2field = (cam2field[0], torch.zeros_like(cam2field[1]))
        dir = apply_se3mat(cam2field, dir_cam)
        return xyz, dir

    def field_to_cam(self, xyz, field2cam):
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
        return xyz_cam_next

    def backward_warp(
        self, xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict={}
    ):
        """Warp points from camera space to object canonical space

        Args:
            xyz_cam: (M,N,D,3) Points along rays in camera space
            dir_cam: (M,N,D,3) Ray directions in camera space
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            samples_dict (Dict): Only used in Deformable

        Returns:
            xyz: (M,N,D,3) Points along rays in object canonical space
            dir: (M,N,D,3) Ray directions in object canonical space
            xyz_t: (M,N,D,3) Points along rays in object time-t space. Same
                as canonical space for static fields
        """
        xyz, dir = self.cam_to_field(xyz_cam, dir_cam, field2cam)

        backwarp_dict = {"xyz": xyz, "dir": dir, "xyz_t": xyz}
        return backwarp_dict

    def forward_warp(self, xyz, field2cam, frame_id, inst_id, samples_dict={}):
        """Warp points from object canonical space to camera space

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            samples_dict (Dict): Only used in Deformable

        Returns:
            xyz_cam: (M,N,D,3) Points along rays in camera space
        """
        xyz_cam = self.field_to_cam(xyz, field2cam)
        return xyz_cam

    def cycle_loss(self, xyz, xyz_t, frame_id, inst_id, samples_dict={}):
        """Compute cycle-consistency loss between points in object canonical
        space, and points that have been warped backward and then forward

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space
            xyz_t: (M,N,D,3) Points along rays in object time-t space
            frame_id: (M,) Frame id. If None, render at all frames
            inst_id: (M,) Instance id. If None, render for the average instance
            samples_dict (Dict): Used in Deformable

        Returns:
            cyc_dict (Dict): Cycle consistency loss. Keys: "cyc_dist" (M,N,D,1)
        """
        cyc_dist = torch.zeros_like(xyz[..., :1])
        delta_skin = torch.zeros_like(xyz[..., :1])
        skin_entropy = torch.zeros_like(xyz[..., :1])
        cyc_dict = {
            "cyc_dist": cyc_dist,
            "delta_skin": delta_skin,
            "skin_entropy": skin_entropy,
        }
        return cyc_dict

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
            return tuple([NeRF.flip_pair(t) for t in tensor])
        elif isinstance(tensor, dict):
            return {k: NeRF.flip_pair(v) for k, v in tensor.items()}

    @train_only_fields
    def compute_flow(
        self,
        hxy,
        xyz,
        frame_id,
        inst_id,
        field2cam,
        Kinv,
        samples_dict,
        flow_thresh=None,
    ):
        """Compute optical flow proposal by (1) projecting to next camera
        image plane, and (2) taking difference with xy

        Args:
            hxy: (M,N,D,3) Homogeneous pixel coordinates on the image plane
            xyz: (M,N,D,3) Canonical field coordinates
            Kinv: (M,3,3) Inverse of camera intrinsics
            flow_thresh (float): Threshold for flow magnitude

        Returns:
            flow: (M,N,D,2) Optical flow proposal
        """
        # flip the frame id
        frame_id_next = self.flip_pair(frame_id)
        field2cam_next = (self.flip_pair(field2cam[0]), self.flip_pair(field2cam[1]))
        Kinv_next = self.flip_pair(Kinv)
        samples_dict_next = self.flip_pair(samples_dict)

        # forward warp points to camera space
        xyz_cam_next = self.forward_warp(
            xyz, field2cam_next, frame_id_next, inst_id, samples_dict=samples_dict_next
        )

        # project to next camera image plane
        Kmat_next = Kmatinv(Kinv_next)  # (M,1,1,3,3) @ (M,N,D,3) = (M,N,D,3)
        hxy_next = pinhole_projection(Kmat_next, xyz_cam_next)

        # compute 2d difference
        flow = (hxy_next - hxy.unsqueeze(-2))[..., :2]
        xyz_valid = xyz_cam_next[..., -1:] > 1e-6
        if flow_thresh is not None:
            flow_thresh = float(flow_thresh)
            xyz_valid = xyz_valid & (flow.norm(dim=-1, keepdim=True) < flow_thresh)

        # import ipdb;ipdb.set_trace()
        flow = torch.cat([flow, xyz_valid.float()], dim=-1)

        flow_dict = {"flow": flow}
        return flow_dict

    def cam_prior_loss(self):
        """Encourage camera transforms over time to match external priors.

        Returns:
            loss: (0,) Mean squared error of camera SE(3) transforms to priors
        """
        loss = self.camera_mlp.compute_distance_to_prior()
        return loss
