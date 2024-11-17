# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import trimesh
from torch import nn
from torch.nn import functional as F

from lab4d.nnutils.deformable import Deformable
from lab4d.nnutils.warping import SkinningWarp, create_warp
from lab4d.utils.decorator import train_only_fields
from lab4d.utils.geom_utils import extend_aabb
from lab4d.utils.loss_utils import align_vectors
from lab4d.engine.train_utils import get_local_rank
from lab4d.utils.render_utils import sample_cam_rays, sample_pdf, compute_weights

from flexicube_utils.flexicubes_geometry import FlexiCubesGeometry
from flexicube_utils.utils import PerspectiveCamera, NeuralRender

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

class DeformableFlexicube(Deformable):
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
        num_freq_dir=-1,
        appr_channels=0,
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



        # Renderer we used.

        # self.renderer = FlexicubeRenderer(tet_grid_size=opts["tet_grid_size"], scale=input.scale)
        self.flexicubes = FlexiCubesGeometry(grid_res=opts["tet_grid_size"], scale=opts["scale"], device='cuda', renderer=None,)
        self.flexicubes.mesh_renderer = NeuralRender(self.device)
        
        self.weightMlp = nn.Sequential(
                        nn.Linear(3 * 32 * 8, 64),
                        nn.SiLU(),
                        nn.Linear(64, 21))

        self.deformMlp = nn.Sequential(
                        nn.Linear(3 * 32 * 8, 64, bias=True),
                        nn.ReLU(),  
                        nn.Linear(64, 64, bias=True),
                        nn.ReLU(),
                        nn.Linear(64, 3, bias=True),)
        
        # init the output of weightMlp to be 1 and deformMlp output to be 0
        tet_verts = self.flexicubes.verts.unsqueeze(0)
        optimizer = torch.optim.Adam(list(self.weightMlp.parameters()) + list(self.deformMlp.parameters()), lr=0.01)
        for i in range(10000):
            xyz_embed = self.pos_embedding(tet_verts)
            xyz_feat = self.basefield(xyz_embed, None)
            weight = self.weightMlp(xyz_feat)
            deform = self.deformMlp(xyz_feat)
            loss1 = F.mse_loss(weight, torch.ones_like(weight))
            loss2 = F.mse_loss(deform, torch.zeros_like(deform))
            print(f"loss1: {loss1}, loss2: {loss2}")
            optimizer.zero_grad()
            loss1.backward()
            loss2.backward()
            optimizer.step()


    def forward(self, xyz=None, dir=None, frame_id=None, inst_id=None, get_density=True):
        """
        Calculate verts, faces, rgb, and feature using flexcube
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

        # tet_verts = self.renderer.flexicubes.verts.unsqueeze(0)
        # tet_indices = self.renderer.flexicubes.indices

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
            # Do not compute color this way, for xyz to query sdf and color is different for flexicube
            raise NotImplementedError  
                
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

    def export_mesh_wt_uv(self, ctx, data, out_dir, ind, device, res, tri_fea_2=None):
        raise NotImplementedError
        mesh_v = data['verts'].squeeze().cpu().numpy()
        mesh_pos_idx = data['faces'].squeeze().cpu().numpy()

        def interpolate(attr, rast, attr_idx, rast_db=None):
            return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
                                  diff_attrs=None if rast_db is None else 'all')

        vmapping, indices, uvs = xatlas.parametrize(mesh_v, mesh_pos_idx)

        mesh_v = torch.tensor(mesh_v, dtype=torch.float32, device=device)
        mesh_pos_idx = torch.tensor(mesh_pos_idx, dtype=torch.int64, device=device)

        # Convert to tensors
        indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

        uvs = torch.tensor(uvs, dtype=torch.float32, device=mesh_v.device)
        mesh_tex_idx = torch.tensor(indices_int64, dtype=torch.int64, device=mesh_v.device)
        # mesh_v_tex. ture
        uv_clip = uvs[None, ...] * 2.0 - 1.0

        # pad to four component coordinate
        uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[..., 0:1]), torch.ones_like(uv_clip[..., 0:1])), dim=-1)

        # rasterize
        rast, _ = dr.rasterize(ctx, uv_clip4, mesh_tex_idx.int(), res)

        # Interpolate world space position
        gb_pos, _ = interpolate(mesh_v[None, ...], rast, mesh_pos_idx.int())
        mask = rast[..., 3:4] > 0

        # return uvs, mesh_tex_idx, gb_pos, mask
        gb_pos_unsqz = gb_pos.view(-1, 3)
        mask_unsqz = mask.view(-1)
        tex_unsqz = torch.zeros_like(gb_pos_unsqz) + 1

        gb_mask_pos = gb_pos_unsqz[mask_unsqz]

        gb_mask_pos = gb_mask_pos[None, ]

        with torch.no_grad():

            dec_verts = self.decoder(tri_fea_2, gb_mask_pos)
            colors = self.rgbMlp(dec_verts).squeeze()

        # Expect predicted colors value range from [-1, 1]
        lo, hi = (-1, 1)
        colors = (colors - lo) * (255 / (hi - lo))
        colors = colors.clip(0, 255)

        tex_unsqz[mask_unsqz] = colors

        tex = tex_unsqz.view(res + (3,))

        verts = mesh_v.squeeze().cpu().numpy()
        faces = mesh_pos_idx[..., [2, 1, 0]].squeeze().cpu().numpy()
        # faces = mesh_pos_idx
        # faces = faces.detach().cpu().numpy()
        # faces = faces[..., [2, 1, 0]]
        indices = indices[..., [2, 1, 0]]

        # xatlas.export(f"{out_dir}/{ind}.obj", verts[vmapping], indices, uvs)
        matname = f'{out_dir}.mtl'
        # matname = f'{out_dir}/{ind}.mtl'
        fid = open(matname, 'w')
        fid.write('newmtl material_0\n')
        fid.write('Kd 1 1 1\n')
        fid.write('Ka 1 1 1\n')
        # fid.write('Ks 0 0 0\n')
        fid.write('Ks 0.4 0.4 0.4\n')
        fid.write('Ns 10\n')
        fid.write('illum 2\n')
        fid.write(f'map_Kd {out_dir.split("/")[-1]}.png\n')
        fid.close()

        fid = open(f'{out_dir}.obj', 'w')
        # fid = open(f'{out_dir}/{ind}.obj', 'w')
        fid.write('mtllib %s.mtl\n' % out_dir.split("/")[-1])

        for pidx, p in enumerate(verts):
            pp = p
            fid.write('v %f %f %f\n' % (pp[0], pp[2], - pp[1]))

        for pidx, p in enumerate(uvs):
            pp = p
            fid.write('vt %f %f\n' % (pp[0], 1 - pp[1]))

        fid.write('usemtl material_0\n')
        for i, f in enumerate(faces):
            f1 = f + 1
            f2 = indices[i] + 1
            fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
        fid.close()

        img = np.asarray(tex.data.cpu().numpy(), dtype=np.float32)
        mask = np.sum(img.astype(float), axis=-1, keepdims=True)
        mask = (mask <= 3.0).astype(float)
        kernel = np.ones((3, 3), 'uint8')
        dilate_img = cv2.dilate(img, kernel, iterations=1)
        img = img * (1 - mask) + dilate_img * mask
        img = img.clip(0, 255).astype(np.uint8)

        cv2.imwrite(f'{out_dir}.png', img[..., [2, 1, 0]])
        # cv2.imwrite(f'{out_dir}/{ind}.png', img[..., [2, 1, 0]])


    def update_proxy(self, verts=None, faces=None, rgb=None,):
        if verts is None and faces is None and rgb is None:
            verts, faces, rgb = self.DDMC()
        rgb = (rgb * 0.5 + 0.5).clip(0, 1)
        verts = verts.squeeze().cpu().numpy()
        faces = faces[..., [2, 1, 0]].squeeze().cpu().numpy()
        self.proxy_geometry = trimesh.Trimesh(vertices=verts.detach().cpu(), faces=faces.int().detach().cpu(),  vertex_colors=rgb, process=False)

    def DDMC(self, verts=None, faces=None, field2cam=None, samples_dict=None, frame_id=None, inst_id=0, dir=None):
        '''
        Args:
            verts: mesh vertices in camera space. If is None, use DMC
            frameid: If is None, no warp.
        '''

        tet_verts = self.flexicubes.verts.unsqueeze(0)
        tet_indices = self.flexicubes.indices

        if dir is None:
            dir = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0)
            dir = dir.repeat(1, verts.shape[0], 1, 1)

        if verts is None and faces is None:
            # 3种路线：
            # 1 tet_verts代表规范空间，则先查询SDF，走DMC再warp到相机空间后渲染。拓扑不变但可能面片错位
            # 2 或者tet_verts代表相机空间，则warp回规范空间查询sdf，再DMC拿到网格的点，网格的点warp回规范空间查询颜色，然后渲染。拓扑会变，但面片位置是好的
            # 3 还有一种tet_verts代表规范空间，先查询SDF，将tet_verts坐标变到相机空间，再DMC，再将网格点变回规范空间查询颜色，然后渲染（一凯的做法）
            # 先试试第2种
            if frame_id is not None:
                assert frame_id.ndim == 1 
                backwarp_dict = self.backward_warp(tet_verts, dir, field2cam, frame_id, inst_id, samples_dict)
                tet_verts_canonical = backwarp_dict['xyz']
                dir_canonical = backwarp_dict['dir']
                xyz_embed = self.pos_embedding(tet_verts_canonical)
            else:
                xyz_embed = self.pos_embedding(tet_verts)
            

            xyz_feat = self.basefield(xyz_embed, None)
            sdf = self.sdf(xyz_feat)
            # dec_verts = self.decoder(triplane_feature2, tet_verts)
            deform = self.deformMlp(xyz_feat)
            weight = self.weightMlp(xyz_feat) * 0.1

            # if self.spob:
            #     pred_sdf = pred_sdf + self.radius - torch.sqrt((tet_verts**2).sum(-1))

            _, verts_camera, faces_camera = self.renderer(None, sdf, deform, tet_verts, tet_indices, weight=weight)
        # self.verts = verts[0]
        # self.faces = faces[0]

        # calculate real verts' attributes
        dir_embed = self.dir_embedding(dir_canonical)
        if self.appr_channels > 0:
            appr_embed = self.appr_embedding.get_vals(frame_id)
            appr_embed = appr_embed[:, None, None].expand(
                dir_embed.shape[:-1] + (appr_embed.shape[-1],)
            )
            appr_embed = torch.cat([dir_embed, appr_embed], -1)
        else:
            appr_embed = dir_embed

        xyz_embed = self.pos_embedding_color(verts_camera)
        xyz_feat = xyz_feat + self.colorfield(xyz_embed, inst_id)
        rgb = self.rgb(torch.cat([xyz_feat, appr_embed], -1))
        rgb = rgb.sigmoid()
        # self.rgb = rgb

        # check check verts face rgb 
        import ipdb; ipdb.set_trace()
        return verts_camera, faces_camera, rgb
        # mesh.metadata['feature'] = feature
        # mesh.visual.vertex_colors = colors

    def export_mesh(self, out_dir):
        self.proxy_geometry.export(out_dir)

    def render_view(self, mesh_v, mesh_f, cam_mv, Kinv=None, H=None, W=None,):
        '''
        Function to render a generated mesh with nvdiffrast
        :param mesh_v: List of vertices for the mesh
        :param mesh_f: List of faces for the mesh
        :param cam_mv:  4x4 rotation matrix
        :return:
        '''
        return_value_list = []
        for i_mesh in range(len(mesh_v)):
            return_value = self.flexicubes.render_mesh(
                mesh_v[i_mesh],
                mesh_f[i_mesh].int(),
                cam_mv[i_mesh],
                resolution=self.img_resolution,
                hierarchical_mask=False,
                Kinv=Kinv, H=H, W=W,
            )
            return_value_list.append(return_value)

        return_keys = return_value_list[0].keys()
        return_value = dict()
        for k in return_keys:
            value = [v[k] for v in return_value_list]
            return_value[k] = value

        mask_list, hard_mask_list = torch.cat(return_value['mask'], dim=0), torch.cat(return_value['hard_mask'], dim=0)
        return mask_list, hard_mask_list, return_value


    def init_proxy(self, geom_path, init_scale):
        """Initialize proxy geometry as a sphere

        Args:
            geom_path (str): Unused
            init_scale (float): Unused
        """
        self.update_proxy()

    def get_init_sdf_fn(self):
        """Initialize signed distance function as a skeleton or sphere

        Returns:
            sdf_fn_torch (Function): Signed distance function
        """

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

        if "skel-" in self.fg_motion:
            return sdf_fn_torch_skel
        else:
            return sdf_fn_torch_sphere

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
        density_gauss = self.warp.get_gauss_density(pts)  # (N,1)
        with torch.no_grad():
            density = self.forward(pts, inst_id=None, get_density=True)
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
        Kinv = samples_dict["Kinv"]  # (M,3,3)
        field2cam = samples_dict["field2cam"]  # (M,SE(3))
        frame_id = samples_dict["frame_id"]  # (M,)
        inst_id = samples_dict["inst_id"]  # (M,)
        near_far = samples_dict["near_far"]  # (M,2)
        hxy = samples_dict["hxy"]  # (M,N,2)
        H = samples_dict["H"]  # (M,)r
        W = samples_dict["W"]  # (M,)

        batch_size = Kinv.shape[0]
        feat_dict, deltas, aux_dict = ({}, {}, {})
        gauss_field = self.compute_gauss_density(self._xyz, samples_dict)
        aux_dict.update(gauss_field)
        gauss_density = gauss_field["gauss_density"]

        xyz_cam, dir_cam, deltas, depth = sample_cam_rays(
            hxy, Kinv, near_far, perturb=False, depth=None
        )
        
        # Dynamic Deep Marching Cubes
        verts_camera, faces_camera, rgb = self.DDMC(verts=None, faces=None, field2cam=field2cam, samples_dict=samples_dict, frame_id=frame_id, inst_id=inst_id, dir=dir_cam)        
        # 渲染非对称相机，主要改变neuralrender里的camera的投影方式
        import ipdb; ipdb.set_trace()
        field2cam_mat = quaternion_translation_to_se3(field2cam[0], field2cam[1])
        cam_mv = field2cam_mat.inverse()
        mask_list, hard_mask_list, return_value = self.render_view(verts_camera, faces_camera, rgb, cam_mv, Kinv=Kinv, H=H, W=W)

        # calculate other outputs
        verts_cano = self.backward_warp(verts_camera, dir_cam, field2cam, frame_id, inst_id, samples_dict)["xyz"]
        if not "is_gen3d" in samples_dict.keys() and not "no_warp" in samples_dict.keys():
            # flow
            pointwise_flow = self.compute_flow(
                hxy,
                verts_cano,
                frame_id,
                inst_id,
                field2cam,
                Kinv,
                samples_dict,
                flow_thresh=None,
            )
        else:
            pointwise_flow = torch.ones_like(verts_cam[..., :2])
        flow_scale = torch.max(pointwise_flow.max(), torch.abs(pointwise_flow.min())) + 1e-6
        pointwise_flow_scaled = pointwise_flow / flow_scale


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


