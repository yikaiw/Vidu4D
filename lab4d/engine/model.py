# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
from collections import defaultdict
import re

from matplotlib.pyplot import isinteractive
import numpy as np
import torch, pickle
import torch.nn as nn
from tqdm import tqdm
import random
from lab4d.engine.train_utils import get_local_rank
from lab4d.nnutils.intrinsics import IntrinsicsMLP
from lab4d.nnutils.multifields import MultiFields
from lab4d.nnutils.util import arap_loss
from lab4d.utils.render_utils import sample_cam_rays
from lab4d.utils.geom_utils import K2inv, K2mat, Kmatinv
from lab4d.utils.numpy_utils import interp_wt
from lab4d.utils.render_utils import render_pixel
from gs.utils.loss_utils import l1_loss, ssim
from torchvision.utils import save_image
from lab4d.utils.quat_transform import (
    dual_quaternion_to_quaternion_translation,quaternion_to_matrix
)
from gs.scene.cameras import Camera
from torchvision.utils import save_image
from preprocess.scripts import depth
import cv2

class dvr_model(nn.Module):
    """A model that contains a collection of static/deformable neural fields

    Args:
        config (Dict): Command-line args
        data_info (Dict): Dataset metadata from get_data_info()
    """

    def __init__(self, config, data_info):
        super().__init__()
        self.config = config
        self.device = get_local_rank()
        self.data_info = data_info

        self.fields = MultiFields(
            data_info=data_info,
            field_type=config["field_type"],
            fg_motion=config["fg_motion"],
            num_inst=1
            if config["single_inst"]
            else len(data_info["frame_info"]["frame_offset"]) - 1,
            opts=config,
        )
        self.intrinsics = IntrinsicsMLP(
            self.data_info["intrinsics"],
            frame_info=self.data_info["frame_info"],
            num_freq_t=0,
        )

        self.current_steps = 0
        self.ref_cams = []

    def mlp_init(self):
        """Initialize camera transforms, geometry, articulations, and camera
        intrinsics for all neural fields from external priors
        """
        self.fields.mlp_init()
        self.intrinsics.mlp_init()

    def forward(self, batch):
        """Run forward pass and compute losses

        Args:
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2),
                "hxy" (M,2,N,3), and "is_detected" (M,2)
        Returns:
            loss_dict (Dict): Computed losses. Keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1),
                "reg_gauss_mask" (M,N,1), "reg_visibility" (0,),
                "reg_eikonal" (0,), "reg_deform_cyc" (0,),
                "reg_soft_deform" (0,), "reg_gauss_skin" (0,),
                "reg_cam_prior" (0,), and "reg_skel_prior" (0,).
        """
        config = self.config
        self.process_frameid(batch)
        self.reshape_batch(batch)
        results = self.render(batch, flow_thresh=config["train_res"])
        loss_dict = self.compute_loss(batch, results)
        self.cached_results = results
        return loss_dict

    def process_frameid(self, batch):
        """Convert frameid within each video to overall frame id

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16), and
                "hxy" (M,N,3). This function modifies it in place to add key
                "frameid" (M,)
        """
        if not hasattr(self, "offset_cuda"):
            self.offset_cache = torch.tensor(
                self.data_info["frame_info"]["frame_offset_raw"],
                device=self.device,
                dtype=torch.long,
            )
        # convert frameid_sub to frameid
        batch["frameid"] = batch["frameid_sub"] + self.offset_cache[batch["dataid"]]

    def set_progress(self, current_steps):
        """Adjust loss weights and other constants throughout training

        Args:
            current_steps (int): Number of optimization steps so far
        """

        self.current_steps = current_steps

        # positional encoding annealing
        anchor_x = (0, 4000)
        anchor_y = (0.6, 1)
        type = "linear"
        alpha = interp_wt(anchor_x, anchor_y, current_steps, type=type)
        if alpha >= 1:
            alpha = None
        self.fields.set_alpha(alpha)

        # beta_prob: steps(0->2k, 1->0.2), range (0.2,1)
        anchor_x = (0, 2000)
        anchor_y = (1.0, 0.2)
        type = "linear"
        beta_prob = interp_wt(anchor_x, anchor_y, current_steps, type=type)
        self.fields.set_beta_prob(beta_prob)

        # camera prior wt: steps(0->800, 1->0), range (0,1)
        loss_name = "reg_cam_prior_wt"
        if self.config["reg_cam_prior_wt"] > 1:
            anchor_x = (0, 4000)
            anchor_y = (1, 0.1)
        else:
            anchor_x = (0, 800)
            anchor_y = (1, 0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # reg_eikonal_wt: steps(0->24000, 1->100), range (1,100)
        loss_name = "reg_eikonal_wt"
        anchor_x = (0, 4000)
        anchor_y = (1, 100)
        type = "log"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # skel prior wt: steps(0->4000, 1->0), range (0,1)
        loss_name = "reg_skel_prior_wt"
        anchor_x = (0, 4000)
        anchor_y = (1, 0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # gauss mask wt: steps(0->4000, 1->0), range (0,1)
        loss_name = "reg_gauss_mask_wt"
        anchor_x = (0, 4000)
        anchor_y = (1, 0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

    def set_loss_weight(
        self, loss_name, anchor_x, anchor_y, current_steps, type="linear"
    ):
        """Set a loss weight according to the current training step

        Args:
            loss_name (str): Name of loss weight to set
            anchor_x: Tuple of optimization steps [x0, x1]
            anchor_y: Tuple of loss values [y0, y1]
            current_steps (int): Current optimization step
            type (str): Interpolation type ("linear" or "log")
        """
        if "%s_init" % loss_name not in self.config.keys():
            self.config["%s_init" % loss_name] = self.config[loss_name]
        factor = interp_wt(anchor_x, anchor_y, current_steps, type=type)
        self.config[loss_name] = self.config["%s_init" % loss_name] * factor

    @torch.no_grad()
    def evaluate(self, batch, is_pair=True, nowarp=False):
        """Evaluate a Lab4D model

        Args:
            batch (Dict): Dataset metadata from `construct_eval_batch()`. Keys:
                "dataid" (M,), "frameid_sub" (M,), "crop2raw" (M,4),
                "feature" (M,N,16), and "hxy" (M,N,3)
            is_pair (bool): Whether to evaluate by rendering pairs
        Returns:
            rendered (Dict): Dict of rendered outputs. Keys: "mask" (M,H,W,1),
                "vis" (M,H,W,1), "depth" (M,H,W,1), "flow" (M,H,W,2),
                "feature" (M,H,W,16), "normal" (M,H,W,3), and
                "eikonal" (M,H,W,1)
        """
        if is_pair:
            div_factor = 2
        else:
            div_factor = 1
        self.process_frameid(batch)

        if "render_ref" in batch.keys():
            render_ref = True
            del batch["render_ref"]
        else:
            render_ref = False

        rendered = defaultdict(list)
        # split batch
        if "gs" in self.config["fg_motion"] and False:
            rendered = self.render(batch)["rendered"]
            for k, v in rendered.items():
                # 每两个frame取一个
                rendered[k] = v[::div_factor]

        else:
            for i in tqdm(range(0, len(batch["frameid"]) // div_factor)):
                batch_sub = {}
                for k, v in batch.items():
                    if isinstance(v, dict):
                        batch_sub[k] = {}
                        for k2, v2 in v.items():
                            batch_sub[k][k2] = v2[i * div_factor : (i + 1) * div_factor]
                    else:
                        batch_sub[k] = v[i * div_factor : (i + 1) * div_factor]
                rendered_sub = self.render(batch_sub, nowarp=nowarp)["rendered"]
                for k, v in rendered_sub.items():
                    if len(v.shape) == 4:
                        # print(k, v.shape)
                        rendered[k].append(v[0])
                    else:
                        res = int(np.sqrt(v.shape[1]))
                        rendered[k].append(v.view(div_factor, res, res, -1)[0])
            del rendered_sub
            torch.cuda.empty_cache()
            for k, v in rendered.items():
                rendered[k] = torch.stack(v, 0)
                
            # blend with mask: render = render * mask + 0*(1-mask)
            for k, v in rendered.items():
                if "mask" in k or "feature" in k or "xyz" in k :
                    continue
                else:
                    if "neus" in k:
                        rendered[k] = rendered[k] * rendered["mask_neus"]
                    else:
                        rendered[k] = rendered[k] * rendered["mask"]
        
        if self.config["gs_learnable_bg"] and "gs" in self.config["fg_motion"]:
            bkgd_rgb = self.fields.field_params['fg'].learnable_bkgd[None,None,None,:]#.repeat(rendered['rendered'].shape[0], rendered['rendered'].shape[1], rendered['rendered'].shape[2], 1)
            rendered['rendered'] = rendered['rendered'] + (1 - rendered["mask"].repeat(1,1,1,3)) * bkgd_rgb

        if render_ref:
            with open("%s/%s-%s/%s" % (self.config["logroot"], self.config['seqname'],self.config['logname'], f'{"Kcams.pkl"}'), 'wb') as f:
                pickle.dump(self.ref_cams, f)
            print("saving Kcams.pkl, please ensure it from rendering ref views")
        
        # import trimesh
        # from lab4d.utils.vis_utils import draw_cams
        # from lab4d.utils.quat_transform import quaternion_translation_to_se3
        # import numpy as np
        # cams = []
        # for cam in self.ref_cams:
        #     R = cam.R.detach().cpu().numpy()
        #     T = cam.T.detach().cpu().numpy()
        #     cam_rot_inv = R.T
        #     cam_tran_inv = -R.dot(T)
        #     rtmat = np.zeros((4,4))
        #     rtmat[:3,:3] = cam_rot_inv
        #     rtmat[:3,3] = cam_tran_inv
        #     rtmat[3,3] = 1
        #     # rtmat = np.invert(rtmat)
        #     cams.append(rtmat)

        # draw_cams(cams).export('/mnt/mfs/xinzhou.wang/repo/lab4d-gs-2d-final/cams_inv.ply')
        # pts = self.fields.field_params["fg"]._xyz.detach().cpu().numpy()
        # trimesh.points.PointCloud(pts).export('/mnt/mfs/xinzhou.wang/repo/lab4d-gs-2d-final/pts.ply')

        return rendered

    def update_geometry_aux(self):
        """Extract proxy geometry for all neural fields"""
        self.fields.update_geometry_aux()

    def export_geometry_aux(self, path):
        """Export proxy geometry for all neural fields"""
        return self.fields.export_geometry_aux(path)

    def render(self, batch, flow_thresh=None, nowarp=False):
        """Render model outputs

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16),
                "hxy" (M,N,3), and "frameid" (M,)
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            results: Rendered outputs. Keys: "rendered", "aux_dict"
            results["rendered"]: "mask" (M,N,1), "rgb" (M,N,3),
                "vis" (M,N,1), "depth" (M,N,1), "flow" (M,N,2),
                "feature" (M,N,16), "normal" (M,N,3), and "eikonal" (M,N,1)
            results["aux_dict"]["fg"]: "xy_reproj" (M,N,2) and "feature" (M,N,16)
        """
        samples_dict = self.get_samples(batch)
        if samples_dict['fg']['Kinv'].shape[0] == 1 and not self.training and "gs" in self.config["fg_motion"]: # evaluating, cache Kcams for render2d meshing
            q,T = samples_dict['fg']['field2cam']
            R = quaternion_to_matrix(q)[0]
            R = R.T
            FoVx = 2*torch.arctan((samples_dict['fg']['Kinv'][0,0,0] * samples_dict['fg']["W"])/2)
            FoVy = 2*torch.arctan((samples_dict['fg']['Kinv'][0,1,1] * samples_dict['fg']["H"])/2)
            idx = len(self.ref_cams)
            assert R.shape == (3, 3)
            test_cam = Camera(idx, R, T, FoVx, FoVy, torch.zeros(1,samples_dict['fg']["H"], samples_dict['fg']["W"]), None, str(idx), idx)
            self.ref_cams.append(test_cam)
        if nowarp:
            samples_dict['fg']['no_warp'] = True
        # import ipdb; ipdb.set_trace()
        if "gs" in self.config["fg_motion"]:
            if self.config["two_branch"]:
                # render gs
                results = self.render_samples({"fg":samples_dict["fg"]}, flow_thresh=flow_thresh)


                # # adjust resolution of intrinsics
                # samples_dict["fgneus"]["Kinv"][:,0,0] *= reso_mult
                # samples_dict["fgneus"]["Kinv"][:,1,1] *= reso_mult

                # use gs depth to guide neus sampling
                if self.training and self.current_steps > self.config["optim_warp_neus_iters"]:
                    # change resolution
                    reso_mult = samples_dict["fg"]["hxy"].shape[1] // self.config["neus_branch_reso"]
                    assert self.config["neus_branch_reso"] * reso_mult == samples_dict["fg"]["hxy"].shape[1]

                    # change size and shape

                    batch_size = samples_dict["fgneus"]["hxy"].shape[0]
                    samples_dict["fgneus"]["hxy"] = samples_dict["fgneus"]["hxy"][:, ::reso_mult, ::reso_mult, :].reshape(batch_size, -1, 3)
                    samples_dict["fgneus"]["feature"] = samples_dict["fgneus"]["feature"][:, ::reso_mult, ::reso_mult, :].reshape(batch_size, -1, 16)
                    resized_depth = results["rendered"]["depth"].clone().detach()
                    resized_depth = resized_depth[:, ::reso_mult, ::reso_mult, :].reshape(batch_size, -1, 1, 1)
                    # 只采样gs depth提供深度处的那一个点
                    xyz_cam, dir_cam, deltas, depth = sample_cam_rays(samples_dict["fgneus"]["hxy"], samples_dict["fgneus"]["Kinv"], samples_dict["fgneus"]["near_far"], perturb=False, depth=resized_depth)

                    with torch.no_grad():
                        # 把xyz_cam转到canonical并计算sdf
                        xyz_cano = self.fields.field_params["fgneus"].backward_warp(xyz_cam, dir_cam, samples_dict["fgneus"]["field2cam"], samples_dict["fgneus"]["frame_id"], samples_dict["fgneus"]["inst_id"],samples_dict["fgneus"])["xyz"]
                        sample_pts_sdf = self.fields.field_params["fgneus"].forward(xyz_cano, dir=None, frame_id=samples_dict["fgneus"]["frame_id"], inst_id=samples_dict["fgneus"]["inst_id"], get_density=False)
                        sample_pts_sdf = sample_pts_sdf.abs()

                        print("sample_pts_sdf mean is: ", sample_pts_sdf.mean())
                        save_image(sample_pts_sdf[0].reshape(1,1,128,128)*3, f"tmp/sdf.jpg")
                        save_image(resized_depth[0].reshape(1,1,128,128)*3, f"tmp/depth.jpg")
                        save_image(results["rendered"]["rendered"][0:1].permute(0,3,1,2), f"tmp/rgb.jpg")
                        import trimesh
                        trimesh.Trimesh(vertices=xyz_cam[0,:,0].detach().cpu().numpy()).export("/mnt/mfs/xinzhou.wang/repo/lab4d-gs/tmp/xyz_cam.obj")
                        trimesh.Trimesh(vertices=xyz_cano[0,:,0].detach().cpu().numpy()).export("/mnt/mfs/xinzhou.wang/repo/lab4d-gs/tmp/xyz_cano.obj")


                        dgs_k = self.config["dgs_k"]
                        guided_depth = resized_depth.repeat(1, 1, 2 * dgs_k + 1, 1)

                        for i in range(dgs_k * 2 + 1):
                            guided_depth[:, :, i] += (i - dgs_k) * sample_pts_sdf[:, :, 0]
                        
                        out_of_range = torch.logical_or(guided_depth.max(axis=2)[0] >= 2, guided_depth.min(axis=2)[0] <= 0)[..., 0]

                        near_far = samples_dict["fgneus"]["near_far"]
                        z_steps = torch.linspace(0, 1, guided_depth.shape[2], device=guided_depth.device)[None]  # (1, D)
                        depth_uniform = near_far[:, 0:1] * (1 - z_steps) + near_far[:, 1:2] * z_steps  # (M, D)
                        guided_depth[0][out_of_range[0]] = depth_uniform[0,:,None]
                        guided_depth[1][out_of_range[1]] = depth_uniform[1,:,None]
                        assert guided_depth.shape[0] == 2

                        samples_dict["fgneus"]["guided_depth"] = guided_depth.detach()

                    # TODO 直接不去算gs mask之外的点了
                    with torch.no_grad():
                        pass

                        results_neus = self.render_samples_chunk({"fgneus":samples_dict["fgneus"]}, flow_thresh=flow_thresh)
                        for k, v in results_neus["rendered"].items():
                            results["rendered"][k+"_neus"] = v.reshape(batch_size, self.config["neus_branch_reso"], self.config["neus_branch_reso"], -1)

                    # results_neus["aux_dict"]["fgneus"].keys()
                    # dict_keys(['mask', 'rgb', 'xyz', 'xyz_cam', 'depth', 'mask_fgneus', 'eikonal', 'vis', 'gauss_mask'])
                    # results_neus["rendered"].keys()
                    # dict_keys(['mask', 'xyz_cam', 'rgb', 'xyz', 'depth', 'mask_fgneus', 'eikonal', 'vis', 'gauss_mask'])

                # results["aux_dict"].update(results_neus["aux_dict"])
            else:
                results = self.render_samples(samples_dict, flow_thresh=flow_thresh)
        else:
            results = self.render_samples_chunk(samples_dict, flow_thresh=flow_thresh)
        return results
    
    def get_samples(self, batch):
        """Compute time-dependent camera and articulation parameters for all
        neural fields.

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16),
                "hxy" (M,N,3), and "frameid" (M,)
        Returns:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,4,4), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,3), and
                "feature" (M,N,16).
        """
        # import ipdb; ipdb.set_trace()
        if "Kinv" in batch.keys():
            Kinv = batch["Kinv"]
        else:
            Kmat = self.intrinsics.get_vals(batch["frameid"])
            if self.config["force_center_cam"]:
                Kmat[:, 2] = self.data_info["intrinsics"][0, 2]
                Kmat[:, 3] = self.data_info["intrinsics"][0, 3]
                # import ipdb; ipdb.set_trace()
                assert batch["crop2raw"][0,2] == 0
                assert batch["crop2raw"][0,3] == 0
            Kinv = K2inv(Kmat) @ K2mat(batch["crop2raw"])

        # Kinv = Kinv.detach()
        samples_dict = self.fields.get_samples(Kinv, batch)
        return samples_dict

    def render_samples_chunk(self, samples_dict, flow_thresh=None, chunk_size=8192*1):
        """Render outputs from all neural fields. Divide in chunks along pixel
        dimension N to avoid running out of memory.

        Args:
            samples_dict (Dict): Maps neural field types ("bg" or "fg") to
                dicts of input metadata and time-dependent outputs.
                Each dict has keys: "Kinv" (M,3,3), "field2cam" (M,4,4),
                "frame_id" (M,), "inst_id" (M,), "near_far" (M,2),
                "hxy" (M,N,3), and "feature" (M,N,16).
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
            chunk_size (int): Number of pixels to render per chunk
        Returns:
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        """
        # get chunk size
        category = list(samples_dict.keys())[0]
        total_pixels = (
            samples_dict[category]["hxy"].shape[0]
            * samples_dict[category]["hxy"].shape[1]
        )
        num_chunks = int(np.ceil(total_pixels / chunk_size))
        chunk_size_n = int(
            np.ceil(chunk_size // samples_dict[category]["hxy"].shape[0])
        )  # at n dimension

        results = {
            "rendered": defaultdict(list),
            "aux_dict": defaultdict(defaultdict),
        }
        for i in range(num_chunks):
            # construct chunk input
            samples_dict_chunk = defaultdict(list)
            for category, category_v in samples_dict.items():
                samples_dict_chunk[category] = defaultdict(list)
                for k, v in category_v.items():
                    if k == "hxy" or k == "guided_depth":
                        samples_dict_chunk[category][k] = v[
                            :, i * chunk_size_n : (i + 1) * chunk_size_n
                        ]
                    else:
                        samples_dict_chunk[category][k] = v

            # get chunk output
            results_chunk = self.render_samples(
                samples_dict_chunk, flow_thresh=flow_thresh
            )

            # merge chunk output
            for k, v in results_chunk["rendered"].items():
                if k not in results["rendered"].keys():
                    results["rendered"][k] = []
                results["rendered"][k].append(v)

            for cate in results_chunk["aux_dict"].keys():
                for k, v in results_chunk["aux_dict"][cate].items():
                    if k not in results["aux_dict"][cate].keys():
                        results["aux_dict"][cate][k] = []
                    results["aux_dict"][cate][k].append(v)
        # concat chunk output
        for k, v in results["rendered"].items():
            results["rendered"][k] = torch.cat(v, 1)

        for cate in results["aux_dict"].keys():
            for k, v in results["aux_dict"][cate].items():
                results["aux_dict"][cate][k] = torch.cat(v, 1)
        return results

    def render_samples(self, samples_dict, flow_thresh=None):
        """Render outputs from all neural fields.

        Args:
            samples_dict (Dict): Maps neural field types ("bg" or "fg") to
                dicts of input metadata and time-dependent outputs.
                Each dict has keys: "Kinv" (M,3,3), "field2cam" (M,4,4),
                "frame_id" (M,), "inst_id" (M,), "near_far" (M,2),
                "hxy" (M,N,3), and "feature" (M,N,16).
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        """
        multifields_dict, deltas_dict, aux_dict = self.fields.query_multifields(
            samples_dict, flow_thresh=flow_thresh
        )

        if 'gs' in self.config["fg_motion"] and "fgneus" not in multifields_dict.keys():
            rendered = multifields_dict['fg']
        else:
            field_dict, deltas = self.fields.compose_fields(multifields_dict, deltas_dict)
            rendered = render_pixel(field_dict, deltas)

            for cate in multifields_dict.keys():
                # render each field and put into aux_dict
                rendered_cate = render_pixel(multifields_dict[cate], deltas_dict[cate])
                for k, v in rendered_cate.items():
                    aux_dict[cate][k] = v

        if "fg" in aux_dict.keys():
            # move for visualization
            if "xyz_matches" in aux_dict["fg"].keys():
                rendered["xyz_matches"] = aux_dict["fg"]["xyz_matches"]
                rendered["xyz_reproj"] = aux_dict["fg"]["xyz_reproj"]

        results = {"rendered": rendered, "aux_dict": aux_dict}
        return results

    @staticmethod
    def reshape_batch(batch):
        """Reshape a batch to merge the pair dimension into the batch dimension

        Args:
            batch (Dict): Arbitrary dataloader outputs (M, 2, ...). This is
                modified in place to reshape each value to (M*2, ...)
        """
        for k, v in batch.items():
            batch[k] = v.view(-1, *v.shape[2:])

    def compute_loss(self, batch, results):
        """Compute model losses

        Args:
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2), and
                "hxy" (M,2,N,3)
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        Returns:
            loss_dict (Dict): Computed losses. Keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1),
                "reg_gauss_mask" (M,N,1), "reg_visibility" (0,),
                "reg_eikonal" (0,), "reg_deform_cyc" (0,),
                "reg_soft_deform" (0,), "reg_gauss_skin" (0,),
                "reg_cam_prior" (0,), and "reg_skel_prior" (0,).
        """
        config = self.config
        loss_dict = {}
        self.compute_recon_loss(loss_dict, results, batch, config)
        self.mask_losses(loss_dict, batch, config)
        
        # save_image((batch['mask']*1.0).permute(0,3,1,2), "tmp/mask_gt.png")
        # save_image(((batch['vis2d']>0)*1.0).permute(0,3,1,2), "tmp/vis2d_batch.png")
        # save_image(loss_dict['mask'].permute(0,3,1,2), "tmp/mask_loss.png")
        # save_image(results['rendered']['mask'].permute(0,3,1,2), "tmp/mask_rendered.png")
        # save_image(loss_dict['rgb'].permute(0,3,1,2), "tmp/rgb_loss.png")
        # save_image(results['rendered']['rendered'].permute(0,3,1,2), "tmp/rgb_rendered.png")
        # save_image(((loss_dict['vis']>0)*1.0).permute(0,3,1,2), "tmp/vis_rendered.png")
        #### init loss
        # self.init_loss(loss_dict)
        self.compute_reg_loss(loss_dict, results)
        self.apply_loss_weights(loss_dict, config)
        return loss_dict

    @staticmethod
    def get_mask_balance_wt(mask, vis2d, is_detected):
        """Balance contribution of positive and negative pixels in mask.

        Args:
            mask: (M,N,1) Object segmentation mask
            vis2d: (M,N,1) Whether each pixel is visible in the video frame
            is_detected: (M,) Whether there is segmentation mask in the frame
        Returns:
            mask_balance_wt: (M,N,1) Balanced mask
        """
        # import ipdb; ipdb.set_trace()
        # all the positive labels
        mask = mask.float()
        # all the labels
        if len(vis2d.shape) == 3:
            vis2d = vis2d.float() * is_detected.float()[:, None, None]
        else:
            vis2d = vis2d.float() * is_detected.float()[:, None, None, None]
        if mask.sum() > 0 and (1 - mask).sum() > 0:
            pos_wt = vis2d.sum() / mask[vis2d > 0].sum()
            neg_wt = vis2d.sum() / (1 - mask[vis2d > 0]).sum()
            mask_balance_wt = 0.5 * pos_wt * mask + 0.5 * neg_wt * (1 - mask)
        else:
            mask_balance_wt = 1
        return mask_balance_wt

    # @staticmethod
    def compute_recon_loss(self, loss_dict, results, batch, config):
        """Compute reconstruction losses.

        Args:
            loss_dict (Dict): Updated in place to add keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1), and
                "reg_gauss_mask" (M,N,1)
            results: Rendered outputs. Keys: "rendered", "aux_dict"
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2), and
                "hxy" (M,2,N,3)
            config (Dict): Command-line options
        """
        rendered = results["rendered"]
        aux_dict = results["aux_dict"]
        # reconstruction loss
        # get rendered fg mask
        if config["field_type"] == "fg":
            rendered_fg_mask = rendered["mask"]
        elif config["field_type"] == "comp":
            rendered_fg_mask = rendered["mask_fg"]
        elif config["field_type"] == "bg":
            rendered_fg_mask = None
        else:
            raise ("field_type %s not supported" % config["field_type"])
        # get fg mask balance factor
        mask_balance_wt = dvr_model.get_mask_balance_wt(
            batch["mask"], batch["vis2d"], batch["is_detected"]
        )
        if config["field_type"] == "bg":
            loss_dict["mask"] = (rendered["mask"] - 1).pow(2)
        elif config["field_type"] == "fg":
            loss_dict["mask"] = (rendered_fg_mask - batch["mask"].float()).pow(2)
            loss_dict["mask"] *= mask_balance_wt
        elif config["field_type"] == "comp":
            loss_dict["mask"] = (rendered_fg_mask - batch["mask"].float()).pow(2)
            loss_dict["mask"] *= mask_balance_wt
            loss_dict["mask"] += (rendered["mask"] - 1).pow(2)
        else:
            raise ("field_type %s not supported" % config["field_type"])

        if config["field_type"] == "fg" or config["field_type"] == "comp" and "xy_reproj" in aux_dict["fg"].keys():
            # if "gs" in config["fg_motion"]:
            #     resized_feature = torch.nn.functional.interpolate(aux_dict["fg"]["feature"].permute(0,3,1,2), size=batch["feature"].shape[1:3], scale_factor=None, mode='nearest', align_corners=None).permute(0,2,3,1)
            # else: 
            #     resized_feature = 
            if "feature" in aux_dict["fg"].keys() and aux_dict["fg"]["feature"].shape[-1] > 0:
                # import ipdb; ipdb.set_trace()
                loss_dict["feature"] = (aux_dict["fg"]["feature"] - batch["feature"]).norm(
                    2, -1, keepdim=True
                )
            # import pdb;pdb.set_trace()
            if "xy_reproj" in aux_dict["fg"].keys():
                loss_dict["feat_reproj"] = (
                    aux_dict["fg"]["xy_reproj"] - batch["hxy"][..., :2]
                ).norm(2, -1, keepdim=True)
            
        if "gs" in config["fg_motion"]:
            vis2d = batch["vis2d"].float()
            maskfg = batch["mask"].float()
            mask = maskfg * vis2d

            # only compute loss on where vis2d is true
            # new L1
            vis2d_repeat = vis2d.repeat(1,1,1,3)
            rendered_vis = rendered["rendered"][vis2d_repeat > 0]
            batch_vis = batch["rgb"][vis2d_repeat > 0]
            crop_L1 = torch.abs((rendered_vis - batch_vis))
            L1 = torch.zeros_like(rendered["rendered"])
            L1[vis2d_repeat > 0] = crop_L1
            L1 = L1.mean()

            # old L1
            # L1 = torch.abs((rendered["rendered"] - batch["rgb"])).mean()

            loss_dict["rgb"] = (1.0 - config["lambda_dssim"]) * L1
            # loss_dict["rgb_ssim"] = config["lambda_dssim"] * (1.0 - ssim(rendered["rendered"].to(torch.half), (batch["rgb"] * mask).to(torch.half)))
        else:
            loss_dict["rgb"] = (rendered["rgb"] - batch["rgb"]).pow(2)

        try:
            loss_dict["depth"] = (
                (rendered["depth"] - batch["depth"]).norm(2, -1, keepdim=True).clone()
            )
            loss_dict["flow"] = (rendered["flow"] - batch["flow"]).norm(2, -1, keepdim=True)
            loss_dict["flow"] = loss_dict["flow"] * (batch["flow_uct"] > 0).float()
        except:
            pass

        # convert depth to point cloud
        # with torch.no_grad():
        #     import trimesh
        #     xyz_render = batch['hxy'][0] / batch['hxy'][0].max()
        #     xyz_render[...,2] = rendered["depth"][0].squeeze() / rendered["depth"][0].max()
        #     xyz_render = xyz_render.reshape(-1,3)
        #     fliter = xyz_render[:,2] > 1e-1
        #     xyz_render = xyz_render[fliter]
        #     trimesh.points.PointCloud(xyz_render.detach().cpu().numpy()).export("tmp/xyz_render.ply")

        #     xyz_batch = batch['hxy'][0] / batch['hxy'][0].max()
        #     xyz_batch[...,2] = batch["depth"][0].squeeze() / batch["depth"][0].max()
        #     xyz_batch = xyz_batch.reshape(-1,3)
        #     xyz_batch = xyz_batch[fliter]
        #     trimesh.points.PointCloud(xyz_batch.detach().cpu().numpy()).export("tmp/xyz_batch.ply")

        # visibility: supervise on fg and bg separately
        vis_loss = []
        # for aux_cate_dict in aux_dict.values():
        for cate, aux_cate_dict in aux_dict.items():
            if cate == "bg":
                # use smaller weight for bg
                aux_cate_dict["vis"] *= 0.01
            if "vis" in aux_cate_dict.keys():
                vis_loss.append(aux_cate_dict["vis"])

        if "vis_neus" in rendered.keys():
            loss_dict["vis_neus"] = rendered["vis_neus"]

        if len(vis_loss) > 0:
            vis_loss = torch.stack(vis_loss, 0).sum(0)
        else:
            vis_loss = torch.zeros_like(loss_dict["rgb"])
        loss_dict["vis"] = vis_loss

        # consistency between rendered mask and gauss mask
        if "gauss_mask" in rendered.keys():
            loss_dict["reg_gauss_mask"] = (
                aux_dict["fg"]["gauss_mask"] - rendered_fg_mask.detach()
            ).pow(2)

        # for neus branch loss
        if self.config["two_branch"] and "mask_neus" in rendered.keys():
            reso_mult = batch["hxy"].shape[1] // self.config["neus_branch_reso"]

            batch["mask_neus"] = batch["mask"][:, ::reso_mult, ::reso_mult, :]
            mask_balance_wt = mask_balance_wt[:, ::reso_mult, ::reso_mult, :]
            loss_dict["mask_neus"] = (rendered["mask_neus"] - batch["mask_neus"].float()).pow(2) * mask_balance_wt

            batch["rgb_neus"] = batch["rgb"][:, ::reso_mult, ::reso_mult, :]
            # loss_dict["rgb"] = (rendered["rgb_neus"] - batch["rgb_neus"]).pow(2)
            
            mask = maskfg * vis2d
            mask = mask[:, ::reso_mult, ::reso_mult, :]
            Ll1 = l1_loss(rendered["rgb_neus"], batch["rgb_neus"])
            loss_dict["rgb_neus"] = (1.0 - config["lambda_dssim"]) * Ll1
            loss_dict["rgb_ssim_neus"] = config["lambda_dssim"] * (1.0 - ssim(rendered["rgb_neus"].to(torch.half), (batch["rgb_neus"] * mask).to(torch.half)))
            loss_dict["eikonal_neus"] = rendered["eikonal_neus"]

            mask_neus_upscale = torch.nn.functional.interpolate(rendered["mask_neus"].permute(0,3,1,2), size=rendered["mask"].shape[1:3], mode='bilinear', align_corners=False).permute(0,2,3,1)
            depth_neus_upscale = torch.nn.functional.interpolate(rendered["depth_neus"].permute(0,3,1,2), size=rendered["depth"].shape[1:3], mode='bilinear', align_corners=False).permute(0,2,3,1)
            normal_neus_upscale = torch.nn.functional.interpolate(rendered["normal_neus"].permute(0,3,1,2), size=rendered["normal"].shape[1:3], mode='bilinear', align_corners=False).permute(0,2,3,1)

            depth_neus_upscale = depth_neus_upscale * mask_neus_upscale
            normal_neus_upscale = normal_neus_upscale * mask_neus_upscale

            loss_dict["mutual_depth"] = 0.5 * (depth_neus_upscale - rendered["depth"]).pow(2) * mask_neus_upscale
            loss_dict["mutual_normal"] = 0.01 * (1 - torch.nn.functional.cosine_similarity(normal_neus_upscale, rendered["normal"], dim=-1)[...,None]) * mask_neus_upscale
            loss_dict["mutual_mask"] = (mask_neus_upscale - rendered["mask"]).pow(2)

            if True:
                save_image(depth_neus_upscale[0:1].permute(0,3,1,2), f"tmp/mutual_depth_neus.jpg")
                save_image(normal_neus_upscale[0:1].permute(0,3,1,2), f"tmp/mutual_normal_neus.jpg")
                save_image(mask_neus_upscale[0:1].permute(0,3,1,2), f"tmp/mutual_mask_neus.jpg")
                           
                save_image(rendered["depth"].permute(0,3,1,2), f"tmp/mutual_depth.jpg")
                save_image(rendered["normal"].permute(0,3,1,2), f"tmp/mutual_normal.jpg")
                save_image(rendered["mask"].permute(0,3,1,2), f"tmp/mutual_mask.jpg")

                save_image(loss_dict["mutual_depth"][0:1].permute(0,3,1,2)*2, f"tmp/loss_mutual_depth.jpg")
                save_image(loss_dict["mutual_normal"][0:1].permute(0,3,1,2)*100, f"tmp/loss_mutual_normal.jpg")
                save_image(loss_dict["mutual_mask"][0:1].permute(0,3,1,2)*1, f"tmp/loss_mutual_mask.jpg")

                # from lab4d.utils.quat_transform import quaternion_translation_to_se3
                # from lab4d.utils.vis_utils import draw_cams, mesh_cat
                
                # rtmat = quaternion_translation_to_se3(batch["field2cam"]["fg"][...,:4], batch["field2cam"]["fg"][...,4:]).cpu()
                # # evenly pick max 200 cameras
                # mesh_cam = draw_cams(rtmat)
                # if hasattr(self, "mesh"):
                #     self.cam_mesh = mesh_cat(self.cam_mesh, mesh_cam)
                # else:
                #     self.cam_mesh = mesh_cam
                # mesh_geo = mesh_cat(self.fields.field_params['fgneus'].proxy_geometry, self.fields.field_params['fg'].proxy_geometry)
                # mesh_merge = mesh_cat(mesh_geo,self.cam_mesh)
                # mesh_merge.export("tmp/cam_geo.obj")

    def compute_reg_loss(self, loss_dict, results):
        """Compute regularization losses.

        Args:
            loss_dict (Dict): Updated in place to add keys:
                "reg_visibility" (0,), "reg_eikonal" (0,),
                "reg_deform_cyc" (0,), "reg_soft_deform" (0,),
                "reg_gauss_skin" (0,), "reg_cam_prior" (0,), and
                "reg_skel_prior" (0,).
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        """
        
        rendered = results["rendered"]
        
        lambda_dist = self.config["lambda_dist"]
        lambda_normal = self.config["lambda_normal"]
        
        lambda_normal = lambda_normal if self.current_steps > 8000 else 0.0
        lambda_dist = lambda_dist if self.current_steps > 8000 else 0.0
        if self.config["reg_in_cano"] :
            rend_dist = results["aux_dict"]['fg']["rend_dist_cano"]
            rend_normal  = results["aux_dict"]['fg']['rend_normal_cano']
            surf_normal = results["aux_dict"]['fg']['surf_normal_cano']
            # import ipdb; ipdb.set_trace()
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()
            loss_dict["normal_loss"] = normal_loss
            loss_dict["dist_loss"] = dist_loss
        
        elif "rend_dist" in rendered.keys():
            rend_dist = rendered["rend_dist"]
            rend_normal  = rendered['rend_normal']
            surf_normal = rendered['surf_normal']
            # import ipdb; ipdb.set_trace()
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()
            loss_dict["normal_loss"] = normal_loss
            loss_dict["dist_loss"] = dist_loss

        # rendered, rendered['fg']  ['depth'] ['normal']
        
        aux_dict = results["aux_dict"]
        # regularization loss
        loss_dict["reg_visibility"] = self.fields.visibility_decay_loss()
        loss_dict["reg_eikonal"] = rendered["eikonal"]
        if "fg" in aux_dict.keys():
            loss_dict["reg_deform_cyc"] = aux_dict["fg"]["cyc_dist"]
            loss_dict["reg_delta_skin"] = aux_dict["fg"]["delta_skin"]
            loss_dict["reg_skin_entropy"] = aux_dict["fg"]["skin_entropy"]
        loss_dict["reg_soft_deform"] = self.fields.soft_deform_loss()
        loss_dict["reg_gauss_skin"] = self.fields.gauss_skin_consistency_loss()
        loss_dict["reg_cam_prior"] = self.fields.cam_prior_loss()
        loss_dict["reg_skel_prior"] = self.fields.skel_prior_loss()

        if self.config["arap_wt"] > 0:
            # random int from 0 128
            loss_dict["arap"] = torch.tensor(0.0).cuda()
            for i in range(10):
                frame_map = self.data_info["frame_info"]['frame_mapping']
                delta = random.randint(1, 8)
                first_frame = random.randint(0, len(frame_map) - delta - 1)
                second_frame = first_frame + delta
                frame_id = torch.tensor([frame_map[first_frame], frame_map[second_frame]]).cuda()
                (t_articulation, rest_articulation) = self.fields.field_params['fg'].warp.articulation.get_vals_and_mean(frame_id)
                loss_dict["arap"] += arap_loss(dual_quaternion_to_quaternion_translation(t_articulation)[1])
        
        if "gs" in self.config["fg_motion"] and self.config["reg_volume_loss_wt"] > 0:
            scaling = self.fields.field_params['fg'].get_scaling
            loss_dict["reg_volume_loss"] = scaling.prod(dim=1).mean()

    def init_loss(self, loss_dict, nsample=10000):
        """Apply init loss

        Args:
            loss_dict (Dict): Dense losses. Keys: "mask" (M,N,1), "rgb" (M,N,3),
                "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1), "feature" (M,N,1),
                "feat_reproj" (M,N,1), and "reg_gauss_mask" (M,N,1). Modified in
                place to multiply loss_dict["mask"] with the other losses
        """
        # import pdb;pdb.set_trace()
        device = next(self.parameters()).device
        inst_id = torch.randint(0,self.fields.field_params['fg'].num_inst, (nsample * 2,), device=device)
        pts_mesh = torch.tensor(self.fields.field_params['fg'].init_geometry.sample(nsample, return_index=False), dtype=torch.float32, device=device)
        pts_rand = self.fields.field_params['fg'].sample_points_aabb(nsample, extend_factor=0.25)
        pts = torch.cat((pts_mesh,pts_rand),dim = 0)
        sdf_gt = self.fields.field_params['fg'].sdf_fn_torch(pts)
        sdf = self.fields.field_params['fg'].forward(pts, inst_id=inst_id, get_density=False)
        loss_dict["init_loss"] = (sdf - sdf_gt).pow(2).mean()

    # @staticmethod
    def mask_losses(self, loss_dict, batch, config):
        """Apply segmentation mask on dense losses

        Args:
            loss_dict (Dict): Dense losses. Keys: "mask" (M,N,1), "rgb" (M,N,3),
                "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1), "feature" (M,N,1),
                "feat_reproj" (M,N,1), and "reg_gauss_mask" (M,N,1). Modified in
                place to multiply loss_dict["mask"] with the other losses
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2), and
                "hxy" (M,2,N,3)
            config (Dict): Command-line options
        """
        # ignore the masking step
        keys_ignore_masking = ["reg_gauss_mask"]
        # always mask-out non-visible (out-of-frame) pixels
        keys_allpix = ["mask", "mutual_normal", "mutual_depth", "mutual_mask"]
        # always mask-out non-object pixels
        keys_fg = ["feature", "feat_reproj"]
        # field type specific keys
        keys_type_specific = ["rgb", "depth", "flow", "vis", "rgb_ssim"]
        
        if self.config["vis2d_dilate"]:
            import ipdb; ipdb.set_trace()
            kernel = torch.ones((1, 1, 3, 3)).to(batch["vis2d"].device)
            batch["vis2d"] = torch.nn.functional.max_pool2d(batch["vis2d"], kernel_size=3, stride=1, padding=1)

        # type-specific masking rules
        vis2d = batch["vis2d"].float()
        maskfg = batch["mask"].float()
        if config["field_type"] == "bg":
            mask = (1 - maskfg) * vis2d
        elif config["field_type"] == "fg":
            mask = maskfg * vis2d
        elif config["field_type"] == "comp":
            mask = vis2d
        else:
            raise ("field_type %s not supported" % config["field_type"])
        
        # no mask
        if self.config["no_loss_mask"]:
            mask = torch.ones_like(mask)
            maskfg = torch.ones_like(maskfg)
            vis2d = torch.ones_like(vis2d)
        # apply mask
        for k, v in loss_dict.items():
            if self.config["maskloss_no_vis2d"] and "mask" in k:
                new_vis2d = vis2d.clone()
                new_vis2d[vis2d == 0] = 0.1
                loss_dict[k] = v * new_vis2d
                continue
            if k in keys_ignore_masking:
                continue
            elif k in keys_allpix:
                loss_dict[k] = v * vis2d
            elif k in keys_fg:
                loss_dict[k] = v * maskfg
            elif k in keys_type_specific:
                loss_dict[k] = v * mask
            else:
                if "neus" not in k:
                    raise ("loss %s not defined" % k)

        # mask out the following losses if obj is not detected
        keys_mask_not_detected = ["mask", "feature", "feat_reproj"]
        for k, v in loss_dict.items():
            if k in keys_mask_not_detected:
                if len(v.shape) == 3:
                    loss_dict[k] = v * batch["is_detected"].float()[:, None, None]
                else:
                    loss_dict[k] = v * batch["is_detected"].float()[:, None, None, None]
                # loss_dict[k] = v * batch["is_detected"].float()[:, None, None]
        
        if "rgb_neus" in loss_dict.keys():
            reso_mult = batch["hxy"].shape[1] // self.config["neus_branch_reso"]
            mask_neus = mask[:, ::reso_mult, ::reso_mult, :]

            for k,v in loss_dict.items():
                if "neus" in k:
                    loss_dict[k] = v * mask_neus
            

    @staticmethod
    def apply_loss_weights(loss_dict, config):
        """Weigh each loss term according to command-line configs

        Args:
            loss_dict (Dict): Computed losses. Keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1),
                "reg_gauss_mask" (M,N,1), "reg_visibility" (0,),
                "reg_eikonal" (0,), "reg_deform_cyc" (0,),
                "reg_soft_deform" (0,), "reg_gauss_skin" (0,),
                "reg_cam_prior" (0,), and "reg_skel_prior" (0,). Modified in
                place to multiply each term with a scalar weight.
            config (Dict): Command-line options
        """
        px_unit_keys = ["flow", "feat_reproj"]
        for k, v in loss_dict.items():
            # average over non-zero pixels
            if k == "rgb_ssim":
                loss_dict[k] *= config["rgb_wt"]

            if (v > 0).sum() == 0:
                loss_dict[k] = v.mean()
            else:
                loss_dict[k] = v[v > 0].mean()

            # scale with image resolution
            if k in px_unit_keys:
                loss_dict[k] /= config["train_res"]

            # scale with loss weights
            wt_name = k + "_wt"
            if wt_name in config.keys():
                loss_dict[k] *= config[wt_name]

from lab4d.utils.quat_transform import matrix_to_quaternion


def se3_to_quaternion_translation(se3, tuple=True):
    q = matrix_to_quaternion(se3[..., :3, :3])
    t = se3[..., :3, 3]
    if tuple:
        return q, t
    else:
        return torch.cat((q, t), -1)