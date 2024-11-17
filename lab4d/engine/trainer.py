# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os, math
import time
from collections import defaultdict
from copy import deepcopy

from networkx import k_components
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import trimesh
from torch.utils.tensorboard import SummaryWriter
import PIL
cudnn.benchmark = True

from lab4d.dataloader import data_utils
from lab4d.dataloader.vidloader import VidDataset
from lab4d.engine.model import dvr_model
from lab4d.engine.train_utils import DataParallelPassthrough, get_local_rank
from lab4d.utils.profile_utils import torch_profile
from lab4d.utils.torch_utils import remove_ddp_prefix
from lab4d.utils.vis_utils import img2color, make_image_grid
import open3d as o3d
from torchvision.utils import save_image
import datetime

current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

class Trainer:
    def __init__(self, opts):
        """Train and evaluate a Lab4D model.

        Args:
            opts (Dict): Command-line args from absl (defined in lab4d/config.py)
        """
        # When profiling, use fewer iterations per round so trace files are smaller
        is_resumed = opts["load_path"] != ""
        if opts["profile"]:
            opts["iters_per_round"] = 10

        self.opts = opts
        self.define_dataset()
        self.trainer_init()
        self.define_model() # 初始化gs 读取ply
        self.optimizer_init(is_resumed=is_resumed) # 定义gs优化器

        if 'gs' in self.opts['fg_motion']:
            self.model.fields.field_params['fg'].optimizer = self.gs_optimizer

        # load model
        if is_resumed:
            self.load_checkpoint_train() # 读取所有ckpt有的参数，包括warp和相机和gs参数,但是如果load ply，就不读GS的了

    def trainer_init(self):
        """Initialize logger and other misc things""" 
        opts = self.opts

        logname = "%s-%s" % (opts["seqname"], opts["logname"])
        
        self.save_dir = os.path.join(opts["logroot"], logname)
        self.save_dir = os.path.join(os.getcwd(), self.save_dir)
        if get_local_rank() == 0:
            os.makedirs("tmp/", exist_ok=True)
            os.makedirs(self.save_dir, exist_ok=True)

            # tensorboard
            self.log = SummaryWriter(
                "%s/%s/%s" % (opts["logroot"], logname, f'{"log"}')
            )
        else:
            self.log = None

        self.current_steps = 0  # 0-total_steps
        self.current_round = 0  # 0-num_rounds

        # 0-last image in eval dataset
        self.eval_fid = np.linspace(0, len(self.evalloader) - 1, 9).astype(int)

        # torch.manual_seed(8)  # do it again
        # torch.cuda.manual_seed(1)

    def define_dataset(self):
        """Construct training and evaluation dataloaders."""
        opts = self.opts
        train_dict = self.construct_dataset_opts(opts)

        if "gs" in self.opts["fg_motion"]:
            train_dict["pixels_per_image"] = -1
        if self.opts["quant_exp"]:
            train_dict["quant_exp"] = True
            train_dict["delta_list"] = [4, 8]

        self.trainloader = data_utils.train_loader(train_dict)

        eval_dict = self.construct_dataset_opts(opts, is_eval=True)
        self.evalloader = data_utils.eval_loader(eval_dict)

        self.data_info, self.data_path_dict = data_utils.get_data_info(self.evalloader)

        self.total_steps = opts["num_rounds"] * min(
            opts["iters_per_round"], len(self.trainloader)
        )

    def init_model(self):
        """Initialize camera transforms, geometry, articulations, and camera
        intrinsics from external priors, if this is the first run"""
        opts = self.opts
        # init mlp
        if get_local_rank() == 0:
            self.model.mlp_init()

    def define_model(self):
        """Define a Lab4D model and wrap it with DistributedDataParallel"""
        opts = self.opts
        data_info = self.data_info

        self.device = torch.device("cuda:{}".format(get_local_rank()))
        self.model = dvr_model(opts, data_info)

        # ddp
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = self.model.to(self.device)

        self.init_model()
        try:
            self.model = DataParallelPassthrough(
                self.model,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
                find_unused_parameters=False,
            )
        except:
            pass
        # cache queue of length 2
        self.model_cache = [None, None]
        self.optimizer_cache = [None, None]
        self.scheduler_cache = [None, None]

    def get_lr_dict(self):
        """Return the learning rate for each category of trainable parameters

        Returns:
            param_lr_startwith (Dict(str, float)): Learning rate for base model
            param_lr_with (Dict(str, float)): Learning rate for explicit params
        """
        # define a dict for (tensor_name, learning) pair
        opts = self.opts
        lr_base = opts["learning_rate"]
        lr_explicit = lr_base * 10

        param_lr_startwith = {
            "module.fields.field_params": lr_base,
            "module.intrinsics": lr_base,
        }
        param_lr_with = {
            ".logibeta": lr_explicit,
            ".logsigma": lr_explicit,
            ".logscale": lr_explicit,
            ".log_gauss": lr_explicit,
            ".base_quat": lr_explicit,
            ".base_logfocal": lr_explicit,
            ".base_ppoint": lr_explicit,
            ".shift": lr_explicit,
        }
    
        # if "gs" in self.opts["fg_motion"]:
        #     param_lr_with["._xyz"] = opts["position_lr_init"] * self.model.fields.field_params['fg'].cameras_extent
        #     param_lr_with["._features_dc"] = opts["feature_lr"]
        #     param_lr_with["._features_rest"] = opts["feature_lr"] / 20.0
        #     param_lr_with["._opacity"] = opts["opacity_lr"]
        #     param_lr_with["._scaling"] = opts["scaling_lr"]
        #     param_lr_with["._rotation"] = opts["rotation_lr"]
        #     param_lr_with["._regist_feat"] = opts["regist_feat_lr"]

        return param_lr_startwith, param_lr_with

    def optimizer_init(self, is_resumed=False):
        """Set the learning rate for all trainable parameters and initialize
        the optimizer and scheduler.

        Args:
            is_resumed (bool): True if resuming from checkpoint
        """
        opts = self.opts

        param_lr_startwith, param_lr_with = self.get_lr_dict()

        if opts["freeze_bone_len"]:
            param_lr_with[".log_bone_len"] = 0

        param_lr_startwith["module.intrinsics"] *= opts["intrinsics_lr_mult"]
        param_lr_with['.base_logfocal'] *= opts["intrinsics_lr_mult"]
        param_lr_with['.base_ppoint'] *= opts["intrinsics_lr_mult"]


        params_list = []
        lr_list = []

        name_mapping = {"_xyz": "xyz", "_features_dc": "f_dc", "_features_rest": "f_rest", "_opacity": "opacity", "_scaling": "scaling", "_rotation": "rotation", "_regist_feat": "regist_feat"}
        fix_geo = ["fields.field_params.fg.logibeta","fields.field_params.fg.logsigma", "fields.field_params.fg.sdf.weight","fields.field_params.fg.sdf.bias",
                    "fields.field_params.fg.basefield.linear_1.0.weight", "fields.field_params.fg.basefield.linear_1.0.bias",
                    "fields.field_params.fg.basefield.linear_2.0.weight", "fields.field_params.fg.basefield.linear_2.0.bias", 
                    "fields.field_params.fg.basefield.linear_3.0.weight", "fields.field_params.fg.basefield.linear_3.0.bias",
                    "fields.field_params.fg.basefield.linear_4.0.weight", "fields.field_params.fg.basefield.linear_4.0.bias" ,
                    "fields.field_params.fg.basefield.linear_5.0.weight", "fields.field_params.fg.basefield.linear_5.0.bias" ,
                    "fields.field_params.fg.basefield.linear_6.0.weight", "fields.field_params.fg.basefield.linear_6.0.bias" ,
                    "fields.field_params.fg.basefield.linear_7.0.weight", "fields.field_params.fg.basefield.linear_7.0.bias" ,
                    "fields.field_params.fg.basefield.linear_8.0.weight", "fields.field_params.fg.basefield.linear_8.0.bias" ,
                    "fields.field_params.fg.basefield.linear_final.0.weight", "fields.field_params.fg.basefield.linear_final.0.bias",
                    "fields.field_params.fg.basefield.inst_embedding.mapping.weight"]
        for name, p in self.model.named_parameters():
            name_found = False
            for params_name, lr in param_lr_with.items():
                if params_name in name:
                    # import pdb;pdb.set_trace()
                    # if name.split(".")[-1] in name_mapping or name in fix_geo:
                    if name.split(".")[-1] in name_mapping:
                        continue
                        # name = name_mapping[name.split(".")[-1]]
                    params_list.append({"params": p, "name": name})
                    lr_list.append(lr)
                    name_found = True
                    if get_local_rank() == 0:
                        print(name, p.shape, lr)

            if name_found:
                continue
            for params_name, lr in param_lr_startwith.items():
                if name.startswith(params_name) or name.startswith(params_name[7:]):
                    # if name.split(".")[-1] in name_mapping or name in fix_geo:
                    if name.split(".")[-1] in name_mapping :

                        continue
                        name = name_mapping[name.split(".")[-1]]
                    params_list.append({"params": p, "name": name})
                    lr_list.append(lr)
                    if get_local_rank() == 0:
                        print(name, p.shape, lr)

        if "gs" in self.opts["fg_motion"]:

            self.spatial_lr_scale = 1
            l = [
                {'params': [self.model.fields.field_params["fg"]._xyz], 'lr': self.opts["position_lr_init"] * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self.model.fields.field_params["fg"]._features_dc], 'lr': self.opts["feature_lr"], "name": "f_dc"},
                {'params': [self.model.fields.field_params["fg"]._features_rest], 'lr': self.opts["feature_lr"] / 20.0, "name": "f_rest"},
                {'params': [self.model.fields.field_params["fg"]._opacity], 'lr': self.opts["opacity_lr"], "name": "opacity"},
                {'params': [self.model.fields.field_params["fg"]._scaling], 'lr': self.opts["scaling_lr"], "name": "scaling"},
                {'params': [self.model.fields.field_params["fg"]._rotation], 'lr': self.opts["rotation_lr"], "name": "rotation"},
                {'params': [self.model.fields.field_params["fg"]._regist_feat], 'lr': self.opts["feature_lr"], "name": "regist_feat"}
            ]
            
            if self.opts["gs_learnable_bg"]:
                l.append({'params': [self.model.fields.field_params["fg"].learnable_bkgd], 'lr': self.opts["feature_lr"], "name": "bg_rgb"})
            self.gs_optimizer = torch.optim.Adam(l, lr=opts["learning_rate"], eps=1e-15)
        # self.optimizer = torch.optim.Adam(params_list, lr=opts["learning_rate"], eps=1e-15)

        self.optimizer = torch.optim.AdamW(
            params_list,
            lr=opts["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )

        # self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # initial_lr = lr/div_factor
        # min_lr = initial_lr/final_div_factor
        if is_resumed:
            div_factor = 1.0
            final_div_factor = 5.0
            pct_start = 0.0  # cannot be 0
        else:
            div_factor = 25.0
            final_div_factor = 1.0
            pct_start = 2.0 / opts["num_rounds"]  # use 2 epochs to warm up
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            lr_list,
            int(self.total_steps),
            pct_start=pct_start,
            cycle_momentum=False,
            anneal_strategy="linear",
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )

    def train(self):
        """Training loop"""
        opts = self.opts

        # clear buffers for pytorch1.10+
        try:
            self.model._assign_modules_buffers()
        except:
            pass

        # start training loop
        self.save_checkpoint(round_count=self.current_round)
        for round_count in range(
            self.current_round, self.current_round + opts["num_rounds"]
        ):
            start_time = time.time()
            with torch_profile(
                self.save_dir, f"{round_count:03d}", enabled=opts["profile"]
            ):
                self.run_one_round(round_count)

            if get_local_rank() == 0:
                print(f"Round {round_count:03d}: time={time.time() - start_time:.3f}s")

    def run_one_round(self, round_count):
        """Evaluation and training for a single round

        Args:
            round_count (int): Current round index
        """

        self.model.eval()
        if get_local_rank() == 0:# and self.current_round > 0
            if not (self.current_round == 0 and self.opts["debug"]):
                with torch.no_grad():
                    self.model_eval()
        self.model.update_geometry_aux()

        if round_count % 20 == 0 or round_count in [0,1,2,3,4,5,6,7,8,9,10]:
            self.model.export_geometry_aux("%s/%03d" % (self.save_dir, round_count))
        if round_count == self.opts["num_rounds"] - 1:
            self.model.export_geometry_aux("%s/%03d" % (self.save_dir, round_count+1))
        self.model.train()
        self.train_one_round(round_count)
        self.current_round += 1
        self.save_checkpoint(round_count=self.current_round)

    def save_checkpoint(self, round_count):
        """Save model checkpoint to disk

        Args:
            round_count (int): Current round index
        """
        opts = self.opts
        # move to the left
        self.model_cache[0] = self.model_cache[1]
        self.optimizer_cache[0] = self.optimizer_cache[1]
        self.scheduler_cache[0] = self.scheduler_cache[1]
        # enqueue
        self.model_cache[1] = deepcopy(self.model.state_dict())
        try:
            self.optimizer_cache[1] = deepcopy(self.optimizer.state_dict())
        except:
            print("optimizer_cache is not saved")
            self.optimizer_cache[1] = self.optimizer_cache[0]
        self.scheduler_cache[1] = deepcopy(self.scheduler.state_dict())

        if get_local_rank() == 0 and round_count % opts["save_freq"] == 0:
            print("saving round %d" % round_count)
            param_path = "%s/ckpt_%04d.pth" % (self.save_dir, round_count)

            checkpoint = {
                "current_steps": self.current_steps,
                "current_round": self.current_round,
                "model": self.model_cache[1],
                "optimizer": self.optimizer_cache[1],
            }

            torch.save(checkpoint, param_path)
            # copy to latest
            latest_path = "%s/ckpt_latest.pth" % (self.save_dir)
            os.system("cp %s %s" % (param_path, latest_path))

    @staticmethod
    def load_checkpoint(load_path, model, optimizer=None, load_ply=False):
        """Load a model from checkpoint

        Args:
            load_path (str): Path to checkpoint
            model (dvr_model): Model to update in place
            optimizer (torch.optim.Optimizer or None): If provided, load
                learning rate from checkpoint
        """
        checkpoint = torch.load(load_path)
        model_states = checkpoint["model"]
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_states = remove_ddp_prefix(model_states)

        if "fields.field_params.fg._xyz" in model_states.keys() and not load_ply:
            scale = torch.tensor([1])  # scale of the field
            model.fields.field_params.fg.logscale = torch.nn.Parameter(scale.log())

            num_pts = model_states["fields.field_params.fg._xyz"].shape[0]
            new_xyz = model.fields.field_params.fg._xyz[:num_pts].clone().detach()
            model.fields.field_params.fg._xyz = torch.nn.Parameter(new_xyz)
            model.fields.field_params.fg._xyz = torch.nn.Parameter(model.fields.field_params.fg._xyz[:num_pts].clone().detach())
            model.fields.field_params.fg._features_dc = torch.nn.Parameter(model.fields.field_params.fg._features_dc[:num_pts].clone().detach())
            model.fields.field_params.fg._features_rest = torch.nn.Parameter(model.fields.field_params.fg._features_rest[:num_pts].clone().detach())
            model.fields.field_params.fg._opacity = torch.nn.Parameter(model.fields.field_params.fg._opacity[:num_pts].clone().detach())
            model.fields.field_params.fg._scaling = torch.nn.Parameter(model.fields.field_params.fg._scaling[:num_pts].clone().detach())
            model.fields.field_params.fg._rotation = torch.nn.Parameter(model.fields.field_params.fg._rotation[:num_pts].clone().detach())
            model.fields.field_params.fg._regist_feat = torch.nn.Parameter(model.fields.field_params.fg._regist_feat[:num_pts].clone().detach())
        
        if model.config["not_load_warping"] and False:
            new_state_dict = {k: v for k, v in model_states.items() if "fields.field_params.fg.warp" not in k}
            # import ipdb; ipdb.set_trace()
            model_states = new_state_dict

        model.load_state_dict(model_states, strict=False)

        if hasattr(model.fields.field_params,"fgneus"):
            new_state_dict = {}
            for key, value in model_states.items():
                new_key = key.replace("fg.", "fgneus.")
                new_state_dict[new_key] = value
            
            model.load_state_dict(new_state_dict, strict=False)

        # if optimizer is not None:
        #     # use the new param_groups that contains the learning rate
        #     checkpoint["optimizer"]["param_groups"] = optimizer.state_dict()[
        #         "param_groups"
        #     ]
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint

    def load_checkpoint_train(self):
        """Load a checkpoint at training time and update the current step count
        and round count
        """
        # training time load_path, model, optimizer=None, load_ply=False
        checkpoint = self.load_checkpoint(
            self.opts["load_path"], self.model, optimizer=self.optimizer, load_ply=len(self.opts["gs_init_ply"])>0
        )
        if not self.opts["reset_steps"]:
            self.current_steps = checkpoint["current_steps"]
            self.current_round = checkpoint["current_round"]

        # reset near_far
        self.model.fields.reset_geometry_aux()

    def train_one_round(self, round_count):
        """Train a single round (going over mini-batches)

        Args:
            round_count (int): round index
        """

        opts = self.opts
        torch.cuda.empty_cache()
        self.model.train()
        self.optimizer.zero_grad()
        self.trainloader.sampler.set_epoch(round_count)  # necessary for shuffling

        
        for i, batch in enumerate(self.trainloader):
            if i == opts["iters_per_round"]:
                break

            # batch["H"] = (torch.ones_like(batch['frameid_sub']) * self.data_info["raw_size"][0,0] / self.opts["resolution_scale"]).long()
            # batch["W"] = (torch.ones_like(batch['frameid_sub']) * self.data_info["raw_size"][0,1] / self.opts["resolution_scale"]).long()
            batch["H"] = (torch.ones_like(batch['frameid_sub']) * batch['hxy'].shape[2]).long()
            batch["W"] = (torch.ones_like(batch['frameid_sub']) * batch['hxy'].shape[3]).long()
            self.model.set_progress(self.current_steps)
            if "gs" in self.opts['fg_motion']:
                # self.model.fields.field_params["fg"].active_sh_degree = self.model.fields.field_params["fg"].max_sh_degree # max sh degree
                self.model.fields.field_params["fg"].update_learning_rate(self.gs_optimizer, self.current_steps)
                if self.current_steps % 1000 == 0:
                    self.model.fields.field_params["fg"].oneupSHdegree()

                assert batch["feature"].shape[0] == 1
                reshaped_feat = batch["feature"][0].permute(0,3,1,2)
                batch["feature"] = torch.nn.functional.interpolate(reshaped_feat, size=(batch["H"][0,0],batch["W"][0,0]), scale_factor=None, mode='nearest', align_corners=None).permute(0,2,3,1)[None]

            for k,v in batch.items():
                if v.device != self.device:
                    batch[k] = v.to(self.device)

            loss_dict = self.model(batch)
            if self.opts["rgb_loss_only"]:
                loss_to_keep = ["rgb", "mask", "normal_loss", "dist_loss", "reg_volume_loss"]
                # loss_to_keep = ["rgb", "mask"]
                # loss_to_keep = ["rgb", "mask","feature",'depth','feat_reproj','flow','reg_deform_cyc','reg_skin_entropy',]
                for k in list(loss_dict.keys()):
                    if k not in loss_to_keep:
                        loss_dict.pop(k)

            total_loss = torch.sum(torch.stack(list(loss_dict.values())))
            total_loss.mean().backward()


            # visualize gs result
            if "gs" in self.opts['fg_motion']:
                mask = batch["mask"].float()
                rendered = self.model.cached_results["rendered"]
                for k, v in rendered.items():
                    if k in ['ref_rgb', 'rendered', 'mask', "rend_dist", "rend_normal", "surf_normal"]:
                        img_grid = v[0].detach()
                        if img_grid.shape[2] > 3:
                            img_grid = img_grid[...,:3]
                        elif img_grid.shape[2] == 2:
                            img_grid = torch.concat([img_grid, torch.zeros_like(img_grid[...,:1])], dim=2)
                        # save_image(img_grid.permute(2,0,1), f"logdir/{self.opts['logname']}/{k}_{self.current_round}.jpg")
                        save_image(img_grid.permute(2,0,1), f"tmp/{self.opts['seqname']}_{self.opts['logname']}_{k}.jpg")

                        if self.current_round % 1000 == 0:
                            os.makedirs(f"logdir/{self.opts['logname']}/imglog", exist_ok=True)
                            save_image(img_grid.permute(2,0,1), f"logdir/{self.opts['logname']}/imglog/{self.current_round}_{k}.jpg")
            # tensorboard logger
            if "gs" in self.opts['fg_motion']:
                log_dict = {}
                log_dict["num of points"] = self.model.fields.field_params["fg"]._xyz.shape[0]
                log_dict["xyz mean"] = self.model.fields.field_params["fg"]._xyz.abs().mean().item()
                log_dict["f_dc mean"] = self.model.fields.field_params["fg"]._features_dc.abs().mean().item()
                log_dict["f_rest mean"] = self.model.fields.field_params["fg"]._features_rest.abs().mean().item()
                log_dict["opacity mean"] = self.model.fields.field_params["fg"]._opacity.abs().mean().item()
                log_dict["scaling mean"] = self.model.fields.field_params["fg"]._scaling.abs().mean().item()
                log_dict["rotation mean"] = self.model.fields.field_params["fg"]._rotation.abs().mean().item()
                log_dict["scaling max"] = self.model.fields.field_params["fg"]._scaling.abs().max().item()

                log_dict["xyz.grad mean"] = self.model.fields.field_params["fg"]._xyz.grad.abs().mean().item()
                log_dict["f_dc.grad mean"] = self.model.fields.field_params["fg"]._features_dc.grad.abs().mean().item()
                log_dict["f_rest.grad mean"] = self.model.fields.field_params["fg"]._features_rest.grad.abs().mean().item()
                log_dict["opacity.grad mean"] = self.model.fields.field_params["fg"]._opacity.grad.abs().mean().item()
                log_dict["scaling.grad mean"] = self.model.fields.field_params["fg"]._scaling.grad.abs().mean().item()
                log_dict["rotation.grad mean"] = self.model.fields.field_params["fg"]._rotation.grad.abs().mean().item()

                # self.add_scalar(self.log, log_dict, self.current_steps)

            # tmp image logger and console logger
            if self.current_steps % 100 == 0:
                print("\n-------iter", self.current_steps, "total loss", total_loss.item(), "-------")
                # print("\n-------iter", self.current_steps, "total loss", "-------")

                # for k,v in self.model.cached_results["rendered"].items():
                #     if v is not None and k in ["rendered", "mask", "depth"]:
                #         # import pdb;pdb.set_trace()
                #         self.add_image(self.log, "training_"+k, v[0].detach(), self.current_steps)
                # self.add_image(self.log, "training_rgb", batch["rgb"][0], self.current_steps)
                # PIL.Image.fromarray(np.clip(batch['rgb'].detach().cpu().numpy()*255,0,255).astype(np.uint8)).save(f'debug/ori_{self.current_steps}.png')
                # PIL.Image.fromarray(np.clip(self.model.cached_results["rendered"]['rgb'].detach().cpu().numpy()*255,0,255).astype(np.uint8)).save(f'debug/render_{self.current_steps}.png')
                #print("num of gs is", self.model.fields.field_params["fg"]._xyz.shape[0])
                # print("aabb is", self.model.fields.field_params["fg"].aabb)
                # sort the value of loss_dict
                sorted_loss_dict = dict(sorted(loss_dict.items(), key=lambda item: -item[1]))
                for k, v in sorted_loss_dict.items():
                    if v > 0 or math.isnan(v):
                        print(k, v.sum().item())

            self.check_grad()

            # GS densification
            with torch.no_grad():
                if "gs" in self.opts['fg_motion'] and self.current_steps < self.opts["densify_until_iter"]:
                    # TODO: check check
                    # Keep track of max radii in image-space for pruning
                    for batch_idx in range(batch['H'].shape[0]):
                        visibility_filter = self.model.fields.field_params['fg']._visibility_filter_batch[batch_idx]
                        viewspace_points = self.model.fields.field_params['fg']._viewspace_points_batch[batch_idx]
                        radii = self.model.fields.field_params['fg']._radii_batch[batch_idx]

                        self.model.fields.field_params["fg"].max_radii2D[visibility_filter] = torch.max(self.model.fields.field_params["fg"].max_radii2D[visibility_filter], radii[visibility_filter])
                        self.model.fields.field_params["fg"].add_densification_stats(viewspace_points, visibility_filter)

                    if self.current_steps > self.opts["densify_from_iter"] and self.current_steps % self.opts["densification_interval"] == 0:
                        size_threshold = 20 if self.current_steps > self.opts["opacity_reset_interval"] else None
                        self.model.fields.field_params["fg"].densify_and_prune(self.opts["densify_grad_threshold"], 0.005, self.model.fields.field_params["fg"].cameras_extent, size_threshold)

                    # densify extrme large point
                    if self.current_steps > self.opts["densify_from_iter"] and self.current_steps % (10 * self.opts["densification_interval"]) == 0:
                        self.model.fields.field_params["fg"].densify_and_prune(self.opts["densify_grad_threshold"] * 0.1, 0.002, self.model.fields.field_params["fg"].cameras_extent * 100 , size_threshold)
                    
                    if self.current_steps % self.opts["opacity_reset_interval"] == 0 or (self.opts["white_background"] and self.current_steps == self.opts["densify_from_iter"]):
                        self.model.fields.field_params["fg"].reset_opacity()
                    
                    if self.current_steps > self.opts["densify_from_iter"] and self.current_steps < self.opts["outlier_stop_iter"] and self.current_steps % self.opts["outlier_filtering_interval"] == 0 :
                        # self.model.fields.field_params["fg"].densify_and_prune(self.opts["densify_grad_threshold"] * 0.1, 0.01, self.model.fields.field_params["fg"].cameras_extent * 100 , size_threshold)

                        xyz = self.model.fields.field_params["fg"].get_xyz.cpu().detach()
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz))
                        pcd_rad, ind_rad = pcd.remove_radius_outlier(nb_points=20, radius = 0.004)
                        prune_mask = torch.ones_like(xyz[:,0],dtype=bool)
                        prune_mask[ind_rad] = False
                        # pcd_rad, ind_rad = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.02)
                        # pcd = pcd.select_by_index(pcd_rad)
                        # o3d.io.write_point_cloud('outlier.ply', pcd)
                        # o3d.io.write_point_cloud('outlier2.ply', pcd_rad)

                        # prune_mask = None
                        self.model.fields.field_params["fg"].prune_points(prune_mask)
            if "gs" in self.opts['fg_motion']:
                self.gs_optimizer.step()
                self.gs_optimizer.zero_grad(set_to_none=True)
            if not "gs" in self.opts['fg_motion'] or self.opts["gs_optim_warp"]:
                if not (self.current_steps < self.opts["optim_warp_neus_iters"] and "gs" in self.opts['fg_motion']):
                    # import ipdb; ipdb.set_trace()
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            # if get_local_rank() == 0:
            #     self.add_scalar(self.log, loss_dict, self.current_steps)
            self.current_steps += 1

    @staticmethod
    def construct_dataset_opts(opts, is_eval=False, dataset_constructor=VidDataset):
        """Extract train/eval dataloader options from command-line args.

        Args:
            opts (Dict): Command-line options
            is_eval (bool): When training a model (`is_eval=False`), duplicate
                the dataset to fix the number of iterations per round
            dataset_constructor (torch.utils.data.Dataset): Dataset class to use
        """
        opts_dict = {}
        opts_dict["seqname"] = opts["seqname"]
        opts_dict["load_pair"] = True
        opts_dict["data_prefix"] = "%s-%d" % (opts["data_prefix"], opts["train_res"])
        opts_dict["feature_type"] = opts["feature_type"]
        opts_dict["dataset_constructor"] = dataset_constructor

        if is_eval:
            opts_dict["multiply"] = False
            opts_dict["pixels_per_image"] = -1
            opts_dict["delta_list"] = []
        else:
            # duplicate dataset to fix number of iterations per round
            opts_dict["multiply"] = True
            opts_dict["pixels_per_image"] = opts["pixels_per_image"]
            opts_dict["delta_list"] = [2, 4, 8]
            opts_dict["num_workers"] = opts["num_workers"]

            opts_dict["imgs_per_gpu"] = opts["imgs_per_gpu"]
            opts_dict["iters_per_round"] = opts["iters_per_round"]
            opts_dict["ngpu"] = opts["ngpu"]
            opts_dict["local_rank"] = get_local_rank()
        return opts_dict

    def print_sum_params(self):
        """Print the sum of parameters"""
        sum = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                sum += p.abs().sum()
        print(f"{sum:.16f}")

    def model_eval(self):
        """Evaluate the current model"""
        torch.cuda.empty_cache()
        ref_dict, batch = self.load_batch(self.evalloader.dataset, self.eval_fid)
        self.construct_eval_batch(batch)
        # import ipdb; ipdb.set_trace()
        # batch["H"] = torch.tensor(self.data_info["raw_size"][0,0].repeat(batch['dataid'].shape[0]))
        # batch["W"] = torch.tensor(self.data_info["raw_size"][0,1].repeat(batch['dataid'].shape[0]))
        batch["H"] = torch.tensor(self.opts["eval_res"]).repeat(batch['dataid'].shape[0])
        batch["W"] = torch.tensor(self.opts["eval_res"]).repeat(batch['dataid'].shape[0])
        rendered = self.model.evaluate(batch)
        self.add_image_togrid(ref_dict)
        self.add_image_togrid(rendered)
        self.visualize_matches(rendered["xyz"], rendered["xyz_matches"], tag="xyz")
        self.visualize_matches(
            rendered["xyz_cam"], rendered["xyz_reproj"], tag="xyz_cam"
        )

    def visualize_matches(self, xyz, xyz_matches, tag):
        """Visualize dense correspondences outputted by canonical registration

        Args:
            xyz: (M,H,W,3) Predicted xyz points
            xyz_matches: (M,H,W,3) Points to match against in canonical space.
                This is an empty list for the static background model
            tag (str): Name of export mesh
        """
        if len(xyz_matches) == 0:
            return
        xyz = xyz[0].view(-1, 3).detach().cpu().numpy()
        xyz_matches = xyz_matches[0].view(-1, 3).detach().cpu().numpy()
        xyz = trimesh.Trimesh(vertices=xyz)
        xyz_matches = trimesh.Trimesh(vertices=xyz_matches)

        xyz.visual.vertex_colors = [255, 0, 0, 255]
        xyz_matches.visual.vertex_colors = [0, 255, 0, 255]
        xyz_cat = trimesh.util.concatenate([xyz, xyz_matches])

        xyz_cat.export("%s/%03d-%s.obj" % (self.save_dir, self.current_round, tag))

    @staticmethod
    def load_batch(dataset, fids):
        """Load a single batch of reference frames for Tensorboard visualization

        Args:
            dataset (ConcatDataset): Eval dataset for all videos in a sequence
            fids: (nframes,) Frame indices to load
        Returns:
            ref_dict (Dict): Dict with keys "ref_rgb", "ref_mask", "ref_depth",
                "ref_feature", and "ref_flow", each (N,H,W,x)
            batch_aggr (Dict): Batch of input metadata. Keys: "dataid",
                "frameid_sub", "crop2raw", and "feature"
        """
        ref_dict = defaultdict(list)
        batch_aggr = defaultdict(list)
        ref_keys = ["rgb", "mask", "depth", "feature", "vis2d"]
        batch_keys = ["dataid", "frameid_sub", "crop2raw"]
        for fid in fids:
            batch = dataset[fid]
            for k in ref_keys:
                ref_dict["ref_%s" % k].append(batch[k][:1])
            ref_dict["ref_flow"].append(
                batch["flow"][:1] * (batch["flow_uct"][:1] > 0).astype(float)
            )

            for k in batch_keys:
                batch_aggr[k].append(batch[k])
            batch_aggr["feature"].append(
                batch["feature"].reshape(2, -1, batch["feature"].shape[-1])
            )

        for k, v in ref_dict.items():
            ref_dict[k] = np.concatenate(v, 0)

        for k, v in batch_aggr.items():
            batch_aggr[k] = np.concatenate(v, 0)
        return ref_dict, batch_aggr

    def construct_eval_batch(self, batch):
        """Modify a batch in-place for evaluation

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid",
                "frameid_sub", "crop2raw", and "feature". This function
                modifies it in place to add key "hxy"
        """
        opts = self.opts
        # to tensor
        for k, v in batch.items():
            batch[k] = torch.tensor(v, device=self.device)

        batch["crop2raw"][..., :2] *= opts["train_res"] / opts["eval_res"]

        if not hasattr(self, "hxy"):
            hxy = self.create_xy_grid(opts["eval_res"], self.device)
            self.hxy_cache = hxy[None].expand(len(batch["dataid"]), -1, -1)
        batch["hxy"] = self.hxy_cache

    @staticmethod
    def create_xy_grid(eval_res, device):
        """Create a grid of pixel coordinates on the image plane

        Args:
            eval_res (int): Resolution to evaluate at
            device (torch.device): Target device
        Returns:
            hxy: (eval_res^2, 3) Homogeneous pixel coords on the image plane
        """
        # if eval_res is not int
        if not isinstance(eval_res, int):
            H = eval_res[0]
            W = eval_res[1]
            eval_range_H = torch.arange(H, dtype=torch.float32, device=device)
            eval_range_W = torch.arange(W, dtype=torch.float32, device=device)
            hxy = torch.cartesian_prod(eval_range_H, eval_range_W)
        else:
            eval_range = torch.arange(eval_res, dtype=torch.float32, device=device)
            hxy = torch.cartesian_prod(eval_range, eval_range)
        hxy = torch.stack([hxy[:, 1], hxy[:, 0], torch.ones_like(hxy[:, 0])], -1)
        return hxy

    def add_image_togrid(self, rendered_seq):
        """Add rendered outputs to Tensorboard visualization grid

        Args:
            rendered_seq (Dict): Dict of volume-rendered outputs. Keys:
                "mask" (M,H,W,1), "vis2d" (M,H,W,1), "depth" (M,H,W,1),
                "flow" (M,H,W,2), "feature" (M,H,W,16), "normal" (M,H,W,3), and
                "eikonal" (M,H,W,1)
        """
        for k, v in rendered_seq.items():
            # if k in ['ref_rgb', 'ref_mask', 'ref_depth', 'ref_feature', 'ref_vis2d', 'ref_flow', 'rendered', 'depth', 'mask', 'feature', 'flow','rgb_neus', 'depth_neus', 'mask_fgneus_neus','mask_neus', 'feature_neus', 'flow_neus', 'normal_neus']:
            if k in ['ref_rgb', 'ref_mask', 'rendered', 'mask']:
                img_grid = make_image_grid(v)
            # save tensor image 
            
            # import ipdb; ipdb.set_trace()
                self.add_image(self.log, k, img_grid, self.current_round)
                # print("img_grid", img_grid.shape, img_grid.min(), img_grid.max())

                if self.current_round % 20 == 0:
                    if img_grid.shape[2] > 3:
                        img_grid = img_grid[...,:3]
                    elif img_grid.shape[2] == 2:
                        img_grid = torch.concat([img_grid, torch.zeros_like(img_grid[...,:1])], dim=2)
                    
                    logdir = "%s/%s-%s" % (self.opts["logroot"], self.opts["seqname"], self.opts["logname"])
                    os.makedirs(logdir, exist_ok=True)
                    save_image(img_grid.permute(2,0,1), f"{logdir}/eval_{self.current_round}_{k}.jpg")

    def add_image(self, log, tag, img, step):
        # print("img", tag, img.shape)
        """Convert volume-rendered outputs to RGB and add to Tensorboard

        Args:
            log (SummaryWriter): Tensorboard logger
            tag (str): Image tag
            img: (H_out, W_out, x) Image to show
            step (int): Current step
        """
        if len(img.shape) == 2:
            formats = "HW"
        else:
            formats = "HWC"

        img = img2color(tag, img, pca_fn=self.data_info["apply_pca_fn"])

        log.add_image("img_" + tag, img, step, dataformats=formats)

    @staticmethod
    def add_scalar(log, dict, step):
        """Add a scalar value to Tensorboard log"""
        for k, v in dict.items():
            log.add_scalar(k, v, step)

    @staticmethod
    def construct_test_model(opts):
        """Load a model at test time

        Args:
            opts (Dict): Command-line options
        """
        # io
        logname = "%s-%s" % (opts["seqname"], opts["logname"])

        # construct dataset
        eval_dict = Trainer.construct_dataset_opts(opts, is_eval=True)
        evalloader = data_utils.eval_loader(eval_dict)
        data_info, _ = data_utils.get_data_info(evalloader)

        # construct DVR model
        model = dvr_model(opts, data_info)
        load_path = "%s/%s/ckpt_%s.pth" % (
            opts["logroot"],
            logname,
            opts["load_suffix"],
        )
        _ = Trainer.load_checkpoint(load_path, model, load_ply=False)
        model.cuda()
        model.eval()

        # get reference images
        inst_id = opts["inst_id"]
        offset = data_info["frame_info"]["frame_offset"]
        frame_id = np.asarray(
            range(offset[inst_id] - inst_id, offset[inst_id + 1] - inst_id - 1)
        )  # to account for pairs
        ref_dict, _ = Trainer.load_batch(evalloader.dataset, frame_id)

        mask_nps_pth = [x.replace("JPEGImages", "Annotations").replace("jpg","npy") for x in evalloader.dataset.datasets[0].dict_list['ref']]
        mask_nps = [np.load(x) for x in mask_nps_pth]
        ref_dict["ref_mask_full"] = np.stack(mask_nps, 0)[..., None]

        return model, data_info, ref_dict

    def check_grad(self, thresh=5.0):
        """Check if gradients are above a threshold

        Args:
            thresh (float): Gradient clipping threshold
        """
        # parameters that are sensitive to large gradients

        param_list = []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                param_list.append(p)

        grad_norm = torch.nn.utils.clip_grad_norm_(param_list, thresh)
        if grad_norm > thresh and False:
            # clear gradients
            self.optimizer.zero_grad()
            # load cached model from two rounds ago
            if self.model_cache[0] is not None:
                if get_local_rank() == 0:
                    print("large grad: %.2f, resume from cached weights" % grad_norm)
                self.model.load_state_dict(self.model_cache[0])
                self.optimizer.load_state_dict(self.optimizer_cache[0])
                self.scheduler.load_state_dict(self.scheduler_cache[0])


