# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os

from absl import flags

opts = flags.FLAGS

class FlexicubeConfig:
    flags.DEFINE_integer("tet_grid_size", 80,"")
    

class Gen3DConfig:

    flags.DEFINE_integer('ablation_lockframeid', -1,"")
    flags.DEFINE_boolean('w_bg', False,"white backgound")
    flags.DEFINE_boolean('no_negative_prompt', False,"")
    flags.DEFINE_boolean("freeze_warp", False,"")
    flags.DEFINE_boolean("test_in_train", False,"")
    flags.DEFINE_boolean("recon_keep_coarse", False,"")
    flags.DEFINE_boolean('gen3d_nosetprogress', False, "gen3d set_progress like recon")
    flags.DEFINE_boolean("gen3d_optim_all", False, "False not optimize embbeding")

    
    flags.DEFINE_float("top_alpha", 1,"")
    flags.DEFINE_float("gs", 50.0, "guidance_scale")
    flags.DEFINE_string("gen3d_guidance", 'mvd', "if or mvd")
    flags.DEFINE_boolean("recon_no_coarsetofine", False,"")
    flags.DEFINE_boolean('use_wide_near_far', False, "")
    flags.DEFINE_boolean('rgb_timefree', False, "not use time dependet nerf rgb")
    flags.DEFINE_boolean('rgb_dirfree', False, "not use dir dependet nerf rgb")
    flags.DEFINE_float("gen3d_wt", 0, "weight from 3d generation loss")
    flags.DEFINE_integer("gen3d_res", 64, "")
    flags.DEFINE_float("gen3d_dist", 1, 'render distance')
    flags.DEFINE_float("gen3d_freq", 2, "? images one sds")


    # flags.DEFINE_integer("gen3d_start_iters", 0, "for sds_t and camera angle anneling")
    # flags.DEFINE_boolean("gen3d_dirprompt", False, "")

    flags.DEFINE_boolean("render_uncert", False, "")
    flags.DEFINE_integer("gen3d_frameid", -1, "-1 for random")


    flags.DEFINE_integer("seed",-1, "-1 for not seed anything")
    flags.DEFINE_boolean("gen3d_random_bkgd", False, "")
    flags.DEFINE_string("prompt", 'A_photo_of_a_cat', "!!!NOTE: split by _ name of the sequence")
    flags.DEFINE_boolean("reset_rgb_mlp", False, "")
    flags.DEFINE_float("gen3d_sds_t_max", 0.98, "sds_t_max")
    flags.DEFINE_float('rgb_loss_anneal', -1, "x:(0, total_iters * self.config['rgb_loss_anneal']) y:(1, 0)")
    flags.DEFINE_float('mask_loss_anneal', -1, "x:(0, total_iters * self.config['mask_loss_anneal']) y:(1, 0)")
    flags.DEFINE_float('all_reconloss_anneal', -1, "x:(0, total_iters * self.config['mask_loss_anneal']) y:(1, 0)")
    flags.DEFINE_float('gen3d_loss_anneal', 0, "<0 from 1 to 0 >0 from 0 to 1")

    flags.DEFINE_boolean("rgb_only", False, "")
    flags.DEFINE_boolean("geo_only", False, "")
    
    flags.DEFINE_string('rgb_anneal_type', "log", "linear or log")
    flags.DEFINE_string('anneal_type', "linear", "linear or log")

    flags.DEFINE_float('reg_anneal', 1,'') 


    
    # ["gen3d_jacobloss"] or self.opts["gen3d_sds_normal"]
    # flags.DEFINE_boolean("gen3d_regloss", False, "")

    flags.DEFINE_boolean("gen3d_jacobloss", False, "")
    flags.DEFINE_boolean("gen3d_cycloss", False, "")
    flags.DEFINE_boolean("gen3d_sds_normal", False, "")
    # flags.DEFINE_integer("num_rounds", 20, "number of rounds to train")
        

    flags.DEFINE_integer("lock_frameid", -1, "lock frameid for rgb querying")

class TrainModelConfig:
    # weights of reconstruction terms
    flags.DEFINE_float("mask_wt", 0.1, "weight for silhouette loss")
    flags.DEFINE_float("rgb_wt", 0.1, "weight for color loss")
    flags.DEFINE_float("depth_wt", 1e-4, "weight for depth loss")
    flags.DEFINE_float("flow_wt", 0.5, "weight for flow loss")
    flags.DEFINE_float("vis_wt", 1e-2, "weight for visibility loss")
    flags.DEFINE_float("feature_wt", 1e-2, "weight for feature reconstruction loss")
    flags.DEFINE_float("feat_reproj_wt", 5e-2, "weight for feature reprojection loss")


    # weights of regularization terms
    flags.DEFINE_float(
        "reg_visibility_wt", 1e-4, "weight for visibility regularization"
    )
    flags.DEFINE_float("reg_eikonal_wt", 1e-3, "weight for eikonal regularization")
    flags.DEFINE_float(
        "reg_deform_cyc_wt", 0.01, "weight for deform cyc regularization"
    )
    flags.DEFINE_float("reg_delta_skin_wt", 5e-3, "weight for delta skinning reg")
    flags.DEFINE_float("reg_skin_entropy_wt", 5e-4, "weight for delta skinning reg")
    flags.DEFINE_float(
        "reg_gauss_skin_wt", 1e-3, "weight for gauss skinning consistency"
    )
    flags.DEFINE_float("reg_cam_prior_wt", 0.1, "weight for camera regularization")
    flags.DEFINE_float("reg_skel_prior_wt", 0.1, "weight for skeleton regularization")
    flags.DEFINE_float(
        "reg_gauss_mask_wt", 0.01, "weight for gauss mask regularization"
    )
    flags.DEFINE_float("reg_soft_deform_wt", 100.0, "weight for soft deformation reg")

    # model
    flags.DEFINE_string("field_type", "fg", "{bg, fg, comp}")
    flags.DEFINE_string(
        "fg_motion", "rigid", "{rigid, dense, bob, skel-human, skel-quad, gs-XXX}"
    )
    flags.DEFINE_bool("single_inst", True, "assume the same morphology over objs")


class TrainOptConfig:
    # io-related
    flags.DEFINE_string("seqname", "cat", "name of the sequence")
    flags.DEFINE_string("logname", "tmp", "name of the saved log")
    flags.DEFINE_string(
        "data_prefix", "crop", "prefix of the data entries, {crop, full}"
    )
    flags.DEFINE_integer("train_res", 256, "size of training images")
    flags.DEFINE_string("logroot", "logdir/", "root directory for log files")
    flags.DEFINE_string("load_suffix", "", "sufix of params, {latest, 0, 10, ...}")
    flags.DEFINE_string("feature_type", "dinov2", "{dinov2, cse}")
    flags.DEFINE_string("load_path", "", "path to load pretrained model")
    flags.DEFINE_string("lab4d_init_mesh", "", "init mesh")

    # accuracy-related
    flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
    flags.DEFINE_integer("num_rounds", 20, "number of rounds to train")
    flags.DEFINE_integer("iters_per_round", 200, "number of iterations per round")
    flags.DEFINE_integer("imgs_per_gpu", 256, "images samples per iter, per gpu")
    flags.DEFINE_integer("pixels_per_image", 16, "pixel samples per image")
    # flags.DEFINE_integer("imgs_per_gpu", 1, "size of minibatches per iter")
    # flags.DEFINE_integer("pixels_per_image", 4096, "number of pixel samples per image")
    flags.DEFINE_boolean(
        "freeze_bone_len", False, "do not change bone length of skeleton"
    )
    flags.DEFINE_boolean(
        "reset_steps",
        True,
        "reset steps of loss scheduling, set to False if resuming training",
    )

    flags.DEFINE_boolean("no_loss_mask", False, "")

    # efficiency-related
    flags.DEFINE_integer("ngpu", 1, "number of gpus to use")
    flags.DEFINE_integer("num_workers", 8, "Number of workers for dataloading")
    flags.DEFINE_integer("eval_res", 128, "size used for eval visualizations")
    flags.DEFINE_integer("save_freq", 10, "params saving frequency")
    flags.DEFINE_boolean("profile", False, "profile the training loop")

class GaussianConfig:
    flags.DEFINE_boolean("debug_cuda", False, "")
    # flags.DEFINE_boolean("use_gs_optimizer", False, "")
    flags.DEFINE_boolean("gs_optim_warp", True, "")
    flags.DEFINE_boolean("gs_learnable_bg", True, "")
    flags.DEFINE_float("intrinsics_lr_mult", 1, "")
    flags.DEFINE_float("arap_wt", 0.0, "")
    flags.DEFINE_boolean("rgb_loss_only", False, "")
    # quantitative evaluation
    flags.DEFINE_boolean("quant_exp", False, "set opts_dict['delta_list'] = [2, 4, 8] to [4,8] and take idx%4 for train and idx%4 +2 for eval")
    flags.DEFINE_boolean("not_load_warping", False, "")

    # GSDF
    flags.DEFINE_boolean("two_branch", False, "neus+gs")
    flags.DEFINE_integer("dgs_k", 4, "depth guided sampling k")
    flags.DEFINE_integer("neus_branch_reso", 64, "need can divide 256")
    flags.DEFINE_integer("optim_warp_neus_iters", 12000, "use banmo optimzier to optimze neus and warp")
    flags.DEFINE_integer("start_mutual_iters", 999999, "")
    flags.DEFINE_float("mutual_depth_wt", 1, "")
    flags.DEFINE_float("mutual_normal_wt", 1, "")
    flags.DEFINE_float("mutual_mask_wt", 1, "")
    flags.DEFINE_bool("depth_guide_sample", False, "如果有就gs深度图SDF来帮助neus采样,且neus也学习。否则neus不学习,只用来规范gs深度和法向")
    flags.DEFINE_integer("novel_neus_interv", -1, "新视角用neus监督gs mutual loss")
    
    # 2dgs
    flags.DEFINE_boolean("force_center_cam", False, "force center cam")
    flags.DEFINE_boolean("reg_in_cano", False, "2dgs reg in canonical space")
    flags.DEFINE_float("lambda_dist", 0, "weight for distance loss")
    flags.DEFINE_float("lambda_normal", 0.05, "weight for reg loss")
    flags.DEFINE_float("reg_volume_loss_wt", 0.00, "weight for volume loss")

    flags.DEFINE_boolean("maskloss_no_vis2d",False,"")
    flags.DEFINE_boolean("vis2d_dilate",False,"")

    flags.DEFINE_string("ip", "127.0.0.1", "")
    flags.DEFINE_integer("port", 6322, "")
    flags.DEFINE_integer("debug_from", -1, "")
    flags.DEFINE_boolean("detect_anomaly", False, "")
    flags.DEFINE_list("test_iterations", [7000, 30000], "")
    flags.DEFINE_list("save_iterations", [7000, 30000], "")
    flags.DEFINE_boolean("quiet", False, "")
    flags.DEFINE_list("checkpoint_iterations", [30000], "")
    flags.DEFINE_string("start_checkpoint", None, "")

    flags.DEFINE_integer("sh_degree", 3, "3 in gs")
    flags.DEFINE_string("source_path", "", "")
    flags.DEFINE_string("model_path", "", "")
    flags.DEFINE_string("images", "images", "")
    flags.DEFINE_integer("resolution", -1, "")
    flags.DEFINE_boolean("white_background", False, "")
    flags.DEFINE_string("data_device", "cuda", "")
    flags.DEFINE_boolean("eval", False, "")

    flags.DEFINE_boolean("convert_SHs_python", False, "")
    flags.DEFINE_boolean("compute_cov3D_python", False, "")
    flags.DEFINE_boolean("debug", False, "")

    flags.DEFINE_integer("iterations", 30000, "") # 30000
    flags.DEFINE_float("position_lr_init", 0.00005, "") # 0.00016
    flags.DEFINE_float("position_lr_final", 0.0000016, "")
    flags.DEFINE_float("position_lr_delay_mult", 0.01, "")
    flags.DEFINE_integer("position_lr_max_steps", 30000, "") # 30000
    flags.DEFINE_float("feature_lr", 0.0025, "")
    flags.DEFINE_float("opacity_lr", 0.05, "")
    flags.DEFINE_float("scaling_lr", 0.005, "")
    flags.DEFINE_float("rotation_lr", 0.001, "")
    flags.DEFINE_float("regist_feat_lr", 0.0025, "")
    
    flags.DEFINE_float("percent_dense", 0.01, "")
    flags.DEFINE_float("lambda_dssim", 0, "")
    flags.DEFINE_integer("densification_interval", 100, "")
    flags.DEFINE_integer("opacity_reset_interval", 3000, "") # 3000
    flags.DEFINE_integer("outlier_filtering_interval", 2000, "") # 3000
    flags.DEFINE_integer("outlier_stop_iter", 29000, "") # 3000



    flags.DEFINE_integer("densify_from_iter", 500, "")
    flags.DEFINE_integer("densify_until_iter", 15000, "") # 15000
    flags.DEFINE_float("densify_grad_threshold", 0.0002, "")# 0.0002
    # flags.DEFINE_boolean("random_background", False, "")

    flags.DEFINE_string("gs_init_mesh", "", "init mesh")
    flags.DEFINE_string("gs_init_ply", "", "init ply")
    # flags.DEFINE_float("resolution_scale", 2, "training image resolition scale")
    
def get_config():
    return opts.flag_values_dict()


def save_config():
    save_dir = os.path.join(opts.logroot, "%s-%s" % (opts.seqname, opts.logname))
    os.makedirs(save_dir, exist_ok=True)
    opts_path = os.path.join(save_dir, "opts.log")
    if os.path.exists(opts_path):
        os.remove(opts_path)
    opts.append_flags_into_file(opts_path)


