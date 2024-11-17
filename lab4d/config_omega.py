# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
from omegaconf import DictConfig, OmegaConf

# Define the hierarchical configuration using a dictionary
config = DictConfig(
    {
        "train": {
            "weights": {
                "recon": {
                    "mask_wt": 0.1,  # weight for silhouette loss
                    "rgb_wt": 0.1,  # weight for color loss
                    "depth_wt": 0.01,  # weight for depth loss
                    "flow_wt": 0.5,  # weight for flow loss
                    "vis_wt": 0.01,  # weight for visibility loss
                    "feature_wt": 0.01,  # weight for feature reconstruction loss
                    "feat_reproj_wt": 0.05,  # weight for feature reprojection loss
                },
                "reg": {
                    "visibility_wt": 1e-3,  # weight for visibility regularization
                    "eikonal_wt": 1e-5,  # weight for eikonal regularization
                    "deform_cyc_wt": 0.01,  # weight for deform cyc regularization
                    "gauss_skin_wt": 1,  # weight for gauss skinning consistency
                },
            },
            "model": {
                "field_type": "bg",  # {bg, fg, comp}
                "fg_motion": "rigid",  # {rigid, dense, bob, skel}
                "single_inst": True,  # assume the same morphology over objs
            },
            "io": {
                "seqname": "cat",  # name of the sequence
                "logname": "tmp",  # name of the saved log
                "data_prefix": "full",  # prefix of the data entries
                "train_res": 256,  # size of training images
                "logroot": "logdir/",  # root directory for log files
                "load_suffix": "",  # sufix of params, {latest, 0, 10, ...}
                "save_freq": 10,  # params saving frequency
            },
            "optim": {
                "learning_rate": 5e-4,  # learning rate
                "num_rounds": 20,  # number of rounds to trainn
                "iters_per_round": 200,  # number of iterations per round
                "imgs_per_gpu": 128,  # images samples per iter, per gpu
                "pixels_per_image": 16,  # pixel samples per image
                "ngpu": 1,  # number of gpus to use
                "num_workers": 2,  # number of workers for dataloading
            },
            "eval_res": 64,  # size used for eval visualizations
            "profile": False,  # profile the training loop
        },
    }
)


def get_config():
    return opts.flag_values_dict()


def save_config():
    save_dir = os.path.join(opts.logroot, opts.logname)
    os.makedirs(save_dir, exist_ok=True)
    opts_path = os.path.join(save_dir, "opts.log")
    if os.path.exists(opts_path):
        os.remove(opts_path)
    opts.append_flags_into_file(opts_path)


# # Convert the configuration to a dictionary
# config_dict = OmegaConf.to_container(config)

# # Convert the dictionary back to a configuration
# config2 = OmegaConf.create(config_dict)
