# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import configparser
import glob
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from lab4d.utils.numpy_utils import pca_numpy


class FrameInfo:
    """Metadata about the frames in a dataset

    Args:
        ref_list (list(str)): List of paths to all filtered RGB frames in this video
    Attributes:
        num_frames (int): Number of frames after filtering out static frames.
        num_frames_raw (int): Total number of frames.
        frame_map (list(int)): Mapping from JPEGImages (filtered frames) to
          JPEGImagesRaw (all frames).
    """

    def __init__(self, ref_list):
        self.num_frames = len(ref_list)
        # need the raw frame index to apply Fourier time embedding
        raw_dir = ref_list[0].rsplit("/", 1)[0].replace("JPEGImages", "JPEGImagesRaw")
        self.num_frames_raw = len(glob.glob(raw_dir + "/*.jpg"))
        assert self.num_frames_raw > 0 
        self.frame_map = [int(path.split("/")[-1].split(".")[0]) for path in ref_list]


def train_loader(opts_dict):
    """Construct the training dataloader.

    Args:
        opts_dict (Dict): Defined in Trainer::construct_dataset_opts()
    Returns:
        dataloader (:class:`pytorch:torch.utils.data.DataLoader`): Training dataloader
    """
    # Set to 0 to debug the data loader
    num_workers = opts_dict["num_workers"]
    # num_workers = min(num_workers, 4)
    # num_workers = 0
    print("# workers: %d" % num_workers)
    print("# iterations per round: %d" % opts_dict["iters_per_round"])
    print(
        "# image samples per iteration: %d"
        % (opts_dict["imgs_per_gpu"] * opts_dict["ngpu"])
    )
    print("# pixel samples per image: %d" % opts_dict["pixels_per_image"])

    dataset = config_to_dataset(opts_dict)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=opts_dict["ngpu"],
        rank=opts_dict["local_rank"],
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=opts_dict["imgs_per_gpu"],
        num_workers=num_workers,
        drop_last=True,
        # worker_init_fn=_init_fn,
        pin_memory=True,
        sampler=sampler,
    )
    return dataloader


def eval_loader(opts_dict):
    """Construct the evaluation dataloader.

    Args:
        opts_dict (Dict): Defined in Trainer::construct_dataset_opts()
    Returns:
        dataloader (torch.utils.data.DataLoader): Evaluation dataloader
    """
    num_workers = 0

    dataset = config_to_dataset(opts_dict, is_eval=True)
    dataset = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
    )
    return dataset


def duplicate_dataset(opts, datalist):
    """Duplicate a list of per-video datasets, so that the length matches the
    desired number of iterations per round during training.

    Args:
        datalist (List(VidDataset)): A list of per-video datasets
    Returns:
        datalist_mul (List(VidDataset)): Duplicated dataset list
    """
    # import pdb;pdb.set_trace()
    for t in range(3):
        datalist.append(datalist[0])

    num_samp = np.sum([len(i) for i in datalist])
    if num_samp == 0:
        raise ValueError("Dataset is empty")
    # dup_num = opts["iters_per_round"] / (num_samp / opts["ngpu"] / opts["imgs_per_gpu"])
    dup_num = opts["iters_per_round"] * opts["ngpu"] * opts["imgs_per_gpu"]/ num_samp
    
    dup_num = int(dup_num) + 1

    datalist_mul = datalist * dup_num
    return datalist_mul


def config_to_dataset(opts, is_eval=False, gpuid=[]):
    """Construct a PyTorch dataset that includes all videos in a sequence.

    Args:
        opts (Dict): Defined in Trainer::construct_dataset_opts()
        is_eval (bool): Unused
        gpuid (List(int)): Select a subset based on gpuid for npy generation
    Returns:
        dataset (torch.utils.data.Dataset): Concatenation of datasets for each
            video in the sequence `opts["seqname"]`
    """
    config = configparser.RawConfigParser()
    config.read("database/configs/%s.config" % opts["seqname"])
    numvid = len(config.sections()) - 1
    datalist = []
    for vidid in range(numvid):
        dataset = section_to_dataset(opts, config, vidid)
        datalist.append(dataset)

    if opts["multiply"]:
        datalist = duplicate_dataset(opts, datalist)
    # select a subset based on gpuid (for npy generation)
    if len(gpuid) == 2:  # current id, total
        vid_per_gpu = int(np.ceil(len(datalist) / gpuid[1]))
        id_start = gpuid[0] * vid_per_gpu
        datalist = datalist[id_start : id_start + vid_per_gpu]
    dataset = torch.utils.data.ConcatDataset(datalist)
    return dataset


def section_to_dataset(opts, config, vidid):
    """Construct a PyTorch dataset for a single video in a sequence
    using `opts["dataset_constructor"]`

    Args:
        opts (Dict): Defined in Trainer::construct_dataset_opts()
        config (RawConfigParser): Config parser object
        vidid (int): Which video in the sequence
    Returns:
        dataset (torch.utils.data.Dataset): Dataset for the video
    """
    config_dict = load_config(config, "data")
    config_dict = load_config(config, "data_%d" % vidid, current_dict=config_dict)

    rgblist = sorted(glob.glob("%s/*.jpg" % config_dict["rgb_path"]))
    if config_dict["end_frame"] > -1:
        rgblist = rgblist[: config_dict["end_frame"]]
    if config_dict["init_frame"] > 0:
        rgblist = rgblist[config_dict["init_frame"] :]

    dataset = opts["dataset_constructor"](
        opts,
        rgblist=rgblist,
        dataid=vidid,
        ks=config_dict["ks"],
        raw_size=config_dict["raw_size"],
    )
    return dataset


def load_config(config, dataname, current_dict=None):
    """Load a section from a .config metadata file

    Args:
        config (RawConfigParser): Config parser object
        dataname (str): Name of section to load
        currect_dict (Dict): If given, load into an existing dict. Otherwise
            return a new dict
    """
    if current_dict is None:
        config_dict = {}
    else:
        config_dict = current_dict

    try:
        config_dict["rgb_path"] = str(config.get(dataname, "img_path"))
    except:
        pass

    try:
        config_dict["init_frame"] = int(config.get(dataname, "init_frame"))
    except:
        pass

    try:
        config_dict["end_frame"] = int(config.get(dataname, "end_frame"))
    except:
        pass

    try:
        config_dict["ks"] = [float(i) for i in config.get(dataname, "ks").split(" ")]
    except:
        pass

    try:
        config_dict["raw_size"] = [
            int(i) for i in config.get(dataname, "shape").split(" ")
        ]
    except:
        pass

    return config_dict


def get_data_info(loader):
    """Extract dataset metadata from a dataloader

    Args:
        loader (torch.utils.data.DataLoader): Evaluation dataloader
    Returns:
        data_info (Dict): Dataset metadata
    """
    data_info = {}
    dataset_list = loader.dataset.datasets
    frame_offset = [0]
    frame_offset_raw = [0]
    frame_mapping = []
    intrinsics = []
    raw_size = []
    feature_pxs = []

    for dataset in dataset_list:
        frame_info = FrameInfo(dataset.dict_list["ref"])
        frame_offset.append(frame_info.num_frames)
        frame_offset_raw.append(frame_info.num_frames_raw)
        frame_mapping += [
            i + np.sum(frame_offset_raw[:-1]) for i in frame_info.frame_map
        ]
        intrinsics += [dataset.ks] * frame_info.num_frames
        raw_size += [dataset.raw_size]

        feature_array = dataset.mmap_list["feature"].reshape(-1, 16)
        # # randomly sample 1k pixels for PCA per-video:
        # rand_idx = np.random.permutation(len(feature_array))[:1000]
        # feature_pxs.append(feature_array[rand_idx])
        # sampling a fixed set of pixels for PCA:
        num_skip = max(1, len(feature_array) // 1000)
        feature_pxs.append(feature_array[::num_skip])

    # compute PCA on non-zero features
    feature_pxs = np.concatenate(feature_pxs, 0)
    feature_pxs = feature_pxs[np.linalg.norm(feature_pxs, 2, -1) > 0]
    data_info["apply_pca_fn"] = pca_numpy(feature_pxs, n_components=3)

    frame_info = {}
    frame_info["frame_offset"] = np.asarray(frame_offset).cumsum()
    frame_info["frame_offset_raw"] = np.asarray(frame_offset_raw).cumsum()
    frame_info["frame_mapping"] = frame_mapping
    data_info["frame_info"] = frame_info

    data_info["total_frames"] = frame_info["frame_offset"][-1]
    data_info["intrinsics"] = np.asarray(intrinsics)  # N,4
    data_info["raw_size"] = np.asarray(raw_size)  # M,2

    data_path_dict = merge_dict_list(loader)
    data_info.update(load_small_files(data_path_dict))
    return data_info, data_path_dict


def merge_dict_list(loader):
    """For a sequence of videos, construct a dict .npy/.txt paths that contain
    all the frame data and annotations from the whole sequence

    Args:
        loader (torch.utils.data.DataLoader): Dataloader for a video sequence
    Returns:
        dict_list (Dict(str, List(str))): Maps each frame/annotation type to a
            list of .npy/.txt paths for that type
    """
    dataset_list = loader.dataset.datasets
    data_dict = {}
    for dataset in dataset_list:
        for k, path_list in dataset.dict_list.items():
            if k not in data_dict:
                data_dict[k] = []
            if isinstance(path_list, str):
                data_dict[k].append(path_list)
            else:
                data_dict[k] += path_list
    return data_dict


def load_small_files(data_path_dict):
    """For a sequence of videos, load small dataset files into memory

    Args:
        data_path_dict (Dict(str, List(str))): Maps each annotation type to a
            list of .npy/.txt paths for that type
    Returns:
        data_info (Dict): Dataset metadata
    """
    data_info = {}
    # data_info["crop2raw"] = np.concatenate(
    #     [np.load(path).astype(np.float32) for path in data_path_dict["crop2raw"]], 0
    # )  # N,4

    rtmat_bg = np.concatenate(
        [np.load(path).astype(np.float32) for path in data_path_dict["cambg"]], 0
    )  # N,4,4
    rtmat_fg = np.concatenate(
        [np.load(path).astype(np.float32) for path in data_path_dict["camfg"]], 0
    )  # N,4,4
    # hard-code for now
    vis_info = {"bg": 0, "fg": 1}  # video instance segmentation info
    data_info["vis_info"] = vis_info
    data_info["rtmat"] = np.stack([rtmat_bg, rtmat_fg], 0)

    # path to centered mesh files
    camera_prefix = data_path_dict["cambg"][0].rsplit("/", 1)[0]
    data_info["geom_path"] = [
        "%s/mesh-00-centered.obj" % camera_prefix,
        "%s/mesh-01-centered.obj" % camera_prefix,
    ]
    return data_info


def get_vid_length(inst_id, data_info):
    """Compute the length of a video

    Args:
        inst_id (int): Video to check
        data_info (Dict): Dataset metadata
    """
    frame_offset_raw = data_info["frame_info"]["frame_offset_raw"]
    vid_length = frame_offset_raw[1:] - frame_offset_raw[:-1]
    vid_length = vid_length[inst_id]
    return vid_length
