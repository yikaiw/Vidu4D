# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
from math import e
import os
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from lab4d.utils.numpy_utils import bilinear_interp
from lab4d.dataloader.data_utils import FrameInfo


class RangeSampler:
    """Sample efficiently without replacement from the range [0, num_elems).

    Args:
        num_elems (int): Upper bound of sample range
    """

    def __init__(self, num_elems):
        self.num_elems = num_elems
        self.init_queue()

    def init_queue(self):
        """Compute the next set of samples by permuting the sample range"""
        self.sample_queue = np.random.permutation(self.num_elems)
        self.curr_idx = 0

    def sample(self, num_samples):
        """Return a set of samples from [0, num_elems) without replacement.

        Args:
            num_samples (int): Number of samples to return
        Returns:
            rand_idx: (num_samples,) Output samples
        """
        # Recompute samples if not enough values
        if self.curr_idx + num_samples > self.num_elems:
            self.init_queue()

        rand_idx = self.sample_queue[self.curr_idx : self.curr_idx + num_samples]
        self.curr_idx += num_samples
        return rand_idx


class VidDataset(Dataset):
    """Frame data and annotations for a single video in a sequence.
    Uses np.mmap internally to load larger-than-memory frame data from disk.

    Args:
        opts (Dict): Defined in Trainer::construct_dataset_opts()
        rgblist (List(str)): List of paths to all RGB frames in this video
        dataid (int): Video ID
        ks (List(int)): Camera intrinsics: [fx, fy, cx, cy]
        raw_size (List(int)): Shape of the raw frames, [H, W]
    """

    def __init__(self, opts, rgblist, dataid, ks, raw_size):
        self.delta_list = opts["delta_list"]
        self.dict_list = self.construct_data_list(
            rgblist, opts["data_prefix"], opts["feature_type"]
        )
        self.pixels_per_image = opts["pixels_per_image"]
        self.dataid = dataid
        self.load_pair = opts["load_pair"]
        self.ks = ks
        self.raw_size = raw_size
        self.img_size = np.load(self.dict_list["rgb"]).shape[1:3]  # (H, W)
        self.load_data_list(self.dict_list)

        self.idx_sampler = RangeSampler(num_elems=self.img_size[0] * self.img_size[1])
        self.frame_info = FrameInfo(self.dict_list["ref"])

        if "quant_exp" in opts:
            self.quant_exp = opts["quant_exp"]
        else:
            self.quant_exp = False

    def construct_data_list(self, reflist, prefix, feature_type):
        """Construct a dict of .npy/.txt paths that contain all the frame data
        and annotations for a particular video

        Args:
            reflist (List(str)): List of paths to all RGB frames in the video
            prefix (str): Type of data to load ("crop-256" or "full-256")
            feature_type (str): Type of image features to use ("cse" or "dino")
        Returns:
            dict_list (Dict(str, List(str))): Maps each frame/annotation type
                to a list of .npy/.txt paths for that type
        """
        rgb_path = reflist[0].replace("00000.jpg", "%s.npy" % prefix)
        mask_path = rgb_path.replace("JPEGImages", "Annotations")
        flowfw_path = rgb_path.replace("JPEGImages", "FlowFW")
        flowbw_path = rgb_path.replace("JPEGImages", "FlowBW")
        depth_path = rgb_path.replace("JPEGImages", "Depth")
        feature_path = str(
            Path(rgb_path.replace("JPEGImages", "Features")).parent
        ) + "/%s-%s-01.npy" % (prefix, feature_type)

        camlist_bg = (
            reflist[0].replace("JPEGImages", "Cameras").replace("00000.jpg", "00.npy")
        )  # bg
        camlist_fg = (
            reflist[0]
            .replace("JPEGImages", "Cameras")
            .replace("00000.jpg", "01-canonical.npy")
        )  # fg

        # TODO load cams directly
        # TODO do not need to return crop2raw from data loader
        crop2raw_path = mask_path.replace(".npy", "-crop2raw.npy")
        is_detected_path = mask_path.replace(".npy", "-is_detected.npy")

        return {
            "ref": reflist,
            "rgb": rgb_path,
            "mask": mask_path,
            "cambg": camlist_bg,
            "camfg": camlist_fg,
            "flowfw": flowfw_path,
            "flowbw": flowbw_path,
            "depth": depth_path,
            "feature": feature_path,
            "crop2raw": crop2raw_path,
            "is_detected": is_detected_path,
        }

    def load_data_list(self, dict_list):
        """Load all the frame data and anotations in this dataset

        Args:
            dict_list (Dict(str, List(str))): From `construct_data_list()`
        Returns:
            mmap_list (Dict): Maps each key to a numpy array of frame data or
                a list of annotations
        """
        # load crop2raw
        self.crop2raw = np.load(dict_list["crop2raw"])
        self.is_detected = np.load(dict_list["is_detected"])

        # load all .npy files using mmap
        # The number of open files is bounded by `ulimit -S -n` and `ulimit -H -n`,
        # both of which could be easily exceeded by many videos.
        self.mmap_list = {}
        for k, path in dict_list.items():
            if k in ("ref", "cambg", "camfg", "crop2raw", "is_detected"):
                continue

            if k in ("flowfw", "flowbw"):
                self.mmap_list[k] = {}
                for delta in [1] + self.delta_list:
                    path_delta = path.replace("FlowFW", f"FlowFW_{delta}").replace(
                        "FlowBW", f"FlowBW_{delta}"
                    )
                    if os.path.exists(path_delta):
                        self.mmap_list[k][delta] = np.load(path_delta, mmap_mode="r")
                continue

            try:
                self.mmap_list[k] = np.load(path, mmap_mode="r")
            except:
                print(f"Warning: cannot load {path}")
                self.mmap_list[k] = np.random.rand(self.__len__() + 1, 112, 112, 16)

    def __len__(self):
        if self.quant_exp:
            return len(self.dict_list["ref"]) // 4 - 2
        else:
            return len(self.dict_list["ref"]) - 1

    def __getitem__(self, index):
        if self.quant_exp:
            index *= 4
        data_dict = self.load_data(index)
        return data_dict

    def sample_delta(self, index):
        """Sample random delta frame

        Args:
            index (int): First index in the pair
        Returns:
            delta (int): Delta between first and second index
        """
        delta_list = [1] + [
            delta
            for delta in self.delta_list
            if (index % delta == 0) and int(index + delta) < len(self.dict_list["ref"])
        ]
        if self.quant_exp:
            delta_list = [4 ,8]
        delta = np.random.choice(delta_list)
        return delta

    def sample_xy(self):
        """Sample random pixels from an image

        Returns:
            xy: (N, 2) Sampled pixels
        """
        if self.pixels_per_image == -1:
            return None

        rand_idx = self.idx_sampler.sample(num_samples=self.pixels_per_image)
        y0 = rand_idx % self.img_size[0]
        x0 = rand_idx // self.img_size[0]
        xy = np.stack([x0, y0], axis=-1)  # (num_sample, 2)
        return xy

    def load_data(self, im0idx):
        """Sample pixels from a pair of frames

        Args:
            im0idx (int): First frame id in the pair
        Returns:
            data_dict (Dict): Maps keys to (2, ...) data
        """
        # im0idx = 0
        # delta = 1
        delta = self.sample_delta(im0idx)
        im1idx = im0idx + delta

        rand_xy0 = self.sample_xy()
        rand_xy1 = self.sample_xy()

        data_dict0 = self.read_raw(im0idx, delta, rand_xy=rand_xy0)

        if self.load_pair:
            data_dict1 = self.read_raw(im1idx, -delta, rand_xy=rand_xy1)

            for k in data_dict0.keys():
                data_dict0[k] = np.stack([data_dict0[k], data_dict1[k]])
        return data_dict0

    def read_raw(self, im0idx, delta, rand_xy=None):
        """Read video data for a single frame within a pair

        Args:
            im0idx (int): Frame id to load
            delta (int): Distance to other frame id in the pair
            rand_xy (array or None): (N, 2) pixels to load, if given
        Returns:
            data_dict (Dict): Dict with keys "rgb", "mask", "depth", "feature",
                "flow", "vis2d", "crop2raw", "dataid", "frameid_sub", "hxy"
        """
        rgb = self.read_rgb(im0idx, rand_xy=rand_xy)
        mask, vis2d, crop2raw, is_detected = self.read_mask(im0idx, rand_xy=rand_xy)
        depth = self.read_depth(im0idx, rand_xy=rand_xy)
        flow = self.read_flow(im0idx, delta, rand_xy=rand_xy)
        feature = self.read_feature(im0idx, rand_xy=rand_xy)

        if rand_xy is None:
            x0, y0 = np.meshgrid(range(self.img_size[1]), range(self.img_size[0]))
            hp_crop = np.stack([x0, y0, np.ones_like(x0)], axis=-1)
        else:
            hp_crop = np.concatenate([rand_xy, np.ones_like(rand_xy[..., :1])], -1)
        hp_crop = hp_crop.astype(np.float32)

        raw_frameid_sub = self.frame_info.frame_map[im0idx]

        data_dict = {}
        data_dict["rgb"] = rgb
        data_dict["mask"] = mask
        data_dict["depth"] = depth
        data_dict["feature"] = feature
        data_dict["flow"] = flow[..., :2]
        data_dict["flow_uct"] = flow[..., 2:]
        data_dict["vis2d"] = vis2d
        data_dict["crop2raw"] = crop2raw
        data_dict["is_detected"] = is_detected
        data_dict["dataid"] = self.dataid
        data_dict["frameid_sub"] = raw_frameid_sub  # frameid in each video
        data_dict["hxy"] = hp_crop
        return data_dict

    def read_rgb(self, im0idx, rand_xy=None):
        """Read RGB data for a single frame

        Args:
            im0idx (int): Frame id to load
            rand_xy (np.array or None): (N, 2) Pixels to load, if given
        Returns:
            rgb (np.array): (H,W,3) or (N, 3) Pixels, 0 to 1, float16
        """
        rgb = self.mmap_list["rgb"][im0idx]
        shape = rgb.shape
        if rand_xy is not None:
            rgb = rgb[rand_xy[:, 1], rand_xy[:, 0]]  # N,3

        if len(shape) == 2:  # gray image
            rgb = np.repeat(np.expand_dims(rgb, -1), 3, axis=-1)
        return rgb

    def read_mask(self, im0idx, rand_xy=None):
        """Read segmentation and object-centric bounding box for a single frame

        Args:
            im0idx (int): Frame id to load
            rand_xy (np.array or None): (N,2) Pixels to load, if given
        Returns:
            mask (np.array): (H,W,1) or (N,1) Segmentation mask, bool
            vis2d (np.array): (H,W,1) or (N,1) Mask of whether each
                pixel is part of the original frame, bool. For full frames,
                the entire mask is True
            crop2raw (np.array): (4,) Camera-intrinsics-style transformation
                from cropped (H,W) image to raw image, (fx, fy, cx, cy)
        """
        mask = self.mmap_list["mask"][im0idx]
        if rand_xy is not None:
            mask = mask[rand_xy[:, 1], rand_xy[:, 0]]  # N,3

        vis2d = mask[..., 1:]
        mask = mask[..., :1]

        crop2raw = self.crop2raw[im0idx]
        is_detected = self.is_detected[im0idx]
        return mask, vis2d, crop2raw, is_detected

    def read_depth(self, im0idx, rand_xy=None):
        """Read depth map for a single frame

        Args:
            im0idx (int): Frame id to load
            rand_xy (np.array or None): (N,2) Pixels to load, if given
        Returns:
            depth (np.array): (H,W,1) or (N,1) Depth map, float16
        """
        depth = self.mmap_list["depth"][im0idx]
        if rand_xy is not None:
            depth = depth[rand_xy[:, 1], rand_xy[:, 0]]

        return depth[..., None]

    def read_feature(self, im0idx, rand_xy=None):
        """Read feature map for a single frame

        Args:
            im0idx (int): Frame id to load
            rand_xy (np.array or None): (N,2) Pixels to load, if given
        Returns:
            feat (np.array): (112,112,16) or (N,16) Feature map, float32
        """
        feat = self.mmap_list["feature"][im0idx]  # (112,112,16)
        if rand_xy is not None:
            rand_xy = rand_xy / self.img_size[0] * 112
            feat = bilinear_interp(feat, rand_xy)
        feat = feat.astype(np.float32)
        return feat

    def read_flow(self, im0idx, delta, rand_xy=None):
        """Read flow map for a single frame

        Args:
            im0idx (int): Frame id of flow source
            delta (int): Number of frames from flow source to flow target
            rand_xy (np.array or None): (N,2) Pixels to load, if given
        Returns:
            flow (np.array): (H,W,3) or (N,3) Dense flow map, float32
        """
        is_fw = delta > 0
        delta = abs(delta)
        if is_fw:
            flow = self.mmap_list["flowfw"][delta][im0idx // delta]
        else:
            flow = self.mmap_list["flowbw"][delta][im0idx // delta - 1]
        if rand_xy is not None:
            flow = flow[rand_xy[:, 1], rand_xy[:, 0]]

        flow = flow.astype(np.float32)
        return flow
