# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os
import subprocess
import sys

import cv2
import numpy as np

from lab4d.utils.profile_utils import record_function

sys.path.insert(
    0,
    "%s/../third_party/vcnplus/" % os.path.join(os.path.dirname(__file__)),
)
sys.path.insert(
    0,
    "%s/../../" % os.path.join(os.path.dirname(__file__)),
)
from flowutils.flowlib import warp_flow
from libs.utils import reduce_component

from lab4d.utils.geom_utils import K2mat, compute_crop_params


def run_bash_command(cmd):
    # print(cmd)
    subprocess.run(cmd, shell=True, check=True)


@record_function("read_images_densepose")
def read_images_densepose(imglist):
    # load images
    rgbs = []
    masks = []
    for imgpath in imglist:
        # print(imgpath)
        # # rgb: (s, s, 3), 0-1
        # rgb, _, mask, _ = read_frame_data(imgpath, crop_size, use_full, component_id)
        # rgb = (rgb * 255).astype(np.uint8)[..., ::-1].copy()  # to BGR
        # pdb.set_trace()
        # mask = mask.astype(np.uint8)
        # h, w, _ = rgb.shape

        # crop without resizing
        rgb = cv2.imread(imgpath)
        mask = np.load(
            imgpath.replace("JPEGImages", "Annotations").replace(".jpg", ".npy")
        )
        mask = (mask > 0).astype(float)
        if mask.max() == 0:
            mask[:] = 1

        rgbs.append(rgb)
        masks.append(mask)
    return rgbs, masks


@record_function("read_frame_data")
def read_frame_data(imgpath, crop_size, use_full, component_id, with_flow=True):
    # compute intrincs for the cropped images
    data_dict0 = read_raw(imgpath, 1, crop_size, use_full, with_flow=with_flow)
    depth = data_dict0["depth"]
    rgb = data_dict0["img"]
    mask = data_dict0["mask"][..., 0].astype(int) == component_id
    if component_id > 0:
        # reduce the mask to the largest connected component
        mask = reduce_component(mask)
    return rgb, depth, mask, data_dict0["crop2raw"]


@record_function("read_mask")
def read_mask(mask_path, shape):
    mask = np.load(mask_path)
    if mask.shape[0] != shape[0] or mask.shape[1] != shape[1]:
        mask = cv2.resize(mask, shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    mask = np.expand_dims(mask, -1)

    if mask.min() < 0:
        is_detected = False
    else:
        is_detected = True
    mask = (mask > 0).astype(int)
    vis2d = np.ones_like(mask).astype(int)
    return mask, vis2d, is_detected


@record_function("read_flow")
def read_flow(flow_path, shape):
    flow = np.load(flow_path)
    flow = flow.astype(np.float32)
    occ = flow[..., 2:]
    flow = flow[..., :2]
    h, w, _ = shape
    oh, ow = flow.shape[:2]
    factor_h = h / oh
    factor_w = w / ow
    flow = cv2.resize(flow, (w, h))
    occ = cv2.resize(occ, (w, h))
    flow[..., 0] *= factor_w
    flow[..., 1] *= factor_h
    return flow, occ


@record_function("read_depth")
def read_depth(depth_path, shape):
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape[0] != shape[0] or depth.shape[1] != shape[1]:
        depth = cv2.resize(depth, shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    return depth


@record_function("read_raw")
def read_raw(img_path, delta, crop_size, use_full, with_flow=True):
    img = cv2.imread(img_path)[..., ::-1] / 255.0
    shape = img.shape
    mask_path = img_path.replace("JPEGImages", "Annotations").replace(".jpg", ".npy")
    mask, vis2d, is_detected = read_mask(mask_path, shape)
    if not is_detected:  # force using full if there is no detection
        use_full = True
    crop2raw = compute_crop_params(mask, crop_size=crop_size, use_full=use_full)
    depth_path = img_path.replace("JPEGImages", "Depth").replace(".jpg", ".npy")
    depth = read_depth(depth_path, shape)

    is_fw = delta > 0
    delta = abs(delta)
    if is_fw:
        flowpath = img_path.replace("JPEGImages", "FlowFW_%d" % (delta)).replace(
            ".jpg", ".npy"
        )
    else:
        flowpath = img_path.replace("JPEGImages", "FlowBW_%d" % (delta)).replace(
            ".jpg", ".npy"
        )
    if with_flow:
        flow, occ = read_flow(flowpath, shape)

    # crop the image according to mask
    x0, y0 = np.meshgrid(range(crop_size), range(crop_size))
    hp_crop = np.stack([x0, y0, np.ones_like(x0)], -1)  # augmented coord
    hp_crop = hp_crop.astype(np.float32)
    hp_raw = hp_crop @ K2mat(crop2raw).T  # raw image coord
    x0 = hp_raw[..., 0].astype(np.float32)
    y0 = hp_raw[..., 1].astype(np.float32)
    img = cv2.remap(img, x0, y0, interpolation=cv2.INTER_LINEAR)
    mask = cv2.remap(mask, x0, y0, interpolation=cv2.INTER_NEAREST)
    vis2d = cv2.remap(vis2d, x0, y0, interpolation=cv2.INTER_NEAREST)
    if with_flow:
        flow = cv2.remap(flow, x0, y0, interpolation=cv2.INTER_LINEAR)
        occ = cv2.remap(occ, x0, y0, interpolation=cv2.INTER_LINEAR)
    depth = cv2.remap(depth, x0, y0, interpolation=cv2.INTER_LINEAR)
    # print('crop:%f'%(time.time()-ss))

    data_dict = {}
    data_dict["img"] = img.astype(np.float16)
    data_dict["mask"] = np.stack([mask, vis2d], -1).astype(bool)
    if with_flow:
        data_dict["flow"] = flow
        data_dict["occ"] = occ
    data_dict["depth"] = depth.astype(np.float16)
    data_dict["crop2raw"] = crop2raw
    data_dict["hxy"] = hp_crop
    data_dict["hp_raw"] = hp_raw
    data_dict["is_detected"] = is_detected
    return data_dict


def get_bbox(img_path, component_id):
    """
    [x0, y0, w, h]
    """
    img = cv2.imread(img_path)[..., ::-1] / 255.0
    shape = img.shape
    mask_path = img_path.replace("JPEGImages", "Annotations").replace(".jpg", ".npy")
    mask, _, _ = read_mask(mask_path, shape)
    mask = mask == component_id
    if mask.max() == 0:
        return None
    indices = np.where(mask > 0)
    xid = indices[1]
    yid = indices[0]
    x0, y0 = (xid.min(), yid.min())
    w, h = ((xid.max() - xid.min()), (yid.max() - yid.min()))
    bbox = np.asarray([x0, y0, w, h])
    return bbox


@record_function("compute_flow_uct")
def compute_flow_uct(occ, flow0, hp1, hp0):
    """
    hp1: homogeneous coord displaced by 2nd frame flow backwards
    """
    # cycle uncertainty: distance = ||disp_bw(disp_fw(x,y)) - (x,y)||
    img_size = occ.shape[0]
    dis = warp_flow(hp1[:, :, :2], flow0) - hp0
    dis = np.linalg.norm(dis[:, :, :2], 2, -1)
    dis_norm = dis / img_size * 2
    flow_uct = np.exp(-25 * dis_norm)
    flow_uct[flow_uct < 0.25] = 0.0  # this corresps to 1/40 img size
    flow_uct[occ > 0] = 0  # predictive uncertainty
    return flow_uct


@record_function("flow_process")
def flow_process(data_dict0, data_dict1):
    """
    convert flow to cropped coordinate
    compute uncertainty
    normalize flow
    """
    flow0, flow1, occ0, occ1, hp_raw0, hp_raw1 = (
        data_dict0["flow"],
        data_dict1["flow"],
        data_dict0["occ"],
        data_dict1["occ"],
        data_dict0["hp_raw"],
        data_dict1["hp_raw"],
    )
    hp = data_dict0["hxy"][:, :, :2]
    ones = np.ones_like(hp[..., :1])
    crop2raw0 = K2mat(data_dict0["crop2raw"])
    crop2raw1 = K2mat(data_dict1["crop2raw"])

    # flow in the cropped coordinate
    hp_raw1c = np.concatenate([flow0 + hp_raw0[:, :, :2], ones], -1)
    hp_crop1 = hp_raw1c @ np.linalg.inv(crop2raw1).T
    flow0_crop = hp_crop1[:, :, :2] - hp

    hp_raw0c = np.concatenate([flow1 + hp_raw1[:, :, :2], ones], -1)
    hp_crop0 = hp_raw0c.dot(np.linalg.inv(crop2raw0.T))
    flow1_crop = hp_crop0[:, :, :2] - hp

    # fb check
    flow_uct0 = compute_flow_uct(occ0, flow0_crop, hp_crop0, hp)
    flow_uct1 = compute_flow_uct(occ1, flow1_crop, hp_crop1, hp)

    (
        data_dict0["flow"],
        data_dict1["flow"],
    ) = (
        np.concatenate([flow0_crop, flow_uct0[..., None]], -1),
        np.concatenate([flow1_crop, flow_uct1[..., None]], -1),
    )
    data_dict0["flow"] = data_dict0["flow"].astype(np.float16)
    data_dict1["flow"] = data_dict1["flow"].astype(np.float16)
    return
