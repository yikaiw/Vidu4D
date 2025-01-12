# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import sys
import os

# insert path of current file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    "%s/../../" % os.path.join(os.path.dirname(__file__)),
)

import cv2
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import glob
from models.VCNplus import VCN
from models.inference import (
    load_eval_checkpoint,
    modify_flow_module,
    process_flow_input,
    make_disc_aux,
)
from flowutils.flowlib import point_vec, warp_flow

from libs.utils import resize_to_target

cudnn.benchmark = True


def compute_flow(seqname, outdir, dframe):
    model_path = "./preprocess/third_party/vcnplus/vcn_rob.pth"
    maxdisp = 256  # maxium disparity. Only affect the coarsest cost volume size
    fac = (
        1  # controls the shape of search grid. Only affect the coarse cost volume size
    )

    # construct model
    model = load_eval_checkpoint(model_path, maxdisp=maxdisp, fac=fac)

    fw_path = "%s/FlowFW_%d/Full-Resolution/%s/" % (outdir, dframe, seqname)
    bw_path = "%s/FlowBW_%d/Full-Resolution/%s/" % (outdir, dframe, seqname)
    os.system("mkdir -p %s" % (fw_path))
    os.system("mkdir -p %s" % (bw_path))

    img_paths = sorted(
        glob.glob("%s/JPEGImages/Full-Resolution/%s/*.jpg" % (outdir, seqname))
    )

    # load image 0 and compute resize ratio
    img0_path = img_paths[0]
    img0_o = cv2.imread(img0_path)[:, :, ::-1]

    input_size = img0_o.shape
    inp_h, inp_w, _ = input_size
    max_res = 2e6
    res_fac = np.sqrt(max_res / (inp_h * inp_w))
    max_h = int(np.ceil(inp_h * res_fac / 64) * 64)
    max_w = int(np.ceil(inp_w * res_fac / 64) * 64)

    # modify flow module according to input size
    modify_flow_module(model, max_h, max_w)
    model.eval()

    # compute forward and backward flow
    img0, img0_noaug = process_flow_input(img0_o, model.mean_L, max_h, max_w)
    for jnx in range(dframe, len(img_paths), dframe):
        # print("%s/%s" % (img_paths[inx], img_paths[jnx]))
        img1_path = img_paths[jnx]
        img1_o = cv2.imread(img1_path)[:, :, ::-1]
        img1, img1_noaug = process_flow_input(img1_o, model.mean_R, max_h, max_w)

        # forward and backward inference
        disc_aux0 = make_disc_aux(img0_noaug, max_h, max_w, input_size)
        disc_aux1 = make_disc_aux(img1_noaug, max_h, max_w, input_size)
        with torch.no_grad():
            img01 = torch.cat([img0, img1], dim=0)
            flowfw, occfw, _, _ = model(img01, disc_aux0)
            img10 = torch.cat([img1, img0], dim=0)
            flowbw, occbw, _, _ = model(img10, disc_aux1)

        # flow: (1, 2, max_h, max_w) => (max_h, max_w, 2)
        # occ: (max_h // 4, max_w // 4)
        flowfw_np = torch.moveaxis(flowfw[0], 0, -1).cpu().numpy()
        flowbw_np = torch.moveaxis(flowbw[0], 0, -1).cpu().numpy()
        occfw_np = occfw.cpu().numpy()
        occbw_np = occbw.cpu().numpy()

        # downsample first
        flowfw_ds = resize_to_target(flowfw_np, aspect_ratio=input_size, is_flow=True)
        flowbw_ds = resize_to_target(flowbw_np, aspect_ratio=input_size, is_flow=True)
        occfw_ds = resize_to_target(occfw_np, aspect_ratio=input_size, is_flow=False)
        occbw_ds = resize_to_target(occbw_np, aspect_ratio=input_size, is_flow=False)
        img0_ds = resize_to_target(img0_o, aspect_ratio=input_size, is_flow=False)
        img1_ds = resize_to_target(img1_o, aspect_ratio=input_size, is_flow=False)

        flowfw_ds = np.concatenate(
            [flowfw_ds, occfw_ds[:, :, None]], -1
        )  # ds_h, ds_w, 3
        flowbw_ds = np.concatenate(
            [flowbw_ds, occbw_ds[:, :, None]], -1
        )  # ds_h, ds_w, 3

        # save predictions
        frameid0 = int(img0_path.split("/")[-1].split(".")[0])
        frameid1 = int(img1_path.split("/")[-1].split(".")[0])
        np.save("%s/%05d.npy" % (fw_path, frameid0), flowfw_ds.astype(np.float16))
        np.save("%s/%05d.npy" % (bw_path, frameid1), flowbw_ds.astype(np.float16))

        # resize from (max_h, max_w, 2) => (inp_h, inp_w, 2) for visualization
        flowfw_vis = np.concatenate(
            [
                cv2.resize(flowfw_np[:, :, 0], (inp_w, inp_h))[:, :, None],
                cv2.resize(flowfw_np[:, :, 1], (inp_w, inp_h))[:, :, None],
            ],
            -1,
        )
        flowfw_vis[:, :, 0] *= inp_w / max_w
        flowfw_vis[:, :, 1] *= inp_h / max_h

        flowbw_vis = np.concatenate(
            [
                cv2.resize(flowbw_np[:, :, 0], (inp_w, inp_h))[:, :, None],
                cv2.resize(flowbw_np[:, :, 1], (inp_w, inp_h))[:, :, None],
            ],
            -1,
        )
        flowbw_vis[:, :, 0] *= inp_w / max_w
        flowbw_vis[:, :, 1] *= inp_h / max_h

        imwarped = warp_flow(img1_o, flowfw_vis[:, :, :2])
        cv2.imwrite(
            "%s/warp-%05d.jpg" % (fw_path, frameid0),
            imwarped[:, :, ::-1],
        )
        imwarped = warp_flow(img0_o, flowbw_vis[:, :, :2])
        cv2.imwrite(
            "%s/warp-%05d.jpg" % (bw_path, frameid1),
            imwarped[:, :, ::-1],
        )

        # visualize semi-dense flow for forward
        # x0, y0 = np.meshgrid(range(flowfw.shape[1]), range(flowfw.shape[0]))
        # hp0 = np.stack([x0, y0], -1)
        # dis = warp_flow(hp0 + flowbw[..., :2], flowfw[..., :2]) - hp0
        # dis = np.linalg.norm(dis[:, :, :2], 2, -1)
        # dis = dis / np.sqrt(flowfw.shape[0] * flowfw.shape[1]) * 2
        # fb_mask = np.exp(-25 * dis) > 0.8

        # flowvis = flowfw.copy()
        # flowvis[~fb_mask] = 0
        flowvis = point_vec(img0_ds, flowfw_ds, skip=10)
        cv2.imwrite("%s/visflo-%05d.jpg" % (fw_path, frameid0), flowvis)

        flowvis = point_vec(img1_ds, flowbw_ds, skip=10)
        cv2.imwrite("%s/visflo-%05d.jpg" % (bw_path, frameid1), flowvis)

        torch.cuda.empty_cache()

        img0_path = img1_path
        img0_o = img1_o
        img0, img0_noaug = img1, img1_noaug

    print("compute flow (skip=%d) done: %s" % (dframe, seqname))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <seqname> <outdir> <dframe>")
        print(
            f"  Example: python {sys.argv[0]} cat-pikachu-0-0000 'database/processed/' 1"
        )
        exit()
    seqname = sys.argv[1]
    outdir = sys.argv[2]
    dframe = int(sys.argv[3])
    compute_flow(seqname, outdir, dframe)
