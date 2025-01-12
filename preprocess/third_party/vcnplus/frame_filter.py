# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import sys
import os

# insert path of current file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import pdb
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import glob
import shutil

from models.VCNplus import VCN
from models.inference import (
    load_eval_checkpoint,
    modify_flow_module,
    process_flow_input,
    make_disc_aux,
)

cudnn.benchmark = True


def frame_filter(seqname, outdir):
    print("Filtering frames for %s" % (seqname))
    model_path = "./preprocess/third_party/vcnplus/vcn_rob.pth"
    maxdisp = 256  # maxium disparity. Only affect the coarsest cost volume size
    fac = (
        1  # controls the shape of search grid. Only affect the coarse cost volume size
    )
    flow_threshold = 0.05  # flow threshold that controls frame skipping
    max_frames = 500  # maximum number of frames to keep (to avoid oom in tracking etc.)

    # construct model
    model = load_eval_checkpoint(model_path, maxdisp=maxdisp, fac=fac)

    # input and output images
    img_paths = sorted(
        glob.glob("%s/JPEGImagesRaw/Full-Resolution/%s/*.jpg" % (outdir, seqname))
    )
    output_path = "%s/JPEGImages/Full-Resolution/%s/" % (outdir, seqname)
    output_idxs = []

    # load image 0 and compute resize ratio
    img0_o = cv2.imread(img_paths[0])[:, :, ::-1]
    output_idxs.append(0)

    input_size = img0_o.shape
    inp_h, inp_w, _ = input_size
    max_res = 300 * 300
    res_fac = np.sqrt(max_res / (inp_h * inp_w))
    max_h = int(np.ceil(inp_h * res_fac / 64) * 64)
    max_w = int(np.ceil(inp_w * res_fac / 64) * 64)

    # modify flow module according to input size
    modify_flow_module(model, max_h, max_w)
    model.eval()

    # find adjacent frames with sufficiently large flow
    img0, img0_noaug = process_flow_input(img0_o, model.mean_L, max_h, max_w)
    for jnx in range(1, len(img_paths)):
        img1_o = cv2.imread(img_paths[jnx])[:, :, ::-1]
        img1, img1_noaug = process_flow_input(img1_o, model.mean_R, max_h, max_w)

        # forward inference
        disc_aux = make_disc_aux(img0_noaug, max_h, max_w, input_size)
        with torch.no_grad():
            img01 = torch.cat([img0, img1], dim=0)
            flowfw, _, _, _ = model(img01, disc_aux)  # 1, 2, max_h, max_w

        flowfw[:, 0:1] /= max_w
        flowfw[:, 1:2] /= max_h

        maxflow = torch.max(torch.norm(flowfw[0], p=2, dim=0)).item()
        # print(jnx, "%.06f" % (maxflow))

        if maxflow > flow_threshold:
            output_idxs.append(jnx)
            img0_o = img1_o
            img0, img0_noaug = process_flow_input(img0_o, model.mean_L, max_h, max_w)

        if len(output_idxs) >= max_frames:
            break

    # copy selected frames to output
    if len(output_idxs) > 8:
        os.system("mkdir -p %s" % (output_path))
        for output_file in [f"{jnx:05d}.jpg" for jnx in output_idxs]:
            shutil.copy2(
                f"{outdir}/JPEGImagesRaw/Full-Resolution/{seqname}/{output_file}",
                output_path,
            )

        print("frame filtering done: %s" % seqname)
    else:
        print("lack of motion, ignored: %s" % seqname)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <seqname> <outdir>")
        print(f"Example: python {sys.argv[0]} cat-pikachu-0-0000 'database/processed/'")
        exit()
    seqname = sys.argv[1]
    outdir = sys.argv[2]
    frame_filter(seqname, outdir)
