#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if len(fname) < 8:
            continue
        # extract number from fname
        num = int(fname.split(".")[0])
        if num % 4 != 2:
            continue

        render = Image.open(os.path.join(renders_dir, fname))
        gt = Image.open(os.path.join(gt_dir, fname.replace("png", "jpg")))

        # apply mask to render and gt
        mask_full_reso = Image.open(os.path.join(renders_dir, fname).replace("rgb", "ref_mask_full"))
        render = Image.composite(render, Image.new("RGB", render.size, "white"), mask_full_reso)
        gt = Image.composite(gt, Image.new("RGB", gt.size, "white"), mask_full_reso)

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate():

    scenes = ["black-dragon"]
    expname = ["2dgs-final-newL1"]
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in scenes:

        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        for method in expname:
            print("Method:", method)
            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = os.path.join("logdir", f"{scene_dir}-{method}", "imgs")
            # gt_dir = os.path.join(method_dir, "ref_rgb")
            
            gt_dir = os.path.join("/mnt/mfs/xinzhou.wang/repo/DreamBANMo/database/processed/JPEGImagesRaw/Full-Resolution", scene_dir + "-0000")
            renders_dir = os.path.join(method_dir, "rgb")
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
            per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

        with open("logdir/" + scene_dir + "-results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open("logdir/" + scene_dir + "-per-view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    # parser = ArgumentParser(description="Training script parameters")
    # parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    # args = parser.parse_args()
    # evaluate(args.model_paths)
    evaluate()
