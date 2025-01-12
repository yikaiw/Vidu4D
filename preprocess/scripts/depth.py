# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python preprocess/scripts/depth.py 2023-03-30-21-20-57-cat-pikachu-5-0000
import glob
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.insert(
    0,
    "%s/../" % os.path.join(os.path.dirname(__file__)),
)


from libs.utils import resize_to_target


def depth2pts(depth):
    Kmat = np.eye(3)
    Kmat[0, 0] = depth.shape[0]
    Kmat[1, 1] = depth.shape[0]
    Kmat[0, 2] = depth.shape[1] / 2
    Kmat[1, 2] = depth.shape[0] / 2

    xy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    hxy = np.stack(
        [xy[0].flatten(), xy[1].flatten(), np.ones_like(xy[0].flatten())], axis=0
    )
    hxy = np.linalg.inv(Kmat) @ hxy
    xyz = hxy * depth.flatten()
    return xyz.T


def extract_depth(seqname):
    image_dir = "database/processed/JPEGImages/Full-Resolution/%s/" % seqname
    output_dir = image_dir.replace("JPEGImages", "Depth")

    # torch.hub.help(
    #     "intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True
    # )  # Triggers fresh download of MiDaS repo

    model_zoe_nk = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    zoe = model_zoe_nk.to("cuda")

    os.makedirs(output_dir, exist_ok=True)
    for img_path in sorted(glob.glob(f"{image_dir}/*.jpg")):
        # print(img_path)
        image = Image.open(img_path)
        depth = zoe.infer_pil(image)
        depth = resize_to_target(depth, is_flow=False).astype(np.float16)
        out_path = f"{output_dir}/{os.path.basename(img_path).replace('.jpg', '.npy')}"
        np.save(out_path, depth)
        # pts = depth2pts(depth)

    print("zoe depth done: ", seqname)


if __name__ == "__main__":
    seqname = sys.argv[1]

    extract_depth(seqname)
