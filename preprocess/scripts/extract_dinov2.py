# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python preprocess/scripts/extract_dinov2.py cat-pikachu 256 "0"
# this assumes one component (object) per video
import configparser
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.getcwd())

sys.path.insert(
    0,
    "%s/../" % os.path.join(os.path.dirname(__file__)),
)

from libs.io import read_frame_data

from lab4d.utils.gpu_utils import gpu_map


def extract_dino_feat(dinov2_model, rgb, size=None):
    """
    rgb: (s, s, 3), 0-1
    feat: (s,s, 384)
    """
    device = next(dinov2_model.parameters()).device
    h, w, _ = rgb.shape

    img = Image.fromarray(rgb)
    transform = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    img = transform(img)[:3].unsqueeze(0).to(device)
    # TODO: use stride=4 to get high-res feature
    feat = dinov2_model.forward_features(img)["x_norm_patchtokens"].reshape(16, 16, -1)
    if size is None:
        size = (h, w)
    feat = F.interpolate(feat.permute(2, 0, 1)[None], size=size, mode="bilinear")
    feat = feat[0].permute(1, 2, 0)
    feat = feat.cpu().numpy()
    return feat


def load_dino_model(gpu_id=0):
    # load dinov2: small models producds smoother pca-ed features
    dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dinov2_model = dinov2_model.to("cuda:%d" % gpu_id)
    dinov2_model.eval()
    return dinov2_model


def extract_dinov2_seq(seqname, crop_size, use_full, component_id, pca_save):
    dinov2_model = load_dino_model()
    # rgb path
    imgdir = "database/processed/JPEGImages/Full-Resolution/%s" % seqname
    save_path = imgdir.replace("JPEGImages", "Cameras")
    imglist = sorted(glob.glob("%s/*.jpg" % imgdir))
    feats = []
    for it, impath in enumerate(imglist):
        # print(impath)
        # rgb: (s, s, 3), 0-1
        rgb, _, mask, _ = read_frame_data(
            impath, crop_size, use_full, component_id, with_flow=False
        )
        rgb = (rgb * 255).astype(np.uint8)  # to RGB
        mask = mask.astype(np.uint8)
        h, w, _ = rgb.shape

        with torch.no_grad():
            feat = extract_dino_feat(dinov2_model, rgb, size=(112, 112))
        feat = feat.reshape(-1, feat.shape[-1])
        feat = pca_save.transform(feat)
        feat = feat.reshape(112, 112, -1)
        feat = feat / np.linalg.norm(feat, axis=-1)[..., None]
        mask = cv2.resize(mask, (112, 112))
        feat = feat * mask.astype(np.float32)[..., None]
        feats.append(feat)

        # visualization
        # pca_features = pca_vis.transform(feat)
        # pca_features = (pca_features - pca_features.min()) / (
        #     pca_features.max() - pca_features.min()
        # )
        # pca_features = pca_features * 255
        # pdb.set_trace()
        # cv2.imwrite("tmp/0.jpg", pca_features.reshape(h, w, 3).astype(np.uint8))

    feats = np.stack(feats, 0)  # N, h,w, 16
    save_path_dp = save_path.replace("Cameras", "Features")
    os.makedirs(save_path_dp, exist_ok=True)
    if use_full:
        prefix = "full"
    else:
        prefix = "crop"
    save_path_dp = "%s/%s-%d-dinov2-%02d.npy" % (
        save_path_dp,
        prefix,
        crop_size,
        component_id,
    )
    np.save(save_path_dp, feats.astype(np.float16))
    print("dino features saved to %s" % save_path_dp)


def extract_dinov2(seqname, crop_size, component_id=1, gpulist=[0]):
    dinov2_model = load_dino_model(gpu_id=gpulist[0])
    # compute pca matrix over all frames
    # load image path
    config = configparser.RawConfigParser()
    config.read("database/configs/%s.config" % seqname)

    imglist_all = []
    for vidid in range(len(config.sections()) - 1):
        seqname = config.get("data_%d" % vidid, "img_path").strip("/").split("/")[-1]
        # rgb path
        imgdir = "database/processed/JPEGImages/Full-Resolution/%s" % seqname
        imglist_all += sorted(glob.glob("%s/*.jpg" % imgdir))[:-1]
    imglist_all_perm = np.random.permutation(imglist_all)

    feat_sampled = []
    for it, impath in enumerate(imglist_all_perm[:100]):
        # print(impath)
        # rgb: (s, s, 3), 0-1
        rgb, _, mask, _ = read_frame_data(
            impath, crop_size, False, component_id, with_flow=False
        )
        rgb = (rgb * 255).astype(np.uint8)  # to RGB
        mask = mask.astype(np.uint8)

        with torch.no_grad():
            feat = extract_dino_feat(dinov2_model, rgb)
        feat = feat.reshape(-1, feat.shape[-1])[mask.reshape(-1) > 0]
        rand_idx = np.random.permutation(len(feat))[:1000]
        feat_sampled.append(feat[rand_idx])

    feat_sampled = np.concatenate(feat_sampled, 0)

    with threadpool_limits(limits=1):
        pca_vis = PCA(n_components=3)
        pca_vis.fit(feat_sampled)

        pca_save = PCA(n_components=16)
        pca_save.fit(feat_sampled)

    # compute features with pca
    args = []
    for vidid in range(len(config.sections()) - 1):
        seqname = config.get("data_%d" % vidid, "img_path").strip("/").split("/")[-1]
        args.append((seqname, crop_size, True, component_id, pca_save))
        args.append((seqname, crop_size, False, component_id, pca_save))

    gpu_map(extract_dinov2_seq, args, gpus=gpulist)


if __name__ == "__main__":
    seqname = sys.argv[1]
    crop_size = int(sys.argv[2])
    gpulist = [int(n) for n in sys.argv[3].split(",")]

    extract_dinov2(seqname, crop_size, gpulist=gpulist)
