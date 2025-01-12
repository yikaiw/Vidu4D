# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python preprocess/scripts/canonical_registration.py 2023-04-03-18-02-32-cat-pikachu-5-0000 256 quad
# this assumes one component (object) per video
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

sys.path.insert(
    0,
    "%s/../third_party" % os.path.join(os.path.dirname(__file__)),
)

sys.path.insert(
    0,
    "%s/../" % os.path.join(os.path.dirname(__file__)),
)

sys.path.insert(
    0,
    "%s/../../" % os.path.join(os.path.dirname(__file__)),
)

from libs.io import get_bbox, read_images_densepose
from libs.torch_models import CanonicalRegistration, get_class
from libs.utils import robust_rot_align
from viewpoint.dp_viewpoint import ViewponitNet

from lab4d.utils.geom_utils import K2inv, K2mat, Kmatinv
from lab4d.utils.quat_transform import quaternion_translation_to_se3
from lab4d.utils.vis_utils import draw_cams


def save_resampled_feat(imglist, feats, dp2raws, prefix, crop_size):
    # load crop2raw
    crop2raw_path = imglist[0].replace("JPEGImages", "Annotations").rsplit("/", 1)[0]
    crop2raw_path = crop2raw_path + "/%s-%d-crop2raw.npy" % (prefix, crop_size)
    crop2raws = np.load(crop2raw_path)
    feats_expanded = []
    for it, impath in enumerate(imglist):
        feat = feats[it]
        feat_width = feat.shape[-1]
        crop2raw = crop2raws[it]
        # compute transform
        crop2dp = K2inv(dp2raws[it]) @ K2mat(crop2raw)
        # resample
        feat = torch.tensor(feat, dtype=torch.float32)[None]
        crop2dp = torch.tensor(crop2dp, dtype=torch.float32)
        xy_range = torch.linspace(0, crop_size, steps=feat_width, dtype=torch.float32)
        hxy = torch.cartesian_prod(xy_range, xy_range)
        hxy = torch.stack([hxy[:, 1], hxy[:, 0], torch.ones_like(hxy[:, 0])], -1)
        hxy = hxy @ crop2dp.T
        hxy = hxy[..., :2] / feat_width * 2 - 1
        feat = F.grid_sample(feat, hxy.view(-1, feat_width, feat_width, 2))
        feats_expanded.append(feat)
    feats = torch.cat(feats_expanded, dim=0).numpy()
    feats = np.transpose(feats, (0, 2, 3, 1))
    return feats


def canonical_registration(seqname, crop_size, obj_class, component_id=1):
    # load rgb/depth
    imgdir = "database/processed/JPEGImages/Full-Resolution/%s" % seqname
    imglist = sorted(glob.glob("%s/*.jpg" % imgdir))
    save_path = imgdir.replace("JPEGImages", "Cameras")

    cams_view1 = np.load("%s/%02d.npy" % (save_path, component_id))

    # classifiy human or not
    if obj_class == "other":
        import json, pdb

        cam_path = (
            "database/processed/Cameras/Full-Resolution/%s/01-manual.json" % seqname
        )
        with open(cam_path) as f:
            cams_canonical = json.load(f)
            cams_canonical = {int(k): np.asarray(v) for k, v in cams_canonical.items()}
    else:
        if obj_class == "human":
            is_human = True
        elif obj_class == "quad":
            is_human = False
        else:
            raise ValueError("Unknown object class: %s" % obj_class)
        viewpoint_net = ViewponitNet(is_human=is_human)
        viewpoint_net.cuda()
        viewpoint_net.eval()

        # densepose inference
        rgbs, masks = read_images_densepose(imglist)
        with torch.no_grad():
            cams_canonical, feats, dp2raws = viewpoint_net.run_inference(rgbs, masks)

        # save densepose features
        # resample features to the cropped image size
        feats_crop = save_resampled_feat(imglist, feats, dp2raws, "crop", crop_size)
        feats_full = save_resampled_feat(imglist, feats, dp2raws, "full", crop_size)
        save_path_dp = save_path.replace("Cameras", "Features")
        os.makedirs(save_path_dp, exist_ok=True)
        np.save(
            "%s/crop-%d-cse-%02d.npy" % (save_path_dp, crop_size, component_id),
            feats_crop.astype(np.float16),
        )
        np.save(
            "%s/full-%d-cse-%02d.npy" % (save_path_dp, crop_size, component_id),
            feats_full.astype(np.float16),
        )
        cams_canonical = {k: v for k, v in enumerate(cams_canonical)}

    # canonical registration (smoothes the camera poses)
    print("num cams annotated: %d" % len(cams_canonical.keys()))
    rgbpath_list = [imglist[i] for i in cams_canonical.keys()]
    cams_canonical_vals = np.stack(list(cams_canonical.values()), 0)
    draw_cams(cams_canonical_vals, rgbpath_list=rgbpath_list).export(
        "%s/cameras-%02d-canonical-prealign.obj" % (save_path, component_id)
    )
    registration = CanonicalRegistration(cams_canonical, cams_view1)
    registration.cuda()
    quat, trans = registration.optimize()
    cams_pred = quaternion_translation_to_se3(quat, trans).cpu().numpy()

    # fixed depth
    cams_pred[:, :2, 3] = 0
    cams_pred[:, 2, 3] = 3

    # compute initial camera trans with 2d bbox
    # depth = focal * sqrt(surface_area / bbox_area) = focal / bbox_size
    # xytrn = depth * (pxy - crop_size/2) / focal
    # surface_area = 1
    for it, imgpath in enumerate(imglist):
        bbox = get_bbox(imgpath, component_id=component_id)
        if bbox is None:
            continue
        shape = cv2.imread(imgpath).shape[:2]

        focal = max(shape)
        depth = focal / np.sqrt(bbox[2] * bbox[3])
        depth = min(depth, 10)  # depth might be too large for mis-detected frames

        center_bbox = bbox[:2] + bbox[2:] / 2
        center_img = np.array(shape[::-1]) / 2
        xytrn = depth * (center_bbox - center_img) / focal

        cams_pred[it, 2, 3] = depth
        cams_pred[it, :2, 3] = xytrn

    np.save("%s/%02d-canonical.npy" % (save_path, component_id), cams_pred)
    draw_cams(cams_pred, rgbpath_list=imglist).export(
        "%s/cameras-%02d-canonical.obj" % (save_path, component_id)
    )
    print("canonical registration (crop_size: %d) done: %s" % (crop_size, seqname))


if __name__ == "__main__":
    seqname = sys.argv[1]
    crop_size = int(sys.argv[2])
    obj_class = sys.argv[3]

    canonical_registration(seqname, crop_size, obj_class)
