# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pdb

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes as create_boxes

import sys

sys.path.insert(0, "preprocess/third_party/detectron2/projects/DensePose/")
from densepose import add_densepose_config


# load model
def create_cse(is_human):
    if is_human:
        dp_config_path = "preprocess/third_party/viewpoint/configs/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml"
        dp_weight_path = "https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x/250713061/model_final_1d3314.pkl"
    else:
        dp_config_path = "preprocess/third_party/viewpoint/configs/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k.yaml"
        dp_weight_path = "https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k/253498611/model_final_6d69b7.pkl"

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(dp_config_path)
    cfg.MODEL.WEIGHTS = dp_weight_path
    model = build_model(cfg)  # returns a torch.nn.Module
    DetectionCheckpointer(model).load(
        cfg.MODEL.WEIGHTS
    )  # load a file, usually from cfg.MODEL.WEIGHTS
    return model


def preprocess_image(image, mask):
    h, w, _ = image.shape

    # resize
    max_size = 1333
    if h > w:
        h_rszd, w_rszd = max_size, max_size * w // h
    else:
        h_rszd, w_rszd = max_size * h // w, max_size
    image = cv2.resize(image, (w_rszd, h_rszd))
    mask = cv2.resize(mask.astype(float), (w_rszd, h_rszd)).astype(np.uint8)

    # pad
    h_pad = (1 + h_rszd // 32) * 32
    w_pad = (1 + w_rszd // 32) * 32
    image_tmp = np.zeros((h_pad, w_pad, 3)).astype(np.uint8)
    mask_tmp = np.zeros((h_pad, w_pad)).astype(np.uint8)
    image_tmp[:h_rszd, :w_rszd] = image
    mask_tmp[:h_rszd, :w_rszd] = mask
    image = image_tmp
    mask = mask_tmp

    # preprocess image and box
    indices = np.where(mask > 0)
    xid = indices[1]
    yid = indices[0]
    center = ((xid.max() + xid.min()) // 2, (yid.max() + yid.min()) // 2)
    length = (
        int((xid.max() - xid.min()) * 1.0 // 2),
        int((yid.max() - yid.min()) * 1.0 // 2),
    )
    bbox = [center[0] - length[0], center[1] - length[1], length[0] * 2, length[1] * 2]
    bbox = [
        max(0, bbox[0]),
        max(0, bbox[1]),
        min(w_pad, bbox[0] + bbox[2]),
        min(h_pad, bbox[1] + bbox[3]),
    ]
    bbox_raw = bbox.copy()  # bbox in the raw image coordinate
    bbox_raw[0] *= w / w_rszd
    bbox_raw[2] *= w / w_rszd
    bbox_raw[1] *= h / h_rszd
    bbox_raw[3] *= h / h_rszd
    return image, mask, bbox, bbox_raw


def run_cse(model, image, mask):
    image, mask, bbox, bbox_raw = preprocess_image(image, mask)

    image = torch.Tensor(image).cuda().permute(2, 0, 1)[None]
    image = torch.stack([(x - model.pixel_mean) / model.pixel_std for x in image])
    pred_boxes = torch.Tensor([bbox]).cuda()
    pred_boxes = create_boxes(pred_boxes)

    # inference
    model.eval()
    with torch.no_grad():
        features = model.backbone(image)
        features = [features[f] for f in model.roi_heads.in_features]
        features = [model.roi_heads.decoder(features)]
        features_dp = model.roi_heads.densepose_pooler(features, [pred_boxes])
        densepose_head_outputs = model.roi_heads.densepose_head(features_dp)
        densepose_predictor_outputs = model.roi_heads.densepose_predictor(
            densepose_head_outputs
        )
        coarse_segm_resized = densepose_predictor_outputs.coarse_segm[0]
        embedding_resized = densepose_predictor_outputs.embedding[0]

    # use input mask
    x, y, xx, yy = bbox
    mask_box = mask[y:yy, x:xx]
    mask_box = torch.Tensor(mask_box).cuda()[None, None]
    mask_box = (
        F.interpolate(mask_box, coarse_segm_resized.shape[1:3], mode="bilinear")[0, 0]
        > 0
    )

    # output embedding
    embedding = embedding_resized  # size does not matter for a image code
    embedding = embedding * mask_box.float()[None]

    # output dp2raw
    bbox_raw = np.asarray(bbox_raw)
    dp2raw = np.concatenate(
        [(bbox_raw[2:] - bbox_raw[:2]) / embedding.shape[1], bbox_raw[:2]]
    )
    return embedding, dp2raw
