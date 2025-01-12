# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.io import read_frame_data

from lab4d.nnutils.pose import CameraMLP
from lab4d.utils.geom_utils import rot_angle
from lab4d.utils.quat_transform import quaternion_translation_to_se3


# solve optimization problem
class CanonicalRegistration(nn.Module):
    def __init__(self, cams_canonical_dict, cams_view1):
        # annoated canonical cameras
        self.annotated_idx = np.stack(list(cams_canonical_dict.keys()))
        cams_canonical = np.eye(4)[None].repeat(len(cams_view1), axis=0)
        cams_canonical[self.annotated_idx] = np.stack(
            list(cams_canonical_dict.values()), 0
        )
        cams_canonical = cams_canonical.astype(np.float32)
        cams_view1 = np.array(cams_view1).astype(np.float32)
        super(CanonicalRegistration, self).__init__()
        self.cam_net = CameraMLP(cams_canonical)
        self.cams_canonical = nn.Parameter(
            torch.tensor(cams_canonical), requires_grad=False
        )
        cams_rel_gt = cams_view1[1:, :3, :3] @ np.transpose(
            cams_view1[:-1, :3, :3], (0, 2, 1)
        )
        self.cams_rel_gt = nn.Parameter(torch.tensor(cams_rel_gt), requires_grad=False)

    def forward(self, unary_wt=1.0, pairwise_wt=1.0):
        # (1) rotation should be close to canonical cameras
        quat, trans = self.cam_net.get_vals()
        cams_pred = quaternion_translation_to_se3(quat, trans)

        loss_unary = rot_angle(
            cams_pred[self.annotated_idx, :3, :3]
            @ self.cams_canonical[self.annotated_idx, :3, :3].permute(0, 2, 1)
        )

        # (2) relative translation should be close to procrustes
        cams_rel = cams_pred[1:, :3, :3] @ cams_pred[:-1, :3, :3].permute(0, 2, 1)

        loss_pairwise = rot_angle(
            cams_rel[:, :3, :3] @ self.cams_rel_gt[:, :3, :3].permute(0, 2, 1)
        )

        loss = unary_wt * loss_unary.mean() + pairwise_wt * loss_pairwise.mean()
        return loss

    def init_pairwise(self, lr=5e-4, num_iter=2000):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        i = 0
        while True:
            optimizer.zero_grad()
            loss = self.forward(unary_wt=0.0, pairwise_wt=1.0)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("iter %d loss %f" % (i, loss.item()))
            i += 1
            if loss < 0.015 or i > num_iter:
                break

    def optimize(self, lr=5e-4, num_iter=2000):
        self.cam_net.base_init()

        # initialize pairwise loss
        self.init_pairwise()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        i = 0
        while True:
            optimizer.zero_grad()
            loss = self.forward()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("iter %d loss %f" % (i, loss.item()))
            i += 1
            if loss < 0.030 or i > num_iter:
                break
        self.cam_net.eval()
        with torch.no_grad():
            cams = self.cam_net.get_vals()
        return cams


# get class clabel
# TODO: this is a hack, should be replaced with the labels we get from VIS method
def get_class(imglist, component_id):
    # select 10 frames
    if len(imglist) // 20 > 1:
        imglist = imglist[:: len(imglist) // 20]
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    predictor = DefaultPredictor(cfg)

    class_ids = []
    for imgpath in imglist:
        rgb, _, _, _ = read_frame_data(
            imgpath, crop_size=512, use_full=0, component_id=component_id
        )
        rgb = (rgb * 255).astype(np.uint8)[..., ::-1].copy()  # to BGR
        outputs = predictor(rgb)
        classes = outputs["instances"].pred_classes.cpu().numpy()
        if len(classes) == 0:
            continue
        confidence = outputs["instances"].scores.cpu().numpy()
        class_ids.append(classes[confidence.argmax()])

    if np.bincount(class_ids).argmax() == 0:
        print("human detected")
        return 1
    else:
        print("object detected")
        return 0
