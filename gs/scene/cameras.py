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

import torch
from torch import nn
import numpy as np
from gs.utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
            
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

class KCamera(nn.Module):
    def __init__(self, H, W, left, right, top, bottom, 
                 trans=torch.tensor([.0, .0, .0]), scale=1.0, data_device = "cuda"
                 ):
        super(KCamera, self).__init__()
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.R = torch.eye(3).to(self.data_device)
        self.T = torch.zeros(3).to(self.data_device)
        self.FoVx = 2.0 * torch.arctan((right - left) / 2.0)
        self.FoVy = 2.0 * torch.arctan((top - bottom) / 2.0)
            

        self.original_image = torch.zeros(1, H, W).clamp(0.0, 1.0).to(self.data_device)
        self.image_width = W
        self.image_height = H

        # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        # print("Warning: Need specify zfar and znear")
        self.zfar = 10
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = getWorld2View2(self.R, self.T, trans, scale).transpose(0, 1).to(self.data_device)

        # import ipdb; ipdb.set_trace()
        P = torch.zeros(4, 4).to(self.data_device)

        z_sign = 1.0
        znear=self.znear
        zfar=self.zfar

        left = left * znear
        right = right * znear
        top = top * znear
        bottom = bottom * znear

        top_inv = -top
        bottom_inv = -bottom
        bottom = top_inv
        top = bottom_inv

        left_inv = -left
        right_inv = -right
        left = right_inv
        right = left_inv


        # draw the frustum
        # import trimesh
        # pts = torch.tensor([[left, bottom, znear], [right, bottom, znear], [right, top, znear], [left, top, znear]])
        
        # for i in range(20):
        #     z = 0.05 * (i + 1) * (zfar - znear) + znear
        #     ratio = z / znear
        #     pts = torch.concat([pts, torch.tensor([[left * ratio, bottom * ratio, z], [right * ratio, bottom * ratio, z], [right * ratio, top * ratio, z], [left * ratio, top * ratio, z]])], dim=0)

        # cloud = trimesh.points.PointCloud(pts.detach().cpu().numpy())
        # cloud.export('frustum.ply')
        # import ipdb; ipdb.set_trace()


        # print("projmat nftbrl\n", znear, zfar, top, bottom, right, left)
        # print("")

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * (znear + zfar) / (zfar - znear)
        P[2, 3] = - 2 * (zfar * znear) / (zfar - znear)

        # seems buggy
        # P[2, 2] = z_sign * zfar / (zfar - znear)
        # P[2, 3] = -(zfar * znear) / (zfar - znear)

        self.projection_matrix = P.transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        # self.camera_center = self.world_view_transform.inverse()[3, :3]
        assert self.world_view_transform[0,0] == 1.0 and self.world_view_transform[1,1] == 1.0 and self.world_view_transform[2,2] == 1.0 
        assert self.world_view_transform[0,3] == 0.0 and self.world_view_transform[1,3] == 0.0 and self.world_view_transform[2,3] == 0.0
        self.camera_center = self.world_view_transform[3, :3]
        # import ipdb; ipdb.set_trace()