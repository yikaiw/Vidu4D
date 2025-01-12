# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import sys, os
import numpy as np
import pdb
import math
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

sys.path.insert(
    0,
    "%s/../../" % os.path.join(os.path.dirname(__file__)),
)

from lab4d.utils.quat_transform import quaternion_to_matrix
from viewpoint.cselib import create_cse, run_cse


class NeRF_old(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        in_channels_xyz=63,
        in_channels_dir=27,
        out_channels=3,
        skips=[4],
        raw_feat=False,
        init_beta=1.0 / 100,
        activation=nn.ReLU(True),
        in_channels_code=0,
        vid_code=None,
        color_act=True,
    ):
        """
        adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        in_channels_code: only used for nerf_skin,
        """
        super(NeRF_old, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_code = in_channels_code
        self.skips = skips
        self.out_channels = out_channels
        self.raw_feat = raw_feat
        self.color_act = color_act

        # video code
        self.vid_code = vid_code
        if vid_code is not None:
            self.num_vid, self.num_codedim = self.vid_code.weight.shape
            in_channels_xyz += self.num_codedim
            self.rand_ratio = 1.0  # 1: fully random

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, activation)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(W + in_channels_dir, W // 2), activation
        )

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
            nn.Linear(W // 2, self.out_channels),
        )

        self.beta = torch.Tensor([init_beta])  # logbeta
        self.beta = nn.Parameter(self.beta)
        self.symm_ratio = 0
        self.rand_ratio = 0
        self.use_dir = False  # use app code instead of view dir

    def reinit(self, gain=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.weight, "data"):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5 * gain))
                if hasattr(m.bias, "data"):
                    m.bias.data.zero_()

    def forward(self, x, vidid=None, beta=None):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            vidid: same size as input_xyz

        Outputs:
            out: (B, 4), rgb and sigma
        """
        if x.shape[-1] == self.in_channels_xyz and not self.raw_feat:
            sigma_only = True
        else:
            sigma_only = False
        if x.shape[-1] == self.in_channels_xyz:
            input_xyz, input_dir = torch.split(x, [self.in_channels_xyz, 0], dim=-1)
        else:
            input_xyz, input_dir = torch.split(
                x, [self.in_channels_xyz, self.in_channels_dir], dim=-1
            )

        # add instance shape
        if self.vid_code is not None:
            if vidid is None:
                vid_code = self.vid_code.weight.mean(0).expand(
                    input_xyz.shape[:-1] + (-1,)
                )
            else:
                vid_code = self.vid_code(vidid)
            if self.training:
                vidid = torch.randint(self.num_vid, input_xyz.shape[:1])
                vidid = vidid.to(input_xyz.device)
                rand_code = self.vid_code(vidid)
                rand_code = rand_code[:, None].expand(vid_code.shape)
                rand_mask = torch.rand_like(vidid.float()) < self.rand_ratio
                vid_code = torch.where(rand_mask[:, None, None], rand_code, vid_code)
            input_xyz = torch.cat([input_xyz, vid_code], -1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        if self.raw_feat:
            out = rgb
        else:
            if self.color_act:
                rgb = rgb.sigmoid()
            out = torch.cat([rgb, sigma], -1)
        return out


class RTHead_old(NeRF_old):
    """
    modify the output to be rigid transforms
    """

    def __init__(self, use_quat, **kwargs):
        super(RTHead_old, self).__init__(**kwargs)
        # use quaternion when estimating full rotation
        # use exponential map when estimating delta rotation
        self.use_quat = use_quat
        if self.use_quat:
            self.num_output = 7
        else:
            self.num_output = 6
        self.scale_t = 0.1

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.bias, "data"):
                    m.bias.data.zero_()
        self.reinit(gain=1)

    def reinit(self, gain=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.weight, "data"):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5 * gain))
                if hasattr(m.bias, "data"):
                    m.bias.data.zero_()

    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        x = super(RTHead_old, self).forward(x)
        bs = x.shape[0]
        rts = x.view(-1, self.num_output)  # bs B,x
        B = rts.shape[0] // bs

        tmat = rts[:, 0:3] * self.scale_t

        if self.use_quat:
            rquat = rts[:, 3:7]
            rquat = F.normalize(rquat, 2, -1)
            rmat = quaternion_to_matrix(rquat)
        else:
            raise NotImplementedError
        rmat = rmat.view(-1, 3, 3)
        rmat[..., 1:3] *= -1  # gl coordinate to cv coordinate for rotation

        rts = torch.zeros(rmat.shape[0], 4, 4, device=rmat.device)
        rts[:, :3, :3] = rmat
        rts[:, :3, 3] = tmat
        rts[:, 3, 3] = 1
        return rts


class ResNetConv(nn.Module):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    """

    def __init__(self, in_channels):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
        self.resnet.fc = None

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x


## 2D convolution layers
def conv2d(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    """
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )


class Encoder(nn.Module):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, in_channels=3, out_channels=128, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(in_channels=in_channels)
        self.conv1 = conv2d(batch_norm, 512, out_channels, stride=1, kernel_size=3)
        # net_init(self.conv1)

    def forward(self, img):
        feat = self.resnet_conv.forward(img)  # 512,4,4
        feat = self.conv1(feat)  # 128,4,4
        feat = F.max_pool2d(feat, 4, 4)
        feat = feat.view(img.size(0), -1)
        return feat


class ViewponitNet(nn.Module):
    def __init__(self, is_human):
        super(ViewponitNet, self).__init__()
        print("viewpoint loading... is human: ", is_human)
        self.dp_net = create_cse(is_human)
        self.viewpoint_net = self.create_viewpoint_net(is_human)

    def create_viewpoint_net(self, is_human):
        if is_human:
            weights_path = "preprocess/third_party/viewpoint/human.pth"
        else:
            weights_path = "preprocess/third_party/viewpoint/quad.pth"

        cnn_head = RTHead_old(
            use_quat=True,
            D=1,
            in_channels_xyz=128,
            in_channels_dir=0,
            out_channels=7,
            raw_feat=True,
        )
        viewpoint = nn.Sequential(
            Encoder((112, 112), in_channels=16, out_channels=128), cnn_head
        )

        states = torch.load(weights_path, map_location="cpu")
        states = self.rm_module_prefix(states, prefix="module.nerf_root_rts")
        viewpoint.load_state_dict(states, strict=False)
        return viewpoint

    def forward(self, rgb, mask):
        feat, dp2raw = run_cse(self.dp_net, rgb, mask)
        feat = feat[None]
        feat = F.normalize(feat, 2, 1)
        viewpoint = self.viewpoint_net(feat)
        return viewpoint, feat, dp2raw

    def run_inference(self, rgbs, masks):
        cams = []
        feats = []
        dp2raws = []
        for idx, rgb in enumerate(rgbs):
            # print(idx)
            mask = masks[idx]
            # use densepose to process rgb
            cam, feat, dp2raw = self.forward(rgb, mask)
            cams.append(cam.cpu().numpy())
            feats.append(feat.cpu().numpy())
            dp2raws.append(dp2raw)
        cams = np.concatenate(cams, 0)
        cams[:, :2, 3] = 0
        cams[:, 2, 3] = 3
        feats = np.concatenate(feats, 0)
        dp2raws = np.stack(dp2raws, 0)
        return cams, feats, dp2raws

    @staticmethod
    def rm_module_prefix(states, prefix="module"):
        new_dict = {}
        for i in states.keys():
            v = states[i]
            if i[: len(prefix)] == prefix:
                i = i[len(prefix) + 1 :]
            new_dict[i] = v
        return new_dict
