import cv2
import pdb
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import time

from models.VCNplus import VCN, WarpModule, flow_reg


def load_eval_checkpoint(model_path, maxdisp=256, fac=1):
    """Load a VCN model from checkpoint in eval mode

    Args:
        model_path (str): Path to .pth checkpoint
        maxdisp (int): Maximum disparity. Only affects the coarsest cost
            volume size (default: 256)
        fac (int): Controls the shape of search grid. Only affects the
            coarsest cost volume size (default: 1)

    Returns:
        model (VCN): VCN model
    """
    # construct model
    model = VCN([1, 256, 256], md=[int(4 * (maxdisp / 256)), 4, 4, 4, 4], fac=fac)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.cuda()

    pretrained_dict = torch.load(model_path)
    pretrained_dict["state_dict"] = {
        k: v for k, v in pretrained_dict["state_dict"].items()
    }
    model.load_state_dict(pretrained_dict["state_dict"], strict=False)
    model.mean_L = pretrained_dict["mean_L"]
    model.mean_R = pretrained_dict["mean_R"]
    model.eval()

    return model


def modify_flow_module(model, max_h, max_w):
    for i in range(len(model.module.reg_modules)):
        model.module.reg_modules[i] = flow_reg(
            [1, max_w // (2 ** (6 - i)), max_h // (2 ** (6 - i))],
            ent=getattr(model.module, "flow_reg%d" % 2 ** (6 - i)).ent,
            maxdisp=getattr(model.module, "flow_reg%d" % 2 ** (6 - i)).md,
            fac=getattr(model.module, "flow_reg%d" % 2 ** (6 - i)).fac,
        ).cuda()
    for i in range(len(model.module.warp_modules)):
        model.module.warp_modules[i] = WarpModule(
            [1, max_w // (2 ** (6 - i)), max_h // (2 ** (6 - i))]
        ).cuda()


def process_flow_input(img_o, model_mean, max_h, max_w):
    if len(img_o.shape) == 2:
        img_o = np.tile(img_o[:, :, None], (1, 1, 3))  # H_o, W_o, 3

    # resize
    img = cv2.resize(img_o, (max_w, max_h))  # H, W, 3
    img_noaug = torch.from_numpy(img).cuda()  # H, W, 3
    img_noaug = img_noaug.to(torch.float32)[None] / 255.0  # 1, H, W, 3
    model_mean = np.asarray(model_mean, dtype=np.float32).mean(0)  # 3,
    model_mean = torch.from_numpy(model_mean).cuda()  # 3,

    # flip channel, subtract mean
    img = torch.flip(img_noaug, [-1]) - model_mean[None, None, None, :]  # 1, H, W, 3
    img = torch.moveaxis(img, -1, 1) # 1, 3, H, W
    return img, img_noaug


def make_disc_aux(imgL_noaug, max_h, max_w, input_size):
    # fill with dummy values to satisfy the input requirements
    fl_next = 1
    intr_list = [
        1, 1, 1, 1, 1, 0, 0, 1, 0, 0,
        input_size[1] / max_w, input_size[0] / max_h, fl_next
    ]
    intr_list = [
        torch.tensor([x], dtype=torch.float32, device="cuda")
        for x in intr_list
    ]

    disc_aux = [None, None, None, intr_list, imgL_noaug, None]
    return disc_aux


def flow_inference(model, imgL_o, imgR_o, max_res=2e6):
    # resize
    input_size = imgL_o.shape
    res_fac = np.sqrt(max_res / (imgL_o.shape[0] * imgL_o.shape[1]))
    maxh = imgL_o.shape[0] * res_fac
    maxw = imgL_o.shape[1] * res_fac
    max_h = int(np.ceil(maxh / 64) * 64)
    max_w = int(np.ceil(maxw / 64) * 64)

    imgL, imgL_noaug = process_flow_input(imgL_o, model.mean_L, max_h, max_w)
    imgR, imgR_noaug = process_flow_input(imgR_o, model.mean_R, max_h, max_w)

    # modify module according to inputs
    modify_flow_module(model, max_h, max_w)

    # fill with dummy values to satisfy the input requirements
    disc_aux = make_disc_aux(imgL_noaug, max_h, max_w, input_size)

    # forward
    with torch.no_grad():
        imgLR = torch.cat([imgL, imgR], dim=0)
        model.eval()
        # torch.cuda.synchronize()
        start_time = time.time()
        rts = model(imgLR, disc_aux)
        # torch.cuda.synchronize()
        ttime = time.time() - start_time
        # print("time = %.2f" % (ttime * 1000))
        flow, occ, logmid, logexp = rts

    # upsampling
    occ = cv2.resize(
        occ.data.cpu().numpy(),
        (input_size[1], input_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    flow = flow[0].cpu().numpy()  # 2, H, W
    flow = np.concatenate(
        [
            cv2.resize(flow[0], (input_size[1], input_size[0]))[:, :, np.newaxis],
            cv2.resize(flow[1], (input_size[1], input_size[0]))[:, :, np.newaxis],
            np.ones((input_size[0], input_size[1], 1), dtype=np.float32),
        ],
        -1,
    )  # H, W, 3
    flow[:, :, 0] *= imgL_o.shape[1] / max_w
    flow[:, :, 1] *= imgL_o.shape[0] / max_h

    # deal with unequal size
    if imgL_o.shape != imgR_o.shape:
        x0, y0 = np.meshgrid(range(input_size[1]), range(input_size[0]))  # screen coord
        x1 = (flow[:, :, 0] + x0) / float(imgL_o.shape[1]) * float(imgR_o.shape[1])
        y1 = (flow[:, :, 1] + y0) / float(imgL_o.shape[0]) * float(imgR_o.shape[0])
        flow[:, :, 0] = x1 - x0
        flow[:, :, 1] = y1 - y0

    return flow, occ
