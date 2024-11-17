# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python scripts/render.py --seqname --flagfile=logdir/cat-0t10-fg-bob-d0-long/opts.log --load_suffix latest

from curses import raw
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from absl import app, flags

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from lab4d.config import get_config
from lab4d.dataloader import data_utils
from lab4d.engine.trainer import Trainer
from lab4d.utils.camera_utils import (
    construct_batch,
    get_fixed_cam,
    get_object_to_camera_matrix,
    get_orbit_camera,
    get_rotating_cam,
    create_field2cam,
    get_bev_cam,
)
from lab4d.utils.geom_utils import K2inv, K2mat, mat2K
from lab4d.utils.io import make_save_dir, save_rendered
from lab4d.utils.profile_utils import torch_profile

from lab4d.utils.quat_transform import se3_to_quaternion_translation,quaternion_translation_to_se3, quaternion_translation_inverse

cudnn.benchmark = True


class RenderFlags:
    """Flags for the renderer."""

    flags.DEFINE_integer("inst_id", 0, "video/instance id")
    flags.DEFINE_integer("render_res", 128, "rendering resolution")
    flags.DEFINE_string(
        "viewpoint", "ref", "camera viewpoint, {ref,rot-elevation-degree,rot-60-0,...}"
    )
    flags.DEFINE_integer("freeze_id", -1, "freeze frame id to render, no freeze if -1")
    flags.DEFINE_integer("num_frames", -1, "frames to render if freeze_id id is used")
    flags.DEFINE_bool("noskip", False, "render all frames skipped by flow")
    flags.DEFINE_float("rot_dist", 2, "distance to rotate around object")
    flags.DEFINE_bool("nowarp", False, "")


def construct_batch_from_opts(opts, model, data_info):
    device = "cuda"
    # data info
    if "motion_id" in opts:
        video_id = opts["motion_id"]
    else:
        video_id = opts["inst_id"]
    # ref video size
    raw_size = data_info["raw_size"][video_id]  # full range of pixels
    # ref video length
    vid_length = data_utils.get_vid_length(video_id, data_info)

    # whether to freeze a frame
    if opts["freeze_id"] == -1:
        if opts["noskip"]:
            # render all frames
            frameid_sub = np.arange(vid_length)
            render_length = vid_length
        else:
            # render filtered frames
            frame_mapping = data_info["frame_info"]["frame_mapping"]
            frame_offset = data_info["frame_info"]["frame_offset"]
            frameid = frame_mapping[frame_offset[video_id] : frame_offset[video_id + 1]]

            frameid_start = data_info["frame_info"]["frame_offset_raw"][video_id]
            frameid_sub = frameid - frameid_start
            render_length = len(frameid)
    elif opts["freeze_id"] >= 0 and opts["freeze_id"] < vid_length:
        if opts["num_frames"] <= 0:
            num_frames = vid_length
        else:
            num_frames = opts["num_frames"]
        frameid_sub = np.asarray([opts["freeze_id"]] * num_frames)
    else:
        raise ValueError("frame id %d out of range" % opts["freeze_id"])
    # print("rendering frames: %s from video %d" % (str(frameid_sub), video_id))
    frameid = frameid_sub + data_info["frame_info"]["frame_offset_raw"][video_id]
    
    # get cameras wrt each field
    with torch.no_grad():
        field2cam_fr = model.fields.get_cameras(frame_id=frameid)
        intrinsics_fr = model.intrinsics.get_vals(
            frameid_sub + data_info["frame_info"]["frame_offset_raw"][video_id]
        )
        aabb = model.fields.get_aabb()
    # convert to numpy
    for k, v in field2cam_fr.items():
        field2cam_fr[k] = v.cpu().numpy()
        aabb[k] = aabb[k].cpu().numpy()
    intrinsics_fr = intrinsics_fr.cpu().numpy()

    # construct batch from user input
    if opts["viewpoint"] == "ref":
        # rotate around viewpoint
        field2cam = None
        # camera_int = None
        crop2raw = np.zeros((len(frameid_sub), 4))
        if not opts["render_res"] == -1:
            crop2raw[:, 0] = raw_size[1] / opts["render_res"]
            crop2raw[:, 1] = raw_size[0] / opts["render_res"]
        else:
            crop2raw[:, 0] = 1
            crop2raw[:, 1] = 1
        camera_int = mat2K(K2inv(crop2raw) @ K2mat(intrinsics_fr))
        crop2raw = None
    elif opts["viewpoint"].startswith("rot"):
        # rotate around field, format: rot-evelvation-degree
        elev, max_angle = [int(val) for val in opts["viewpoint"].split("-")[1:]]

        # bg_to_cam
        obj_size = (aabb["fg"][1, :] - aabb["fg"][0, :]).max()
        cam_traj = get_rotating_cam(
            len(frameid_sub), distance=obj_size * opts["rot_dist"], max_angle=max_angle
        )
        cam_elev = get_object_to_camera_matrix(elev, [1, 0, 0], 0)[None]
        cam_traj = cam_traj @ cam_elev
        field2cam = create_field2cam(cam_traj, field2cam_fr.keys())

        camera_int = np.zeros((len(frameid_sub), 4))

        # focal length = img height * distance / obj height
        camera_int[:, :2] = opts["render_res"] * 2 * 0.8  # zoom out a bit
        camera_int[:, 2:] = opts["render_res"] / 2
        raw_size = (640, 640)  # full range of pixels
        crop2raw = None

    elif opts["viewpoint"].startswith("brot"):
        # rotate around field, format: rot-evelvation-degree
        min_angle, max_angle = [int(val) for val in opts["viewpoint"].split("-")[1:]]
        if min_angle > max_angle:
            min_angle = -min_angle
        # bg_to_cam
        obj_size = (aabb["fg"][1, :] - aabb["fg"][0, :]).max()
        cam_traj = get_rotating_cam(
            len(frameid_sub), distance=obj_size * opts["rot_dist"], max_angle=max_angle, initial_angle=-min_angle
        )
        cam_elev = get_object_to_camera_matrix(360, [1, 0, 0], 0)[None]
        cam_traj = cam_traj @ cam_elev
        field2cam = create_field2cam(cam_traj, field2cam_fr.keys())

        camera_int = np.zeros((len(frameid_sub), 4))

        # focal length = img height * distance / obj height
        camera_int[:, :2] = opts["render_res"] * 2 * 0.8  # zoom out a bit
        camera_int[:, 2:] = opts["render_res"] / 2
        raw_size = (640, 640)  # full range of pixels
        crop2raw = None

    elif opts["viewpoint"].startswith("bev"):
        radius = int(opts["viewpoint"].split("-")[1])
        # render bird's eye view
        if "bg" in field2cam_fr.keys():
            # get bev wrt first frame image
            # center_to_bev = centered_to_camt0 x centered_to_rotated x camt0_to_centered x bg_to_camt0
            center_to_bev = get_object_to_camera_matrix(radius, [1, 0, 0], 0)[None]
            camt0_to_center = np.eye(4)
            camt0_to_center[2, 3] = -field2cam_fr["bg"][0, 2, 3]
            camt0_to_bev = (
                np.linalg.inv(camt0_to_center) @ center_to_bev @ camt0_to_center
            )
            bg2bev = camt0_to_bev @ field2cam_fr["bg"][:1]
            # push cameras away
            bg2bev[..., 2, 3] *= 3
            field2cam = {"bg": np.tile(bg2bev, (render_length, 1, 1))}
            if "fg" in field2cam_fr.keys():
                # if both fg and bg
                camt2bg = np.linalg.inv(field2cam_fr["bg"])
                fg2camt = field2cam_fr["fg"]
                field2cam["fg"] = field2cam["bg"] @ camt2bg @ fg2camt
        elif "fg" in field2cam_fr.keys():
            # if only fg
            field2cam = {"fg": get_bev_cam(field2cam_fr["fg"], elev=radius)}
        else:
            raise NotImplementedError

        camera_int = np.zeros((len(frameid_sub), 4))
        camera_int[:, :2] = opts["render_res"] * 2
        camera_int[:, 2:] = opts["render_res"] / 2
        raw_size = (640, 640)  # full range of pixels
        crop2raw = None
    elif opts["viewpoint"].startswith("jitter"):
        radius = opts["viewpoint"].split("-")[1]
        rounds = int(opts["viewpoint"].split("-")[2])
        obj_size = (aabb["fg"][1, :] - aabb["fg"][0, :]).max()
        jitter_size = int(radius) * obj_size * 0.1
        
        field2cam = model.fields.field_params['fg'].camera_mlp.get_vals(None)
        num_frames = field2cam[0].shape[0]

        # x,y go 3 rounds on a circle in radius=jitter_size, num_frames in total
        x = np.sin(np.linspace(0, 2*np.pi*rounds, num_frames)) * jitter_size
        y = np.cos(np.linspace(0, 2*np.pi*rounds, num_frames)) * jitter_size

        x = torch.tensor(x, device=device).float()
        y = torch.tensor(y, device=device).float()
        

        # complicated way
        R_distance = obj_size * 0.3
        new_field2cam = []
        Ks = quaternion_translation_to_se3(field2cam[0], field2cam[1]).cuda()
        center = torch.tensor([0, 0, 0], device=device).float().cuda()
        for i in range(field2cam[1].shape[0]):
            Ks[i] = invert_camera_extrinsics(Ks[i])
            Ks[i][0 ,3] += x[i]
            Ks[i][1 ,3] += y[i]

            new_K = compute_camera_extrinsics(center ,Ks[i][:3 ,3])
            new_K = invert_camera_extrinsics(new_K)


            newqt = se3_to_quaternion_translation(new_K, tuple=False)
            new_field2cam.append(newqt[None,...])
        field2cam = torch.concat(new_field2cam, dim=0).to("cuda")


        # # simple way
        # field2cam[1][:,0] += x
        # field2cam[1][:,1] += y
        # field2cam = torch.cat((field2cam[0], field2cam[1]), -1)
        # # field2cam = {"fg": field2cam}
        
        camera_int = None
        crop2raw = np.zeros((len(frameid_sub), 4))
        if not opts["render_res"] == -1:
            crop2raw[:, 0] = raw_size[1] / opts["render_res"]
            crop2raw[:, 1] = raw_size[0] / opts["render_res"]
        else:
            crop2raw[:, 0] = 1
            crop2raw[:, 1] = 1
        camera_int = mat2K(K2inv(crop2raw) @ K2mat(intrinsics_fr))
        crop2raw = None
    else:
        raise ValueError("Unknown viewpoint type %s" % opts.viewpoint)

    batch = construct_batch(
        inst_id=opts["inst_id"],
        frameid_sub=frameid_sub,
        eval_res=opts["render_res"] if opts["render_res"] > 0 else raw_size,
        field2cam=field2cam,
        camera_int=camera_int,
        crop2raw=crop2raw,
        device=device,
    )

    # if opts["viewpoint"].startswith("jitter"):
    return batch, raw_size


@torch.no_grad()
def render_batch(model, batch, nowarp=False, opts=None):
    # render batch
    start_time = time.time()
    rendered = model.evaluate(batch, is_pair=False, nowarp=nowarp)
    print("rendering time: %.3f" % (time.time() - start_time))

    return rendered


def render(opts, construct_batch_func):
    # load model/data
    nowarp = opts["nowarp"]
    opts["logroot"] = sys.argv[1].split("=")[1].rsplit("/", 2)[0]
    model, data_info, ref_dict = Trainer.construct_test_model(opts)
    batch, raw_size = construct_batch_func(opts, model, data_info)

    if "gs" in opts["fg_motion"]:
        if opts["render_res"] == -1:
            assert opts['viewpoint'] == 'ref' or opts['viewpoint'].startswith("jitter")
            batch["H"] = (torch.ones_like(batch['frameid_sub']) * raw_size[0]).long()
            batch["W"] = (torch.ones_like(batch['frameid_sub']) * raw_size[1]).long()
        else:
            if opts['viewpoint'] == 'ref':
                print("sure to render not at ref raw resolution?")
                import ipdb; ipdb.set_trace()
            batch["H"] = (torch.ones_like(batch['frameid_sub']) * opts["render_res"]).long()
            batch["W"] = (torch.ones_like(batch['frameid_sub']) * opts["render_res"]).long()

    save_dir = make_save_dir(
        opts, sub_dir="renderings_%04d/%s" % (opts["inst_id"], opts["viewpoint"])
    )

    batch["render_ref"] = opts["viewpoint"] == "ref"
    # render
    with torch_profile(save_dir, "profile", enabled=opts["profile"]):
        with torch.no_grad():
            rendered = render_batch(model, batch, nowarp=nowarp, opts=opts)
    del model
    # clean cache
    torch.cuda.empty_cache()

    rendered.update(ref_dict)

    # save rendered["render"] to jpg
    # from torchvision.utils import save_image
    # for i in range(rendered["rendered"].shape[0]):
    #     save_image(rendered["rendered"][i].permute(2,0,1), f"tmp/render_rgb{i}.jpg")
    # save_image(rendered["rendered"][0].permute(2,0,1), f"tmp/render_rgb{0}.jpg")
    if 'rendered' in rendered.keys():
        rendered['rendered'].clamp_(0, 1)
        rendered['rgb'] = rendered['rendered']
        del rendered['rendered']

    # if opts["viewpoint"] == "ref":
    #     # mask with ref_dict["ref_mask_full"]
    #     print("rendered H,W ", rendered["rgb"].shape)
    #     print("ref H,W ", ref_dict["ref_mask_full"].shape)
    #     ref_dict["ref_mask_full"] = torch.tensor(ref_dict["ref_mask_full"], device="cuda")
    #     if not ref_dict["ref_mask_full"].shape[0:3] == rendered["rgb"].shape[0:3]:
    #         # interp rendered["rgb"] to ref_dict["ref_mask_full"]
    #         rendered["rgb_interp"] = torch.nn.functional.interpolate(
    #             rendered["rgb"], ref_dict["ref_mask_full"].shape[1:3], mode="bilinear"
    #         )
    #         rendered["rgb_gtmask"] = rendered["rgb_interp"] * ref_dict["ref_mask_full"]
    #     else:
    #         rendered["rgb_gtmask"] = rendered["rgb"] * ref_dict["ref_mask_full"]
    #     if "rend_normal" in rendered.keys():
    #         if not ref_dict["ref_mask_full"].shape[0:3] == rendered["rend_normal"].shape[0:3]:
    #             # interp rendered["rgb"] to ref_dict["ref_mask_full"]
    #             rendered["rend_normal_interp"] = torch.nn.functional.interpolate(
    #                 rendered["rend_normal"], ref_dict["ref_mask_full"].shape[1:3], mode="bilinear"
    #             )
    #             rendered["surf_normal_interp"] = torch.nn.functional.interpolate(
    #                 rendered["surf_normal"], ref_dict["ref_mask_full"].shape[1:3], mode="bilinear"
    #             )
    #             rendered["rend_normal_gtmask"] = rendered["rend_normal_interp"] * ref_dict["ref_mask_full"]
    #             rendered["surf_normal_gtmask"] = rendered["surf_normal_interp"] * ref_dict["ref_mask_full"]
    #         else:
    #             rendered["rend_normal_gtmask"] = rendered["rend_normal"] * ref_dict["ref_mask_full"]
    #             rendered["surf_normal_gtmask"] = rendered["surf_normal"] * ref_dict["ref_mask_full"]

    save_rendered(rendered, save_dir, (opts["render_res"],opts["render_res"]), data_info["apply_pca_fn"])
    print("Saved to %s" % save_dir)

    # save image for eval
    if opts["viewpoint"] == "ref":
        save_dir = make_save_dir(opts, "imgs")
        key_to_save = ["rgb", "mask", "rgb_gtmask", "rgb_interp", "ref_mask_full", "ref_mask", "ref_rgb"]
        for k,v in rendered.items():
            if k in key_to_save:
                os.makedirs(f"{save_dir}/{k}", exist_ok=True)

                if k == "mask":
                    v = v.repeat(1,1,1,3)
                elif k == "ref_mask":
                    # v is numpy array
                    v = v.repeat(3, axis=-1)

                for i in range(rendered["rgb"].shape[0]-1):
                    if isinstance(v, torch.Tensor):
                        img = v[i].cpu().numpy()
                    else:
                        img = v[i]
                    img = (img * 255).astype(np.uint8)
                    img = img[:, :, ::-1]

                    # turn i to 5 digits
                    i = str(i).zfill(5)
                    cv2.imwrite(
                        f"{save_dir}/{k}/{i}.png",
                        img,
                    )



def compute_center(initial_extrinsic, R):
    """
    根据初始外参矩阵计算中心点C
    """
    # 提取旋转矩阵和位移向量
    R_matrix = initial_extrinsic[:3, :3]
    translation = initial_extrinsic[:3, 3]

    # 相机的z轴方向向量
    z_direction = R_matrix[:, 2]  # 第三列是z轴方向

    # 计算中心点C
    center = translation + R * z_direction

    return center

def compute_camera_extrinsics(C, P):
    # 计算相机z轴负方向
    z_axis = C - P
    z_axis = z_axis / torch.norm(z_axis)

    # 假设相机的上方向为y轴
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float).cuda()

    # 计算相机x轴方向
    x_axis = torch.cross(up, z_axis)
    x_axis = x_axis / torch.norm(x_axis)

    # 计算相机y轴方向
    y_axis = torch.cross(z_axis, x_axis)

    # 构建旋转矩阵R
    R = torch.stack((x_axis, y_axis, z_axis), dim=1)

    # 构建平移向量t
    t = P

    # 构建外参矩阵T
    extrinsics = torch.eye(4, dtype=torch.float)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = t

    return extrinsics
def invert_camera_extrinsics(extrinsics):
    # 提取旋转矩阵和平移向量
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]

    # 旋转矩阵的转置是其逆
    R_inv = R.t()

    # 计算新的平移向量
    t_inv = -R_inv @ t

    # 构建逆矩阵
    extrinsics_inv = torch.eye(4, dtype=torch.float)
    extrinsics_inv[:3, :3] = R_inv
    extrinsics_inv[:3, 3] = t_inv

    return extrinsics_inv

def main(_):
    with torch.no_grad():
        opts = get_config()
        render(opts, construct_batch_func=construct_batch_from_opts)


if __name__ == "__main__":
    
    app.run(main)
