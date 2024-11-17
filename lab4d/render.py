# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python scripts/render.py --seqname --flagfile=logdir/cat-0t10-fg-bob-d0-long/opts.log --load_suffix latest

import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from absl import app, flags
import imageio

import warnings
warnings.filterwarnings("ignore")

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
from lab4d.utils.quat_transform import quaternion_translation_to_se3
from lab4d.utils.geom_utils import K2inv, K2mat, mat2K
from lab4d.utils.io import make_save_dir, save_rendered
from lab4d.utils.profile_utils import torch_profile

cudnn.benchmark = True


class RenderFlags:
    """Flags for the renderer."""

    flags.DEFINE_integer("inst_id", 0, "video/instance id")
    flags.DEFINE_integer("motion_id", 1, "video/instance id")
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
    # import pdb;pdb.set_trace()
    raw_size = data_info["raw_size"][video_id]  # full range of pixels
    # ref video length
    vid_length = data_utils.get_vid_length(video_id, data_info)

    save_dir = make_save_dir(
        opts, sub_dir="renderings_%04d/%s" % (opts["inst_id"], opts["viewpoint"])
    )

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
    print("rendering frames: %s from video %d" % (str(frameid_sub), video_id))
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
        crop2raw[:, 0] = raw_size[1] / opts["render_res"]
        crop2raw[:, 1] = raw_size[0] / opts["render_res"]
        # crop2raw[:, 0] = 1
        # crop2raw[:, 1] = 1
        camera_int = mat2K(K2inv(crop2raw) @ K2mat(intrinsics_fr))
        crop2raw = None
    elif opts["viewpoint"].startswith("rot"):
        # rotate around field, format: rot-evelvation-degree
        elev, max_angle = [int(val) for val in opts["viewpoint"].split("_")[1:]]

        # bg_to_cam
        obj_size = (aabb["fg"][1, :] - aabb["fg"][0, :]).max()
        cam_traj = get_rotating_cam(
            len(frameid_sub), distance=obj_size * opts["rot_dist"], max_angle=max_angle
        )
        print(obj_size * opts["rot_dist"])
        # np.save('%s/cam_traj.npy' % save_dir, cam_traj)
        cam_elev = get_object_to_camera_matrix(elev, [1, 0, 0], 0)[None]
        cam_traj = cam_traj @ cam_elev
        # np.save('%s/cam_traj_with_elevation.npy' % save_dir, cam_traj)
        field2cam = create_field2cam(cam_traj, field2cam_fr.keys())
        # np.save('%s/field2cam.npy' % save_dir, field2cam)

        camera_int = np.zeros((len(frameid_sub), 4))

        # focal length = img height * distance / obj height
        camera_int[:, :2] = opts["render_res"] * 2 * 0.8  # zoom out a bit
        # camera_int[:, :2] = opts["render_res"] * 2.5
        camera_int[:, 2:] = opts["render_res"] / 2
        raw_size = (640, 640)  # full range of pixels
        crop2raw = None
    elif opts["viewpoint"].startswith("bev"):
        elev = int(opts["viewpoint"].split("-")[1])
        # render bird's eye view
        if "bg" in field2cam_fr.keys():
            # get bev wrt first frame image
            # center_to_bev = centered_to_camt0 x centered_to_rotated x camt0_to_centered x bg_to_camt0
            center_to_bev = get_object_to_camera_matrix(elev, [1, 0, 0], 0)[None]
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
            field2cam = {"fg": get_bev_cam(field2cam_fr["fg"], elev=elev)}
        else:
            raise NotImplementedError

        camera_int = np.zeros((len(frameid_sub), 4))
        camera_int[:, :2] = opts["render_res"] * 2
        camera_int[:, 2:] = opts["render_res"] / 2
        raw_size = (640, 640)  # full range of pixels
        crop2raw = None
    elif opts["viewpoint"].startswith("refrot"):
        elev, max_angle = [int(val) for val in opts["viewpoint"].split("_")[1:]]
        field2cam = None
        # import pdb;pdb.set_trace()
        obj_size = (aabb["fg"][1, :] - aabb["fg"][0, :]).max()
        index_frameid = frame_mapping[frame_offset[0] : frame_offset[1]]
        # import pdb;pdb.set_trace()
        if frameid.shape[0] < 124:
            index_frameid = np.linspace(frame_offset[0], frame_offset[1], frameid.shape[0], dtype=int)
        # for category, field in model.fields.field_params.items():
        #     ref_transform = field.camera_mlp.get_vals(frameid)
        #     ave_rot = torch.mean(ref_transform[0],dim=0)
        #     ave_tran = torch.mean(ref_transform[1],dim=0)
        #     ref_matrix = quaternion_translation_to_se3(ave_rot, ave_tran).cpu().detach().numpy()
        for category, field in model.fields.field_params.items():
            ref_transform = field.camera_mlp.get_vals(index_frameid)
        distance = np.sqrt(np.sum(field2cam_fr['fg'][0,:3,3]**2))
        # import pdb;pdb.set_trace()
        cam_traj = quaternion_translation_to_se3(ref_transform[0], ref_transform[1]).cpu().detach().numpy()
        field2cam = {'fg': cam_traj}
        # field2cam = {'fg': ref_matrix @ cam_traj}
        #   field2cam=None
        # np.save('%s/field2cam.npy' % save_dir, field2cam)
        
        # field2cam = create_field2cam(cam_traj, field2cam_fr.keys())
        # camera_int = None
        crop2raw = np.zeros((len(frameid_sub), 4))
        crop2raw[:, 0] = raw_size[1] / opts["render_res"]
        crop2raw[:, 1] = raw_size[0] / opts["render_res"]
        # crop2raw[:, 0] = 1
        # crop2raw[:, 1] = 1
        camera_int = mat2K(K2inv(crop2raw) @ K2mat(intrinsics_fr))
        crop2raw = None

    elif opts["viewpoint"].startswith("novel"):
        elev, max_angle = [int(val) for val in opts["viewpoint"].split("_")[1:]]
        field2cam = None
        # import pdb;pdb.set_trace()
        obj_size = (aabb["fg"][1, :] - aabb["fg"][0, :]).max()
        index_frameid = frame_mapping[frame_offset[0] : frame_offset[1]]
        # index_frameid = torch.full_like(index_frameid, int(124 / 360 * max_angle), dtype=int)
        index_frameid = [int(124 / 360 * max_angle) for _ in index_frameid]
        # import pdb;pdb.set_trace()
        if frameid.shape[0] < 124:
            index_frameid = np.linspace(int(124 / 360 * max_angle), int(124 / 360 * max_angle) + 1, frameid.shape[0], dtype=int)
        # for category, field in model.fields.field_params.items():
        #     ref_transform = field.camera_mlp.get_vals(frameid)
        #     ave_rot = torch.mean(ref_transform[0],dim=0)
        #     ave_tran = torch.mean(ref_transform[1],dim=0)
        #     ref_matrix = quaternion_translation_to_se3(ave_rot, ave_tran).cpu().detach().numpy()
        for category, field in model.fields.field_params.items():
            ref_transform = field.camera_mlp.get_vals(index_frameid)
        # distance = np.sqrt(np.sum(field2cam_fr['fg'][0,:3,3]**2))
        
        # import pdb;pdb.set_trace()
        cam_traj = quaternion_translation_to_se3(ref_transform[0], ref_transform[1] * 1.2).cpu().detach().numpy()
        field2cam = {'fg': cam_traj}
        # field2cam = {'fg': ref_matrix @ cam_traj}
        #   field2cam=None
        # np.save('%s/field2cam.npy' % save_dir, field2cam)
        
        # field2cam = create_field2cam(cam_traj, field2cam_fr.keys())
        # camera_int = None
        crop2raw = np.zeros((len(frameid_sub), 4))
        crop2raw[:, 0] = raw_size[1] / opts["render_res"]
        crop2raw[:, 1] = raw_size[0] / opts["render_res"]
        # crop2raw[:, 0] = 1
        # crop2raw[:, 1] = 1
        camera_int = mat2K(K2inv(crop2raw) @ K2mat(intrinsics_fr))
        crop2raw = None
    else:
        raise ValueError("Unknown viewpoint type %s" % opts.viewpoint)

    batch = construct_batch(
        inst_id=opts["motion_id"],
        frameid_sub=frameid_sub,
        eval_res=opts["render_res"],
        field2cam=field2cam,
        camera_int=camera_int,
        crop2raw=crop2raw,
        device=device,
    )
    return batch, raw_size


@torch.no_grad()
def render_batch(model, batch, nowarp=False):
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
        batch["H"] = (torch.ones_like(batch['frameid_sub']) * opts["render_res"]).long()
        batch["W"] = (torch.ones_like(batch['frameid_sub']) * opts["render_res"]).long()

    save_dir = make_save_dir(
        opts, sub_dir="renderings_%04d/%s" % (opts["motion_id"], opts["viewpoint"])
    )

    # render
    with torch_profile(save_dir, "profile", enabled=opts["profile"]):
        with torch.no_grad():
            rendered = render_batch(model, batch, nowarp=nowarp)

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
    # print(rendered.keys())
    
    print(rendered.keys())
    # print(rendered['rend_normal'].shape, rendered['rend_normal'].min(), rendered['rend_normal'].max())
    # print(rendered['surf_depth'].shape, rendered['surf_depth'].min(), rendered['surf_depth'].max(), rendered['render_depth_median'].shape, rendered['render_depth_median'].min(), rendered['render_depth_median'].max())
    
    # surf_depth = np.float16(rendered['surf_depth'].cpu().numpy())
    # render_depth_median = np.float16(rendered['render_depth_median'].cpu().numpy())
    # render_depth_expected = np.float16(rendered['render_depth_expected'].cpu().numpy())
    
    rgb = np.float16(rendered['rgb'].cpu().numpy())
    rend_normal = np.float16(rendered['rend_normal'].cpu().numpy())
    surf_normal = np.float16(rendered['surf_normal'].cpu().numpy())
    normal_mask = np.abs(surf_normal - rend_normal)
    # np.save('%s/surf_depth.npy' % save_dir, surf_depth)
    # np.save('%s/render_depth_median.npy' % save_dir, render_depth_median)
    # np.save('%s/render_depth_expected.npy' % save_dir, render_depth_expected)
    torch.save({"rgb":rgb, "mask":normal_mask}, '%s/rgb.pth' % save_dir)

    save_rendered(rendered, save_dir, (opts["render_res"],opts["render_res"]), data_info["apply_pca_fn"])

    # rendered['surf_depth'] = normalize(rendered['surf_depth'])
    # rendered['render_depth_median'] = normalize(rendered['render_depth_median'])
    # rendered['render_depth_expected'] = normalize(rendered['render_depth_expected'])

    # surf_depth = np.uint8(rendered['surf_depth'].cpu().numpy())
    # render_depth_median = np.uint8(rendered['render_depth_median'].cpu().numpy())
    # render_depth_expected = np.uint8(rendered['render_depth_expected'].cpu().numpy())
    # images = [frame for frame in surf_depth]
    # print('%s/surf_depth.mp4' % save_dir)
    # imageio.mimsave('%s/surf_depth.mp4' % save_dir, images, fps=30)
    # images = [frame for frame in render_depth_median]
    # imageio.mimsave('%s/render_depth_median.mp4' % save_dir, images, fps=30)
    # images = [frame for frame in render_depth_expected]
    # imageio.mimsave('%s/render_depth_expected.mp4' % save_dir, images, fps=30)
    
    # print("Saved to %s" % save_dir)


def normalize(t):
    # return ((t - t.min()) / (t.max() - t.min()) - 0.5) * 2
    return (t - t.min()) / (t.max() - t.min()) * 255

def main(_):
    opts = get_config()
    render(opts, construct_batch_func=construct_batch_from_opts)


if __name__ == "__main__":
    
    app.run(main)

