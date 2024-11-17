# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
from typing import NamedTuple

import cv2
import numpy as np
import torch

from lab4d.engine.trainer import Trainer
from lab4d.utils.geom_utils import K2inv
from lab4d.utils.quat_transform import se3_to_quaternion_translation


def create_field2cam(cam_traj, keys):
    """Create a dict containing camera trajectory for each field type

    Args:
        cam_traj: (N,3,4) Camera trajectory to view
        keys (List): Contains field names ("fg" or "bg")
    Returns:
        field2cam (Dict): Maps field names ("fg" or "bg") to (N,3,4) cameras
    """
    field2cam = {}
    if "bg" in keys and "fg" in keys:
        raise NotImplementedError
    elif "bg" in keys:
        field2cam["bg"] = cam_traj
    elif "fg" in keys:
        field2cam["fg"] = cam_traj
    else:
        raise NotImplementedError
    return field2cam


def get_bev_cam(field2cam, elev=90):
    """
    get a bird's eye view camera wrt view space object
    Args:
        field2cam: ndarray, N, 4, 4
    Returns:
        cam_traj: ndarray, N, 4, 4
    """
    # center2bev x camt2center x fg2camt
    ave_depth = field2cam[:, 2, 3].mean()
    center2cam = get_object_to_camera_matrix(0, [1, 0, 0], ave_depth)[None]
    center2bev = get_object_to_camera_matrix(elev, [1, 0, 0], 2 * ave_depth)[None]
    cam_traj = center2bev @ np.linalg.inv(center2cam) @ field2cam
    return cam_traj


def get_object_to_camera_matrix(theta, axis, distance):
    """Generate 3x4 object-to-camera matrix that rotates the object around
    the given axis

    Args:
        theta (float): Angle of rotation in radians.
        axis (ndarray): (3,) Axis of rotation
        distance (float): Distance from camera to object
    Returns:
        extrinsics (ndarray): (3, 4) Object-to-camera matrix
    """
    theta = theta / 180 * np.pi
    rt4x4 = np.eye(4)
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    R, _ = cv2.Rodrigues(theta * axis)
    t = np.asarray([0, 0, distance])
    rtmat = np.concatenate((R, t.reshape(3, 1)), axis=1)
    rt4x4[:3, :4] = rtmat
    return rt4x4


def get_rotating_cam(
    num_cameras, axis=[0, 1, 0], distance=3, initial_angle=0, max_angle=360
):
    """Generate camera sequence rotating around a fixed object

    Args:
        num_cameras (int): Number of cameras in sequence
        axis (ndarray): (3,) Axis of rotation
        distance (float): Distance from camera to object
        initial_angle (float): Initial rotation angle, degrees (default 0)
        max_angle (float): Final rotation angle, degrees (default 360)
    Returns:
        extrinsics (ndarray): (num_cameras, 3, 4) Sequence of camera extrinsics
    """
    angles = np.linspace(initial_angle, max_angle, num_cameras)
    extrinsics = np.zeros((num_cameras, 4, 4))
    for i in range(num_cameras):
        extrinsics[i] = get_object_to_camera_matrix(angles[i], axis, distance)
    return extrinsics


def get_fixed_cam(num_cameras, axis=[0, 1, 0], distance=3, angle=0):
    """Generate camera sequence relative to a fixed object at (0, 0, distance)

    Args:
        num_cameras (int): Number of cameras
        axis (ndarray): (3,) Axis of rotation
        distance (float): Distance from camera to object
        angle (float): Viewing angle, degrees
    Returns:
        extrinsics (ndarray): (num_cameras, 3, 4) Sequence of camera extrinsics
    """
    rshift, lshift = np.eye(4)[None], np.eye(4)[None]
    lshift[0, :3, 3] = np.asarray([0, 0, distance])
    rshift[0, :3, 3] = np.asarray([0, 0, -distance])
    extrinsics = get_rotating_cam(num_cameras, axis, 0, angle, angle)
    extrinsics = lshift @ extrinsics @ rshift
    return extrinsics


def get_orbit_camera(num_cameras, max_angle=5, cycles=2):
    """Generate camera sequence by rotating around a fixed object

    Args:
        num_cameras (int): Number of cameras in sequence
        max_angle (float): Maximum angle of rotation, degrees (default 5)
        cycles (int): Number of cycles of rotation (default 2)
    Returns:
        extrinsics: (num_cameras, 3, 4) Sequence of camera extrinsics
    """
    max_angle = max_angle / 180 * np.pi
    extrinsics = np.zeros((num_cameras, 4, 4))
    extrinsics[:, 3, 3] = 1
    for i in range(num_cameras):
        axis_angle = [
            max_angle * np.cos(cycles * 2 * np.pi * i / num_cameras),
            max_angle * np.sin(cycles * 2 * np.pi * i / num_cameras),
            0,
        ]
        extrinsics[i, :3, :3] = cv2.Rodrigues(np.asarray(axis_angle))[0]
    return extrinsics


class QueryBatch(NamedTuple):
    dataid: torch.Tensor
    frameid_sub: torch.Tensor
    crop2raw: torch.Tensor
    hxy: torch.Tensor
    field2cam: torch.Tensor
    Kinv: torch.Tensor


def construct_batch(
    inst_id,
    frameid_sub,
    eval_res,
    field2cam,
    camera_int,
    crop2raw,
    device,
):
    """Construct batch for rendering

    Args:
        inst_id (int): Video id
        frameid_sub: (N,) Frame ids in the video
        eval_res (int): Size of the rendered image
        field2cam (Dict or None): If provided, maps field type ("fg" or "bg")
            to (N,4,4) SE(3) camera transforms
        camera_int: If provided, (N,4) camera intrinsics (fx, fy, cx, cy)
        crop2raw: If provided, (N,4) parameters from cropped to raw images,
            (fx, fy, cx, cy)
        device (torch.device): Target device
    Returns:
        batch (Dict): Batch with keys: "frameid_sub" (N,), "dataid" (N,),
            "hxy" (N, H*W, 3), and "crop2raw" (N, 4)
    """
    batch = {}
    batch["frameid_sub"] = torch.tensor(frameid_sub, dtype=torch.long, device=device)
    batch["dataid"] = inst_id * torch.ones_like(batch["frameid_sub"])

    hxy = Trainer.create_xy_grid(eval_res, device)
    batch["hxy"] = hxy[None].expand(len(batch["dataid"]), -1, -1)

    if crop2raw is not None:
        batch["crop2raw"] = torch.tensor(crop2raw, dtype=torch.float32, device=device)

    if field2cam is not None:
        if isinstance(field2cam, dict):
            for k, v in field2cam.items():
                field2cam[k] = torch.tensor(v, dtype=torch.float32, device=device)
                field2cam[k] = se3_to_quaternion_translation(field2cam[k], tuple=False)
            batch["field2cam"] = field2cam
        else:
            batch["field2cam"] = {"fg": field2cam}

    if camera_int is not None:
        camera_int = torch.tensor(camera_int, dtype=torch.float32, device=device)
        batch["Kinv"] = K2inv(camera_int)

    return batch
