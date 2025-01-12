# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python preprocess/scripts/tsdf_fusion.py 2023-04-03-18-02-32-cat-pikachu-5-0000 0
import glob
import os
import sys

import cv2
import numpy as np
import trimesh

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

import fusion
from libs.io import read_frame_data

from lab4d.utils.geom_utils import K2inv, K2mat
from lab4d.utils.vis_utils import draw_cams

# def read_cam(imgpath, component_id):
#     campath = imgpath.replace("JPEGImages", "Cameras").replace(
#         ".jpg", "-%02d.txt" % component_id
#     )
#     scene2cam = np.loadtxt(campath)
#     cam2scene = np.linalg.inv(scene2cam)
#     return cam2scene


def tsdf_fusion(seqname, component_id, crop_size=256, use_full=True):
    # load rgb/depth
    imgdir = "database/processed/JPEGImages/Full-Resolution/%s" % seqname
    imglist = sorted(glob.glob("%s/*.jpg" % imgdir))

    # camera path
    save_path = imgdir.replace("JPEGImages", "Cameras")
    save_path = "%s/%02d.npy" % (save_path, component_id)
    cams_prev = np.load(save_path)

    # get camera intrinsics
    raw_shape = cv2.imread(imglist[0]).shape[:2]
    max_l = max(raw_shape)
    Kraw = np.array([max_l, max_l, raw_shape[1] / 2, raw_shape[0] / 2])
    Kraw = K2mat(Kraw)

    # initialize volume
    vol_bnds = np.zeros((3, 2))
    for it, imgpath in enumerate(imglist[:-1]):
        rgb, depth, mask, crop2raw = read_frame_data(
            imgpath, crop_size, use_full, component_id
        )
        K0 = K2inv(crop2raw) @ Kraw
        # cam2scene = read_cam(imgpath, component_id)
        cam2scene = np.linalg.inv(cams_prev[it])
        depth[~mask] = 0
        depth[depth > 10] = 0
        view_frust_pts = fusion.get_view_frustum(depth, K0, cam2scene)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.2, use_gpu=False)

    # fusion
    for it, imgpath in enumerate(imglist[:-1]):
        # print(imgpath)
        rgb, depth, mask, crop2raw = read_frame_data(
            imgpath, crop_size, use_full, component_id
        )
        K0 = K2inv(crop2raw) @ Kraw
        depth[~mask] = 0
        # cam2scene = read_cam(imgpath, component_id)
        cam2scene = np.linalg.inv(cams_prev[it])
        tsdf_vol.integrate(rgb, depth, K0, cam2scene, obs_weight=1.0)

    save_path = imgdir.replace("JPEGImages", "Cameras")
    # get mesh, compute center
    rt = tsdf_vol.get_mesh()
    verts, faces = rt[0], rt[1]
    mesh = trimesh.Trimesh(verts, faces)
    aabb = mesh.bounds
    center = aabb.mean(0)
    mesh.vertices = mesh.vertices - center[None]
    mesh.export("%s/mesh-%02d-centered.obj" % (save_path, component_id))

    # save cameras
    cams = []
    for it, imgpath in enumerate(imglist):
        # campath = imgpath.replace("JPEGImages", "Cameras").replace(
        #     ".jpg", "-%02d.txt" % component_id
        # )
        # cam = np.loadtxt(campath)
        # shift the camera in the scene space
        cam = np.linalg.inv(cams_prev[it])
        cam[:3, 3] -= center
        cam = np.linalg.inv(cam)
        # np.savetxt(campath, cam)
        cams.append(cam)
    np.save("%s/%02d.npy" % (save_path, component_id), cams)
    mesh_cam = draw_cams(cams)
    mesh_cam.export("%s/cameras-%02d-centered.obj" % (save_path, component_id))

    print("tsdf fusion done: %s, %d" % (seqname, component_id))


if __name__ == "__main__":
    seqname = sys.argv[1]
    component_id = int(sys.argv[2])

    tsdf_fusion(seqname, component_id)
