# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python scripts/render_intermediate.py --testdir logdir/human-48-category-comp/
import sys, os
import pdb

os.environ["PYOPENGL_PLATFORM"] = "egl"  # opengl seems to only work with TPU
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import numpy as np
import cv2
import argparse
import trimesh
import pyrender
from pyrender import IntrinsicsCamera, Mesh, Node, Scene, OffscreenRenderer
import matplotlib
import tqdm

from lab4d.utils.io import save_vid

cmap = matplotlib.colormaps.get_cmap("cool")

parser = argparse.ArgumentParser(description="script to render cameras over epochs")
parser.add_argument("--testdir", default="", help="path to test dir")
parser.add_argument(
    "--data_class", default="fg", type=str, help="which data to render, {fg, bg}"
)
args = parser.parse_args()

img_size = 1024

# renderer
r = OffscreenRenderer(img_size, img_size)
cam = IntrinsicsCamera(img_size, img_size, img_size / 2, img_size / 2)
# light
direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
light_pose = np.asarray(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float
)
# cv to gl coords
cam_pose = -np.eye(4)
cam_pose[0, 0] = 1
cam_pose[-1, -1] = 1
rtmat = np.eye(4)
# object to camera transforms
rtmat[:3, :3] = cv2.Rodrigues(np.asarray([np.pi / 2, 0, 0]))[0]  # bev


def main():
    # io
    path_list = [
        i for i in glob.glob("%s/*-%s-proxy.obj" % (args.testdir, args.data_class))
    ]
    if len(path_list) == 0:
        print("no mesh found in %s for %s" % (args.testdir, args.data_class))
        return
    path_list = sorted(path_list, key=lambda x: int(x.split("/")[-1].split("-")[0]))
    outdir = "%s/renderings_proxy" % args.testdir
    os.makedirs(outdir, exist_ok=True)

    mesh_dict = {}
    aabb_min = np.asarray([np.inf, np.inf, np.inf])
    aabb_max = np.asarray([-np.inf, -np.inf, -np.inf])
    for mesh_path in path_list:
        batch_idx = int(mesh_path.split("/")[-1].split("-")[0])
        mesh_obj = trimesh.load(mesh_path)
        mesh_dict[batch_idx] = mesh_obj

        # update aabb
        aabb_min = np.minimum(aabb_min, mesh_obj.bounds[0])
        aabb_max = np.maximum(aabb_max, mesh_obj.bounds[1])

    # set camera translation
    rtmat[2, 3] = max(aabb_max - aabb_min) * 1.2

    # render
    frames = []
    for batch_idx, mesh_obj in tqdm.tqdm(mesh_dict.items()):
        scene = Scene(ambient_light=0.4 * np.asarray([1.0, 1.0, 1.0, 1.0]))

        # add object / camera
        mesh_obj.apply_transform(rtmat)
        scene.add_node(Node(mesh=Mesh.from_trimesh(mesh_obj)))

        # camera
        scene.add(cam, pose=cam_pose)

        # light
        scene.add(direc_l, pose=light_pose)

        # render
        color, depth = r.render(
            scene,
            flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL
            | pyrender.RenderFlags.SKIP_CULL_FACES,
        )
        # add text
        color = color.astype(np.uint8)
        color = cv2.putText(
            color,
            "batch: %02d" % batch_idx,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (256, 0, 0),
            2,
        )
        frames.append(color)

    save_vid("%s/fg" % outdir, frames, suffix=".mp4", upsample_frame=-1)
    print("saved to %s/fg.mp4" % outdir)


if __name__ == "__main__":
    main()
