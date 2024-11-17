# python scripts/create_collage.py --testdir logdir/penguin-fg-skel-b120/ --prefix renderings_0002

from moviepy.editor import clips_array, VideoFileClip, vfx
import sys, os
import numpy as np
import pdb
import glob
import argparse
import itertools

parser = argparse.ArgumentParser(description="combine results into a collage")
parser.add_argument("--testdir", default="", help="path to test dir")
parser.add_argument(
    "--prefix", default="renderings_ref_", type=str, help="what data to combine"
)
args = parser.parse_args()


def main():
    save_path = "%s/collage.mp4" % args.testdir

    video_list = []
    for sub_seq in sorted(glob.glob("%s/%s*" % (args.testdir, args.prefix))):
        path_list = []
        path_list.append("%s/ref/ref_rgb.mp4" % sub_seq)
        path_list.append("%s/ref/rgb.mp4" % sub_seq)
        path_list.append("%s/ref/xyz.mp4" % sub_seq)
        path_list.append("%s/rot-0-360/rgb.mp4" % sub_seq)
        path_list.append("%s/rot-0-360/xyz.mp4" % sub_seq)

        # make sure these exist
        if np.sum([os.path.exists(path) for path in path_list]) == len(path_list):
            print("found %s" % sub_seq)
            video_list.append([VideoFileClip(path) for path in path_list])

    if len(video_list) == 0:
        print("no video found")
        return

    # align in time
    max_duration = max(
        [clip.duration for clip in list(itertools.chain.from_iterable(video_list))]
    )
    for i, clip_list in enumerate(video_list):
        for j, clip in enumerate(clip_list):
            video_list[i][j] = clip.resize(width=512).fx(
                vfx.freeze, t="end", total_duration=max_duration, padding_end=0.5
            )

    final_clip = clips_array(video_list)
    final_clip.write_videofile(save_path)


if __name__ == "__main__":
    main()
