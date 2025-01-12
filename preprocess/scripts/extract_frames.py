# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python preprocess/scripts/extract_frames.py database/raw/cat-1015/10415567.mp4 tmp/
import sys

import imageio
import numpy as np


def extract_frames(in_path, out_path):
    print("extracting frames: ", in_path)
    # Open the video file
    reader = imageio.get_reader(in_path)

    # Find the first non-black frame
    for i, im in enumerate(reader):
        if np.any(im > 0):
            start_frame = i
            break

    # Write the video starting from the first non-black frame
    count = 0
    for i, im in enumerate(reader):
        if i >= start_frame:
            imageio.imsave("%s/%05d.jpg" % (out_path, count), im)
            count += 1


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    extract_frames(in_path, out_path)
