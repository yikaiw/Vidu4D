# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python preprocess/scripts/compute_diff.py database/processed/JPEGImages/Full-Resolution/cat-pikachu-0000/ database/processed/JPEGImages/Full-Resolution/2023-04-19-01-36-53-cat-pikachu-0000/
import glob
import sys

import cv2
import numpy as np

path1 = sys.argv[1]
path2 = sys.argv[2]

for path1, path2 in zip(
    sorted(glob.glob(path1 + "/*")), sorted(glob.glob(path2 + "/*"))
):
    print(path1, path2)

    if path1.endswith(".npy"):
        t1 = np.load(path1).astype(np.float32)
        t2 = np.load(path2).astype(np.float32)
    elif path1.endswith(".jpg"):
        t1 = cv2.imread(path1).astype(np.float32)
        t2 = cv2.imread(path2).astype(np.float32)
    elif path1.endswith(".txt"):
        t1 = np.loadtxt(path1)
        t2 = np.loadtxt(path2)
    else:
        raise NotImplementedError

    print(np.mean(np.abs(t1 - t2)))
