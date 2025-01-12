# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python preprocess/scripts/write_config.py ${vidname}
import configparser
import glob
import os
import sys

import cv2


def write_config(collection_name):
    min_nframe = 8
    imgroot = "database/processed/JPEGImages/Full-Resolution/"

    config = configparser.ConfigParser()
    config["data"] = {
        "init_frame": "0",
        "end_frame": "-1",
    }

    seqname_all = sorted(
        glob.glob("%s/%s-[0-9][0-9][0-9][0-9]*" % (imgroot, collection_name))
    )
    total_vid = 0
    for i, seqname in enumerate(seqname_all):
        seqname = seqname.split("/")[-1]
        img = cv2.imread("%s/%s/00000.jpg" % (imgroot, seqname), 0)
        num_fr = len(glob.glob("%s/%s/*.jpg" % (imgroot, seqname)))
        if num_fr < min_nframe:
            continue

        fl = max(img.shape)
        px = img.shape[1] // 2
        py = img.shape[0] // 2
        camtxt = [fl, fl, px, py]
        config["data_%d" % total_vid] = {
            "ks": " ".join([str(i) for i in camtxt]),
            "shape": " ".join([str(img.shape[0]), str(img.shape[1])]),
            "img_path": "database/processed/JPEGImages/Full-Resolution/%s/" % seqname,
        }
        total_vid += 1

    os.makedirs("database/configs", exist_ok=True)
    with open("database/configs/%s.config" % collection_name, "w") as configfile:
        config.write(configfile)


if __name__ == "__main__":
    collection_name = sys.argv[1]

    write_config(collection_name)
