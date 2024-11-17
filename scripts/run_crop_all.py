# WIP by Gengshan Yang
# TODO: use config file to go over seqs
# python scripts/run_crop_all.py cat-pikachu
import os
import sys
import glob
import multiprocessing
from functools import partial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocess.scripts.crop import extract_crop

os.environ["OMP_NUM_THREADS"] = "1"

vidname = sys.argv[1]
path = (
    "database/processed/JPEGImages/Full-Resolution/%s*" % vidname
)  # path to the images


def process_seqname(seqname, size, region):
    extract_crop(seqname, size, region)


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=32)  # use up to 32 processes

    for seqname in sorted(glob.glob(path)):
        seqname = seqname.split("/")[-1]
        # we'll use a partial function to bind the common arguments
        func = partial(process_seqname, seqname, 256)
        pool.apply_async(func, args=(0,))
        pool.apply_async(func, args=(1,))

    pool.close()
    pool.join()  # wait for all processes to finish
