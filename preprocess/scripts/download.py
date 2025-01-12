# Usage:
# python preprocess/scripts/download.py <seqname>
import os, sys
import shutil
import subprocess
import zipfile


def download_seq(seqname):
    datadir = os.path.join("database", "raw", seqname)
    if os.path.exists(datadir):
        print(f"Deleting existing directory: {datadir}")
        shutil.rmtree(datadir)

    url_path = os.path.join("database", "vid_data", f"{seqname}.txt")
    if not os.path.exists(url_path):
        # specify the folder of videos
        print(f"URL file does not exist: {url_path}")
        # ask for user input
        vid_path = "video_folder"
        while not os.path.isdir(vid_path):
            vid_path = input("Enter the path to video folder:")
        # copy folder to datadir
        print(f"Copying from directory: {vid_path} to {datadir}")
        shutil.copytree(vid_path, datadir)
    else:
        with open(url_path, "r") as f:
            url = f.read().strip()

        # Download the video
        print(f"Downloading from URL: {url}")
        tmp_zip = "tmp-%s.zip" % seqname
        subprocess.run(
            ["wget", url, "-O", tmp_zip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Unzip the file
        os.makedirs(datadir)
        print(f"Unzipping to directory: {datadir}")
        with zipfile.ZipFile(tmp_zip, "r") as zip_ref:
            zip_ref.extractall(datadir)

        # Remove the zip file
        os.remove(tmp_zip)


def main():
    # Get sequence name from command line arguments
    if len(sys.argv) > 1:
        seqname = sys.argv[1]
        download_seq(seqname)
    else:
        print("Usage: python preprocess/scripts/download.py <seqname>")


if __name__ == "__main__":
    main()
