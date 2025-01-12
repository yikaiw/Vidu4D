# python preprocess/third_party/Track-Anything/track_anything_cli.py --input-folder database/processed/JPEGImages/Full-Resolution/finch-0000/ --text-prompt "bird"
import sys
import os
import pdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__)) + "/")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)) + "/tracker")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)) + "/tracker/model")


import torch
from pathlib import Path
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import imageio

from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import SamPredictor, sam_model_registry
from tracker.base_tracker import BaseTracker
from app import download_checkpoint, wget_checkpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"

root_dir = "./preprocess/third_party/Track-Anything/"
folder = "%s/checkpoints" % root_dir


sam_checkpoint = "sam_vit_h_4b8939.pth"
sam_checkpoint_url = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)
xmem_checkpoint = "XMem-s012.pth"
xmem_checkpoint_url = (
    "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
)

gdino_checkpoint = "groundingdino_swint_ogc.pth"
gdino_checkpoint_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

model_config = {
    "DINO": "./preprocess/third_party/Track-Anything/config/GroundingDINO_SwinT_OGC.py"
}
model_weights = {
    "DINO": wget_checkpoint(gdino_checkpoint_url, folder, gdino_checkpoint),
    "SAM": download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint),
    "XMEM": download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint),
}


def extract_bbox(model, img_path, text_prompt, BOX_THRESHOLD=0.35, TEXT_THRESHOLD=0.25):
    image_source, image = load_image(img_path)
    H, W = image_source.shape[0], image_source.shape[1]

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    annotated_frame = annotate(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
    )

    boxes = boxes * torch.Tensor([W, H, W, H]).repeat(repeats=(boxes.shape[0], 1))

    # from xywh to xyxy
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    boxes = boxes.type("torch.IntTensor")

    # make sure the bbox are sorted by confidence
    boxes = boxes[torch.argsort(logits, descending=True)]
    # TODO enable multi object tracking
    boxes = boxes[:1]
    return boxes, annotated_frame


def extract_mask(model, img_path, boxes):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    track_init = np.zeros((image.shape[0], image.shape[1]))

    model.set_image(image)
    transformed_boxes = model.transform.apply_boxes_torch(
        boxes.to(DEVICE), image.shape[:2]
    )
    masks, _, _ = model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    for i, mask in enumerate(masks):
        track_init = track_init + mask[0].cpu().numpy() * (i + 1)

    return track_init


def extract_tracks(xmem_model, template_mask, images):
    def generator(xmem_model, template_mask, images):
        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images))):
            if i == 0:
                mask, logit, painted_image = xmem_model.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)

            else:
                mask, logit, painted_image = xmem_model.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images

    return generator(xmem_model, template_mask, images)


def isImageFile(str):
    import re

    # Regex to check valid image file extension.
    regex = "([^\\s]+(\\.(?i)(jpe?g|png|gif|bmp))$)"

    # Compile the ReGex
    p = re.compile(regex)

    # If the string is empty
    # return false
    if str == None:
        return False

    # Return if the string
    # matched the ReGex
    if re.search(p, str):
        return True
    else:
        return False


@torch.no_grad()
def track_anything_cli(
    input_folder,
    text_prompt,
    output_folder="",
    BOX_THRESHOLD=0.35,
    TEXT_THRESHOLD=0.25,
):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Processing %s" % input_folder)

    if output_folder == "":
        output_folder = input_folder.replace("JPEGImages", "Annotations")

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    original_size = (None, None)
    dim = (None, None)
    images, image_paths, image_exts = [], [], []
    for img in sorted(os.listdir(input_folder)):
        if not isImageFile(img):
            continue

        img_path = input_folder + "/" + "/" + img
        img_ext = str(Path(img).with_suffix(""))

        img = cv2.imread(img_path)

        if len(images) == 0:
            original_size = (img.shape[1], img.shape[0])
            # make sure the total pixel is less than (800x800)
            scale_percent = np.sqrt(640000 / (original_size[0] * original_size[1]))
            print("scale_percent", scale_percent)
            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            dim = (width, height)

        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        images.append(img)
        image_paths.append(img_path)
        image_exts.append(img_ext)

    dino_model = load_model(model_config["DINO"], model_weights["DINO"]).to(DEVICE)
    sam_model = SamPredictor(
        sam_model_registry["vit_h"](checkpoint=model_weights["SAM"]).to(DEVICE)
    )
    xmem_model = BaseTracker(model_weights["XMEM"], device=DEVICE)

    boxes, annotated_frame = extract_bbox(
        dino_model, image_paths[0], text_prompt, BOX_THRESHOLD, TEXT_THRESHOLD
    )

    if len(boxes) == 0:
        masks = np.zeros((len(images), original_size[1], original_size[0]))
        painted_images = np.zeros((len(images), original_size[1], original_size[0], 3))
        painted_images = painted_images.astype(np.uint8)
        logits = np.zeros((len(images), 2))
        print("Detection failed on the first frame")
    else:
        track_init = extract_mask(sam_model, image_paths[0], boxes)

        track_init = cv2.resize(track_init, dim, interpolation=cv2.INTER_NEAREST)
        masks, logits, painted_images = extract_tracks(xmem_model, track_init, images)

    for mask, painted_img, img_ext, logits_per_frame in zip(
        masks, painted_images, image_exts, logits
    ):
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int8)
        # set to undetected if less than 100 pixels detected, or conf is too low
        if (mask > 0).sum() <= 100 or logits_per_frame[1:].max() < 0.9:
            mask[:] = -1
        np.save("{}/{}.npy".format(output_folder, img_ext), mask)

        painted_img = cv2.resize(
            painted_img, original_size, interpolation=cv2.INTER_AREA
        )
        cv2.imwrite("{}/{}.jpg".format(output_folder, img_ext), painted_img)

    painted_imgs_rgb = []
    for painted_img in painted_images:
        painted_imgs_rgb.append(painted_img[:, :, ::-1])
    imageio.mimsave("%s/vis.mp4" % output_folder, painted_imgs_rgb, fps=10)
    print("Segmentation output saved to %s" % output_folder)


if __name__ == "__main__":
    seqname = sys.argv[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, required=True)
    parser.add_argument("--output-folder", type=str, default="")
    parser.add_argument("--box-threshold", type=float, default=0.7)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--text-prompt", type=str, required=True)

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    text_prompt = args.text_prompt

    BOX_THRESHOLD = args.box_threshold
    TEXT_THRESHOLD = args.text_threshold
    track_anything_cli(
        input_folder,
        text_prompt,
        output_folder,
        BOX_THRESHOLD=BOX_THRESHOLD,
        TEXT_THRESHOLD=TEXT_THRESHOLD,
    )
