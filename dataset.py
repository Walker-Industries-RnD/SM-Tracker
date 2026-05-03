from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TF
from torchvision.io import decode_image
from synthetic import convert, CROP_SIZE, RANDOM_CROP, EVAL_CROP_XY

import cv2
import numpy as np
import random
import torch
import glob, os, json

# ── Image transforms ──────────────────────────────────────────────────────────
BORDER_PAD = 4

# Shared normalization (no augmentation) — used for eval and as the final step in training.
to_gray_tensor = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5]),
])

_photometric = v2.Compose([
    v2.ColorJitter(brightness=0.225, contrast=0.225),
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5]),
])


class SyntheticTransform:
    """
    center_crop(CROP_SIZE) → RandomCrop(RANDOM_CROP) → RandomHorizontalFlip
        → photometric augmentation → zero-pad(BORDER_PAD).

    Returns (img_tensor, crop_xy, flipped).  crop_xy = (left, top) is the
    offset of the random crop within the CROP_SIZE center-crop region;
    pass it directly to convert() / convert_parquet() so GT heatmaps stay
    spatially aligned.

    With BORDER_PAD > 0, model input is (RANDOM_CROP + 2*BORDER_PAD)² so the
    decoder's shallowest skip — and therefore heatmap logits — will be slightly
    larger than heatmap_hw.  The model center-crops logits to heatmap_hw before
    computing loss.
    """
    def __call__(self, img):
        img = TF.center_crop(img, [CROP_SIZE, CROP_SIZE])
        top, left, _, _ = v2.RandomCrop.get_params(img, (RANDOM_CROP, RANDOM_CROP))
        img = TF.crop(img, top, left, RANDOM_CROP, RANDOM_CROP)
        flipped = random.random() < 0.5
        if flipped:
            img = TF.horizontal_flip(img)
        img = _photometric(img)
        img = TF.pad(img, BORDER_PAD)
        return img, (left, top), flipped


synth_transforms = SyntheticTransform()

synth_dir = '/home/john/Downloads/synthetic_v2/'


def parse_tuples(obj):
    it = obj.items() if type(obj) is dict else enumerate(obj)
    for key, value in it:
        if type(value) is str and value[0] == '(' and value[-1] == ')':
            items = value[1:-1].split(', ')
            obj[key] = tuple(float(item) for item in items)
        elif type(value) in (list, dict):
            obj[key] = parse_tuples(value)
        elif type(value) == str:
            try:
                obj[key] = float(value)
            except ValueError:
                pass
    return obj


class SyntheticDS(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        self.img_dir = sorted(
            glob.glob(os.path.join(synth_dir, 'images', '*.jpg')),
            key=lambda x: int(os.path.basename(x).removesuffix('.jpg')),
        )
        self.lbl_dir = sorted(
            glob.glob(os.path.join(synth_dir, 'labels', '*.json')),
            key=lambda x: int(os.path.basename(x).removesuffix('.json')),
        )

    def __len__(self):
        return min(len(self.img_dir), len(self.lbl_dir))

    def __getitem__(self, idx):
        img = decode_image(self.img_dir[idx])
        with open(self.lbl_dir[idx], 'r') as f:
            lbl = json.load(f, object_hook=parse_tuples)

        if self.transforms is not None:
            img, crop_xy, flipped = self.transforms(img)
        else:
            # Deterministic eval preprocessing — canonical center crop, no augmentation.
            img = TF.center_crop(img, [CROP_SIZE, CROP_SIZE])
            img = TF.center_crop(img, [RANDOM_CROP, RANDOM_CROP])
            img = to_gray_tensor(img)
            img = TF.pad(img, BORDER_PAD)
            crop_xy = EVAL_CROP_XY
            flipped = False

        return img, convert(lbl, crop_xy=crop_xy, flipped=flipped)
