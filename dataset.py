from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TF
from torchvision.io import decode_image
from synthetic import convert

import cv2
import numpy as np
import random

import torch
import glob, os, json

_photometric = v2.Compose([
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5]),
])

class SyntheticTransform:
    """
    Resize(256) → RandomCrop(224) → RandomHorizontalFlip → photometric.

    Returns (img_tensor, crop_xy, flipped) so __getitem__ can forward
    crop_xy and flipped to convert() to keep GT heatmaps spatially aligned.
    """
    def __call__(self, img):
        img = TF.resize(img, [256, 256], antialias=True)
        top, left, _, _ = v2.RandomCrop.get_params(img, (224, 224))
        img = TF.crop(img, top, left, 224, 224)
        flipped = random.random() < 0.5
        if flipped:
            img = TF.horizontal_flip(img)
        return _photometric(img), (left, top), flipped   # crop_xy = (x_offset, y_offset)

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
        self.img_dir = sorted(glob.glob(os.path.join(synth_dir, 'images', '*.jpg')), key=lambda x: int(os.path.basename(x).removesuffix('.jpg')))
        self.lbl_dir = sorted(glob.glob(os.path.join(synth_dir, 'labels', '*.json')), key=lambda x: int(os.path.basename(x).removesuffix('.json')))

    def __len__(self):
        return min(len(self.img_dir), len(self.lbl_dir))

    def __getitem__(self, idx):
        img = decode_image(self.img_dir[idx])
        with open(self.lbl_dir[idx], 'r') as lbl_file:
            lbl = json.load(lbl_file, object_hook=parse_tuples)

        crop_xy  = (0, 0)
        flipped  = False
        resize   = (224, 224)

        if self.transforms is not None:
            img, crop_xy, flipped = self.transforms(img)
            resize = (256, 256)

        return img, convert(lbl, crop_xy=crop_xy, flipped=flipped, resize=resize)
