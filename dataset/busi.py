import os, sys
from typing import *
import cv2
from PIL import Image
from rich.progress import track
import numpy as np
import argparse
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torch.nn.functional as F
from torch import Tensor
import albumentations as A

_root = "/".join(__file__.split("/")[:-1]) + "/source/BUSI"

class CustomBusi(Dataset):
    def __init__(self, root:str = _root, split='train', args:argparse = None):
        self.root = root
        self.args = args
        if self.args is None:
            raise ValueError("args cannot be None")
        self._split = split
        self.__mode = "train" if split == 'train' else 'test'
        
        self.resize = A.Compose(
            [
                A.Resize(256, 256),
            ]
        )

        self.aug_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=3, p=0.5, border_mode = cv2.BORDER_CONSTANT),
            ]
        )

        self.norm = A.Compose(
            [
                A.ToFloat(max_value=255),
            ]
        )

        self._images = sorted(glob(self.root+ "/images/*"))
        self._segs = sorted(glob(self.root+ "/masks/*_mask.png"))
        
        
        print("Data Set Setting Up")
        print(len(self._images),len(self._segs))

    def __len__(self):
        return len(self._images)
    def __getitem__(self, idx):
        image = np.array(Image.open(self._images[idx]).convert("RGB"))
        mask = np.array(Image.open(self._segs[idx]))
        image = image.astype(np.uint8)
        mask = mask.astype(np.uint8)


        resized = self.resize(image = image, mask = mask)

        if self.__mode == 'train':
            transformed = self.aug_transforms(image = resized['image'], mask = resized['mask'])
            transformed_img = self.norm(image=transformed["image"])["image"]
            transformed_mask = transformed["mask"]
        else:
            transformed_img = self.norm(image=resized['image'])['image']
            transformed_mask = resized['mask']

        torch_img = torch.from_numpy(transformed_img).permute(-1, 0, 1).float()
        torch_mask = torch.from_numpy(transformed_mask).unsqueeze(-1).permute(-1, 0, 1).float()

        torch_mask[torch_mask > 1] = 1

        return torch_img, torch_mask

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, m):
        if m not in ['train', 'test']:
            raise ValueError(f"mode cannot be {m} and must be ['train', 'test']")
        else:
            self.__mode = m