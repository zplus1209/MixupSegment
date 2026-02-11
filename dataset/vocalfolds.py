import os, sys
from typing import *
import cv2
from PIL import Image
from rich.progress import track
import numpy as np
import argparse
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torch.nn.functional as F
from torch import Tensor
import albumentations as A

_root = "/".join(__file__.split("/")[:-1]) + "/source/vocalfolds"

class Customvocalfolds(Dataset):
    def __init__(self, root:str = _root, split = 'trainval', args:argparse = None):
        self.root = root
        self.args = args
        if self.args is None:
            raise ValueError("args cannot be None")
        self._split = split
        self.__mode = "train" if self._split == 'trainval' else 'test'       
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
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self._images = glob.glob(self.root+ "/img/*/*/*")
        self._segs = glob.glob(self.root+ "/annot/*/*/*")

        print("Data Set Setting Up")
        print(len(self._images),len(self._segs))

    @staticmethod
    def process_mask(x):
        uniques = torch.unique(x, sorted = True)
        if uniques.shape[0] > 3:
            x[x == 0] = uniques[2]
            uniques = torch.unique(x, sorted = True)
        for i, v in enumerate(uniques):
            x[x == v] = i
        
        x = x.to(dtype=torch.long)
        onehot = F.one_hot(x.squeeze(1), 7).permute(0, 3, 1, 2)[0].float()
        return onehot

    def __len__(self):
        return len(self._images)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self._images[idx]).convert("RGB"))
        mask = np.array(Image.open(self._segs[idx]))

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

        return torch_img, self.process_mask(torch_mask)

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, m):
        if m not in ['train', 'test']:
            raise ValueError(f"mode cannot be {m} and must be ['train', 'test']")
        else:
            self.__mode = m
