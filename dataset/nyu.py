import os, sys
from typing import *
import cv2
from PIL import Image
from rich.progress import track
from glob import glob
import numpy as np
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

_root = "/".join(__file__.split("/")[:-1]) + "/source/nyuv2"


class NYUv2(Dataset):
    def __init__(self, root = _root, args:argparse = None):
        print("Data Set Setting Up")
        self.root = root
        self.args = args
        if self.args is None:
            raise ValueError("args cannot be None")
        self.img_paths = sorted(glob(self.root + "/*/image/*"))
        self.msk_paths = sorted(glob(self.root + "/*/label/*"))
        self.dpt_paths = sorted(glob(self.root + "/*/depth/*"))
        self.nor_paths = sorted(glob(self.root + "/*/normal/*"))

        assert len(self.img_paths) == len(self.msk_paths) == len(self.dpt_paths) == len(self.nor_paths)
        print("Done")

    @staticmethod
    def make_semantic_class(x):
        onehot = F.one_hot(x.unsqueeze(0).squeeze(1), 14)
        onehot = onehot.permute(0, 3, 1, 2)[0].float()
        return onehot

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        msk_path = self.msk_paths[idx]
        dpt_path = self.dpt_paths[idx]
        nor_path = self.nor_paths[idx]

        img = np.load(img_path)
        img_torch = torch.from_numpy(img).permute(-1, 0, 1).float()
        if self.args.task == 'seg':
            msk = np.load(msk_path)
            msk_torch = torch.from_numpy(msk + 1).long()
            msk_onehot = self.make_semantic_class(msk_torch)
            target = msk_onehot
        elif self.args.task == 'depth':
            dpt = np.load(dpt_path)
            dpt_torch = torch.from_numpy(dpt).permute(-1, 0, 1)
            target = dpt_torch
        elif self.args.task == 'normal':
            nor = np.load(nor_path)
            nor_torch = torch.from_numpy(nor).permute(-1, 0, 1)
            target = nor_torch

        return img_torch, target