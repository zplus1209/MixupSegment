# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import ImageFilter
import random
from collections import defaultdict
from engine.dataset.utils import classes


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# ==== CSV + Balanced-Mixup Pair Dataset =======================================
import os, os.path as osp
import pandas as pd
import numpy as np
from PIL import Image
import torch

def _ensure_ext(x):
    x = str(x)
    return x if x.lower().endswith(('.jpg', '.jpeg', '.png')) else x + '.jpg'



class StepLambdaScheduler:
    def __init__(
        self,
        total_epochs,
        steps=4, 
        lam_min=0.0, 
        lam_max=0.5,
        alpha_min=1e-4, 
        alpha_max=0.2):
        
        self.total_epochs = int(total_epochs)
        self.steps = int(steps)
        lam_min = float(lam_min); lam_max = float(lam_max)
        alpha_min = float(alpha_min); alpha_max = float(alpha_max)

        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if not (0.0 <= lam_min <= 0.5 and 0.0 <= lam_max <= 0.5):
            raise ValueError("lam_min/lam_max must be in [0, 0.5]")

        self.boundaries = [int(i/self.steps * self.total_epochs) for i in range(self.steps+1)]
        self.lams   = np.linspace(lam_max,  lam_min,  self.steps)   # cap λ: easy -> hard
        self.alphas = np.linspace(alpha_max, alpha_min, self.steps) # Beta α: easy -> hard
        self.stage = 0

    def set_epoch(self, epoch: int):
        self.stage = max([i for i, b in enumerate(self.boundaries[:-1]) if epoch >= b])

    def sample(self, batch_size: int) -> torch.Tensor:
        alpha   = float(self.alphas[self.stage])
        lam_cap = float(self.lams[self.stage])  # <= 0.5
        lam = np.random.beta(alpha, alpha, size=(batch_size, 1, 1, 1))
        lam = np.minimum(lam, lam_cap)
        return torch.from_numpy(lam.astype('float32'))
    
class EndoBalancedMixupDataset(torch.utils.data.Dataset):
    """
    Đọc CSV có cột 'image_id' (và 'finding' nếu muốn same-class pairing).
    Trả về ( [x1, x2], 0 ) cho SimSiam:
      - x1 = base_transform(img_i)
      - x2 = mixup view: x̃_j = λ * base_transform(img_i) + (1-λ) * base_transform(img_j)
    Với curriculum scheduler: λ ~ Beta(α,α), rồi kẹp λ ≤ lam_cap (≤ 0.5).
    """
    def __init__(self, csv_path, root_dir, base_transform,
                 alpha=0.4, same_class=True, scheduler=None):
        super().__init__()
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        if 'image_id' not in self.df.columns:
            raise ValueError("CSV phải có cột 'image_id'")
        self.paths = self.df['image_id'].apply(_ensure_ext).tolist()
        self.root = root_dir
        self.transform = base_transform
        self.alpha = float(alpha)
        self.scheduler = scheduler
        self.cur_epoch = 0
        self.return_raw_for_gpu = True  # trả về xi, xj cho GPU (để tính loss)
        
        # same-class pairing nếu có cột finding
        self.same_class = same_class and ('finding' in self.df.columns)
        if self.same_class:
            # hỗ trợ nhãn dạng text hoặc số
            try:
                labels = self.df['finding'].astype(int).tolist()
            except Exception:
                labels, uniques = pd.factorize(self.df['finding'])
                labels = labels.tolist()
            self.labels = labels

            self.class_to_idx = defaultdict(list)
            for idx, y in enumerate(self.labels):
                self.class_to_idx[int(y)].append(idx)
            for c, arr in self.class_to_idx.items():
                if len(arr) < 2:
                    raise ValueError(f"Class {c} có <2 mẫu, không thể tạo cặp same-class.")
            self.classes = sorted(self.class_to_idx.keys())
        else:
            self.labels = None
            self.class_to_idx = None
            self.classes = None

    def set_epoch(self, epoch: int):
        self.cur_epoch = epoch
        if self.scheduler is not None:
            self.scheduler.set_epoch(epoch)

    def __len__(self):
        return len(self.paths)

    def _load_rgb(self, idx: int) -> Image.Image:
        p = osp.join(self.root, self.paths[idx])
        with Image.open(p) as im:
            return im.convert('RGB')


    def _to_single_tensor(self, out):
        """
        Chuẩn hoá output của transform thành 1 tensor CxHxW.
        - Nếu transform trả (x1, x2) hoặc [x1, x2] -> lấy x1.
        - Nếu transform trả dict có 'image' -> lấy ['image'].
        - Nếu đã là tensor -> trả nguyên.
        """
        if isinstance(out, (list, tuple)):
            out = out[0]
        elif isinstance(out, dict) and 'image' in out:
            out = out['image']
        if not torch.is_tensor(out):
            raise TypeError(f"Transform must return a Tensor, got {type(out)}")
        return out
    
    def __getitem__(self, i: int):
        # anchor i
        img_i = self._load_rgb(i)

        # chọn j
        if self.same_class:
            y = int(self.labels[i])
            pool = self.class_to_idx[y]
            j = np.random.choice(pool)
            # tránh trùng i nếu lớp có >=2 mẫu
            while j == i and len(pool) > 1:
                j = np.random.choice(pool)
        else:
            j = np.random.randint(0, len(self.paths))
        img_j = self._load_rgb(int(j))

        # tạo 2 view + mixup
        xi = self._to_single_tensor(self.transform(img_i))   # view 1
        xj = self._to_single_tensor(self.transform(img_j))   # used in mixup

        if self.scheduler is None:
            lam = np.random.beta(self.alpha, self.alpha)
            lam = min(float(lam), 0.5)  # kẹp ≤ 0.5
            lam = torch.tensor(lam, dtype=xi.dtype).view(1, 1, 1)
        else:
            lam = self.scheduler.sample(1).squeeze(0).to(dtype=xi.dtype)  # (1,1,1)
        
        if self.return_raw_for_gpu:
            return {
                'xi': xi,
                'xj': xj,
                'lam': lam.view(1,1,1).to(dtype=xi.dtype),
            }, 0
        else:
            # công thức paper: x̃_j = λ * x_i + (1 - λ) * x_j  (j đóng góp ≥ 50%)
            x2 = lam * xi + (1.0 - lam) * xj
            # SimSiam expects (images, target) with images=[q, k]
            return [x1, x2], 0
    

class CSVDatasetWithLabel(torch.utils.data.Dataset):
    def __init__(self, csv_path, root_dir, transform):
        import pandas as pd, os.path as osp
        from PIL import Image
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        assert 'image_id' in self.df.columns, "CSV must have 'image_id'"
        assert 'finding' in self.df.columns,   "CSV must have 'finding' for kNN"
        self.paths = self.df['image_id'].tolist()
        try:
            self.labels = self.df['finding'].astype(int).tolist()
        except Exception:
            labels, _ = pd.factorize(self.df['finding'])
            self.labels = labels.tolist()
        
        self.classes = classes if hasattr(classes, '__iter__') else [str(c) for c in classes]
        self.targets = self.labels
        self.root = root_dir
        self.tf = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        # ép thêm .jpg (hoặc .png tuỳ dataset của bạn)
        p = os.path.join(self.root, _ensure_ext(self.paths[idx]))
        with Image.open(p) as im:
            x = im.convert("RGB")
        x = self.tf(x)
        y = int(self.labels[idx])
        return x, y
  