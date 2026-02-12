"""
HyperKvasir Dataset Loader
Supports both labeled and unlabeled data for semi-supervised learning
"""
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from config import (
    LABELED_IMAGES, LABELED_MASKS, UNLABELED_IMAGES,
    AUGMENTATION_CONFIG, TRAIN_CONFIG
)


class HyperKvasirDataset(Dataset):
    """Dataset cho labeled data của HyperKvasir"""
    
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # Binary mask
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': image,
            'mask': mask.unsqueeze(0) if len(mask.shape) == 2 else mask,
            'image_path': str(self.image_paths[idx])
        }


class HyperKvasirUnlabeledDataset(Dataset):
    """Dataset cho unlabeled data của HyperKvasir"""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'image_path': str(self.image_paths[idx])
        }


def get_transforms(mode='train'):
    """Get augmentation transforms"""
    config = AUGMENTATION_CONFIG[mode]
    
    if mode == 'train':
        return A.Compose([
            A.Resize(config['resize'][0], config['resize'][1]),
            A.HorizontalFlip(p=config['horizontal_flip']),
            A.VerticalFlip(p=config['vertical_flip']),
            A.Rotate(limit=config['rotate_limit'], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=config['brightness_limit'],
                contrast_limit=config['contrast_limit'],
                p=0.5
            ),
            A.GaussNoise(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:  # val/test
        return A.Compose([
            A.Resize(config['resize'][0], config['resize'][1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])


def prepare_hyperkvasir_data(val_split=0.2, test_split=0.1, seed=42):
    """
    Chuẩn bị data từ HyperKvasir dataset
    
    Returns:
        train_dataset, val_dataset, test_dataset, unlabeled_dataset
    """

    image_paths = sorted(list(LABELED_IMAGES.glob('*.jpg')))
    mask_paths = []
    
    for img_path in image_paths:
        mask_path = LABELED_MASKS / img_path.name
        if mask_path.exists():
            mask_paths.append(mask_path)
        else:
            print(f"Warning: Mask not found for {img_path.name}")

    assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"
    
    print(f"Found {len(image_paths)} labeled images with masks")
    
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, 
        test_size=val_split + test_split,
        random_state=seed
    )

    val_ratio = val_split / (val_split + test_split)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        val_imgs, val_masks,
        test_size=1-val_ratio,
        random_state=seed
    )
    
    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
    
    train_dataset = HyperKvasirDataset(
        train_imgs, train_masks, 
        transform=get_transforms('train')
    )
    
    val_dataset = HyperKvasirDataset(
        val_imgs, val_masks,
        transform=get_transforms('val')
    )
    
    test_dataset = HyperKvasirDataset(
        test_imgs, test_masks,
        transform=get_transforms('val')
    )
    
    # Unlabeled dataset
    unlabeled_dataset = None
    if UNLABELED_IMAGES.exists():
        unlabeled_paths = sorted(list(UNLABELED_IMAGES.glob('*.jpg')))
        if len(unlabeled_paths) > 0:
            print(f"Found {len(unlabeled_paths)} unlabeled images")
            unlabeled_dataset = HyperKvasirUnlabeledDataset(
                unlabeled_paths,
                transform=get_transforms('train')
            )
    
    return train_dataset, val_dataset, test_dataset, unlabeled_dataset


def get_dataloaders(batch_size=None, num_workers=None):
    """Get train, val, test dataloaders"""
    if batch_size is None:
        batch_size = TRAIN_CONFIG['batch_size']
    if num_workers is None:
        num_workers = TRAIN_CONFIG['num_workers']
    
    train_ds, val_ds, test_ds, unlabeled_ds = prepare_hyperkvasir_data()
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=TRAIN_CONFIG['pin_memory']
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=TRAIN_CONFIG['pin_memory']
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=TRAIN_CONFIG['pin_memory']
    )
    
    unlabeled_loader = None
    if unlabeled_ds is not None and TRAIN_CONFIG['use_unlabeled']:
        unlabeled_loader = DataLoader(
            unlabeled_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=TRAIN_CONFIG['pin_memory']
        )
    
    return train_loader, val_loader, test_loader, unlabeled_loader