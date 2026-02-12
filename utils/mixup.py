"""
Mixup Data Augmentation for Segmentation
Paper: mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
Adapted for segmentation tasks
"""
import numpy as np
import torch
import torch.nn.functional as F


class SegmentationMixup:
    """
    Mixup augmentation cho segmentation
    Trộn lẫn cả images và masks với tỷ lệ lambda
    """
    
    def __init__(self, alpha=0.4, prob=0.5):
        """
        Args:
            alpha: Beta distribution parameter (thường dùng 0.2 - 0.4)
            prob: Xác suất áp dụng mixup
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, images, masks=None, training=True):
        """
        Apply mixup augmentation
        
        Args:
            images: torch.Tensor of shape (B, C, H, W)
            masks: torch.Tensor of shape (B, 1, H, W) or None
            training: whether in training mode
            
        Returns:
            mixed_images, mixed_masks (if masks provided)
        """
        if not training or np.random.random() > self.prob:
            if masks is not None:
                return images, masks
            return images
        
        batch_size = images.size(0)
        
        # Sample lambda từ Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Random permutation
        index = torch.randperm(batch_size).to(images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Mix masks if provided
        if masks is not None:
            mixed_masks = lam * masks + (1 - lam) * masks[index]
            return mixed_images, mixed_masks
        
        return mixed_images


class CutMix:
    """
    CutMix augmentation cho segmentation
    Alternative mixing strategy
    """
    
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, images, masks=None, training=True):
        if not training or np.random.random() > self.prob:
            if masks is not None:
                return images, masks
            return images
        
        batch_size = images.size(0)
        _, _, H, W = images.shape
        
        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random bounding box
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Random permutation
        index = torch.randperm(batch_size).to(images.device)
        
        # Cut and paste
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        
        if masks is not None:
            mixed_masks = masks.clone()
            mixed_masks[:, :, y1:y2, x1:x2] = masks[index, :, y1:y2, x1:x2]
            return mixed_images, mixed_masks
        
        return mixed_images


class MixupWithUnlabeled:
    """
    Mixup với unlabeled data cho semi-supervised learning
    """
    
    def __init__(self, alpha=0.4, prob=0.5):
        self.alpha = alpha
        self.prob = prob
        self.base_mixup = SegmentationMixup(alpha, prob)
    
    def __call__(self, labeled_imgs, labeled_masks, unlabeled_imgs, training=True):
        """
        Mix labeled và unlabeled data
        
        Args:
            labeled_imgs: labeled images
            labeled_masks: labeled masks
            unlabeled_imgs: unlabeled images
            training: training mode
            
        Returns:
            mixed_images, mixed_masks, is_labeled_mask
        """
        if not training or np.random.random() > self.prob:
            # Concatenate labeled and unlabeled
            all_imgs = torch.cat([labeled_imgs, unlabeled_imgs], dim=0)
            # Create masks với zeros cho unlabeled
            unlabeled_masks = torch.zeros_like(labeled_masks[:unlabeled_imgs.size(0)])
            all_masks = torch.cat([labeled_masks, unlabeled_masks], dim=0)
            # Mask để biết đâu là labeled
            is_labeled = torch.cat([
                torch.ones(labeled_imgs.size(0), dtype=torch.bool),
                torch.zeros(unlabeled_imgs.size(0), dtype=torch.bool)
            ])
            return all_imgs, all_masks, is_labeled
        
        # Apply mixup cho labeled data
        mixed_labeled, mixed_masks = self.base_mixup(labeled_imgs, labeled_masks, training=True)
        
        # Apply mixup cho unlabeled data (chỉ images)
        mixed_unlabeled = self.base_mixup(unlabeled_imgs, training=True)
        
        # Combine
        all_imgs = torch.cat([mixed_labeled, mixed_unlabeled], dim=0)
        unlabeled_masks = torch.zeros_like(mixed_masks[:mixed_unlabeled.size(0)])
        all_masks = torch.cat([mixed_masks, unlabeled_masks], dim=0)
        
        is_labeled = torch.cat([
            torch.ones(mixed_labeled.size(0), dtype=torch.bool),
            torch.zeros(mixed_unlabeled.size(0), dtype=torch.bool)
        ])
        
        return all_imgs, all_masks, is_labeled