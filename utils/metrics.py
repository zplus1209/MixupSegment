"""
Evaluation Metrics for Segmentation
IoU, Dice, Precision, Recall, F1
"""
import numpy as np
import torch
import torch.nn as nn


def calculate_iou(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) / Jaccard Index
    
    Args:
        pred: predictions (B, 1, H, W) or (B, H, W)
        target: ground truth (B, 1, H, W) or (B, H, W)
        threshold: threshold for binary prediction
        smooth: smoothing factor to avoid division by zero
    """
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = (pred * target).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def calculate_dice(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Dice Coefficient / F1 Score
    
    Args:
        pred: predictions (B, 1, H, W) or (B, H, W)
        target: ground truth (B, 1, H, W) or (B, H, W)
        threshold: threshold for binary prediction
        smooth: smoothing factor
    """
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = (pred * target).sum(dim=(-2, -1))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + smooth)
    
    return dice.mean()


def calculate_precision(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate Precision"""
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    tp = (pred * target).sum(dim=(-2, -1))
    fp = (pred * (1 - target)).sum(dim=(-2, -1))
    
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision.mean()


def calculate_recall(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate Recall / Sensitivity"""
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    tp = (pred * target).sum(dim=(-2, -1))
    fn = ((1 - pred) * target).sum(dim=(-2, -1))
    
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall.mean()


def calculate_f1(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate F1 Score"""
    precision = calculate_precision(pred, target, threshold, smooth)
    recall = calculate_recall(pred, target, threshold, smooth)
    
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    return f1


def calculate_pixel_accuracy(pred, target, threshold=0.5):
    """Calculate Pixel Accuracy"""
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    correct = (pred == target).float().sum()
    total = target.numel()
    
    return correct / total


class SegmentationMetrics:
    """Tổng hợp tất cả metrics cho segmentation"""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset tất cả metrics"""
        self.iou_sum = 0
        self.dice_sum = 0
        self.precision_sum = 0
        self.recall_sum = 0
        self.f1_sum = 0
        self.pixel_acc_sum = 0
        self.count = 0
    
    def update(self, pred, target):
        """
        Update metrics với batch mới
        
        Args:
            pred: predictions (B, 1, H, W)
            target: ground truth (B, 1, H, W)
        """
        with torch.no_grad():
            self.iou_sum += calculate_iou(pred, target, self.threshold).item()
            self.dice_sum += calculate_dice(pred, target, self.threshold).item()
            self.precision_sum += calculate_precision(pred, target, self.threshold).item()
            self.recall_sum += calculate_recall(pred, target, self.threshold).item()
            self.f1_sum += calculate_f1(pred, target, self.threshold).item()
            self.pixel_acc_sum += calculate_pixel_accuracy(pred, target, self.threshold).item()
            self.count += 1
    
    def get_metrics(self):
        """Get average metrics"""
        if self.count == 0:
            return {}
        
        return {
            'iou': self.iou_sum / self.count,
            'dice': self.dice_sum / self.count,
            'precision': self.precision_sum / self.count,
            'recall': self.recall_sum / self.count,
            'f1': self.f1_sum / self.count,
            'pixel_accuracy': self.pixel_acc_sum / self.count,
        }
    
    def __str__(self):
        metrics = self.get_metrics()
        return f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}, " \
               f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, " \
               f"F1: {metrics['f1']:.4f}, Pixel Acc: {metrics['pixel_accuracy']:.4f}"


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=(-2, -1))
        dice = (2. * intersection + self.smooth) / (
            pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + self.smooth
        )
        
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined BCE + Dice Loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss