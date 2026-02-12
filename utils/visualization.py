"""
Visualization Tools for Segmentation Results
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import torch

from config import VIS_CONFIG, VISUALIZATIONS_DIR


def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image for visualization"""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image = image * np.array(std) + np.array(mean)
    image = np.clip(image, 0, 1)
    return image


def visualize_segmentation(image, mask, pred, save_path=None, title=None):
    """
    Visualize image, ground truth mask, và prediction
    
    Args:
        image: input image (C, H, W) or (H, W, C)
        mask: ground truth mask (H, W) or (1, H, W)
        pred: prediction mask (H, W) or (1, H, W)
        save_path: path to save figure
        title: figure title
    """
    # Prepare image
    if isinstance(image, torch.Tensor):
        image = denormalize_image(image)
    elif len(image.shape) == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    # Prepare masks
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    
    if len(mask.shape) == 3:
        mask = mask.squeeze()
    if len(pred.shape) == 3:
        pred = pred.squeeze()
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=VIS_CONFIG['figsize'])
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay
    overlay = image.copy()
    pred_binary = (pred > 0.5).astype(np.uint8)
    overlay[pred_binary == 1] = overlay[pred_binary == 1] * 0.5 + np.array([1, 0, 0]) * 0.5
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def visualize_batch(images, masks, preds, num_samples=4, save_path=None):
    """
    Visualize multiple samples từ batch
    
    Args:
        images: batch of images (B, C, H, W)
        masks: batch of masks (B, 1, H, W)
        preds: batch of predictions (B, 1, H, W)
        num_samples: number of samples to visualize
        save_path: path to save figure
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(15, num_samples * 3))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image
        img = denormalize_image(images[i])
        mask = masks[i].cpu().numpy().squeeze()
        pred = preds[i].cpu().numpy().squeeze()
        
        # Original
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = img.copy()
        pred_binary = (pred > 0.5).astype(np.uint8)
        overlay[pred_binary == 1] = overlay[pred_binary == 1] * 0.5 + np.array([1, 0, 0]) * 0.5
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
        print(f"Saved batch visualization to {save_path}")
    
    return fig


def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history: dictionary with 'train_loss', 'val_loss', 'train_metrics', 'val_metrics'
        save_path: path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU
    axes[0, 1].plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    axes[0, 1].plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    axes[0, 1].set_title('IoU Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice
    axes[0, 2].plot(epochs, history['train_dice'], 'b-', label='Train Dice')
    axes[0, 2].plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    axes[0, 2].set_title('Dice Score')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Dice')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Precision
    axes[1, 0].plot(epochs, history['train_precision'], 'b-', label='Train Precision')
    axes[1, 0].plot(epochs, history['val_precision'], 'r-', label='Val Precision')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(epochs, history['train_recall'], 'b-', label='Train Recall')
    axes[1, 1].plot(epochs, history['val_recall'], 'r-', label='Val Recall')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # F1
    axes[1, 2].plot(epochs, history['train_f1'], 'b-', label='Train F1')
    axes[1, 2].plot(epochs, history['val_f1'], 'r-', label='Val F1')
    axes[1, 2].set_title('F1 Score')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('F1')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    return fig


def plot_metrics_comparison(results_dict, save_path=None):
    """
    Plot comparison of metrics across different models/experiments
    
    Args:
        results_dict: dict of {model_name: metrics_dict}
        save_path: path to save figure
    """
    metrics_names = ['iou', 'dice', 'precision', 'recall', 'f1']
    models = list(results_dict.keys())
    
    # Prepare data
    data = {metric: [] for metric in metrics_names}
    for model in models:
        for metric in metrics_names:
            data[metric].append(results_dict[model].get(metric, 0))
    
    # Plot
    fig, axes = plt.subplots(1, len(metrics_names), figsize=(20, 4))
    
    for i, metric in enumerate(metrics_names):
        axes[i].bar(models, data[metric])
        axes[i].set_title(metric.upper())
        axes[i].set_ylabel('Score')
        axes[i].set_ylim([0, 1])
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for j, v in enumerate(data[metric]):
            axes[i].text(j, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    
    return fig


def create_comparison_table(results_dict, save_path=None):
    """Create comparison table of results"""
    import pandas as pd
    
    df = pd.DataFrame(results_dict).T
    df = df.round(4)
    
    if save_path:
        df.to_csv(save_path)
        print(f"Saved comparison table to {save_path}")
    
    return df