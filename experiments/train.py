"""
Main Training Script với Mixup Augmentation
"""
import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data import get_dataloaders
from models import ResNet50UNet
from utils import (
    SegmentationMixup, MixupWithUnlabeled,
    SegmentationMetrics, CombinedLoss,
    visualize_batch, plot_training_history,
    ExperimentManager
)
from config import TRAIN_CONFIG, CHECKPOINTS_DIR, LOGS_DIR, VISUALIZATIONS_DIR, get_experiment_dirs


class Trainer:
    """Trainer class cho segmentation model"""
    
    def __init__(self, model, train_loader, val_loader, unlabeled_loader=None, config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.unlabeled_loader = unlabeled_loader
        self.config = config or TRAIN_CONFIG
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Loss function
        self.criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        
        # Optimizer
        if self.config['optimizer'] == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        
        # Scheduler
        if self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs']
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
        
        # Mixup
        if self.config['use_mixup']:
            if unlabeled_loader is not None and self.config['use_unlabeled']:
                self.mixup = MixupWithUnlabeled(
                    alpha=self.config['mixup_alpha'],
                    prob=self.config['mixup_prob']
                )
                self.use_unlabeled = True
            else:
                self.mixup = SegmentationMixup(
                    alpha=self.config['mixup_alpha'],
                    prob=self.config['mixup_prob']
                )
                self.use_unlabeled = False
        else:
            self.mixup = None
            self.use_unlabeled = False
        
        # Metrics
        self.train_metrics = SegmentationMetrics()
        self.val_metrics = SegmentationMetrics()
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_iou': [], 'val_iou': [],
            'train_dice': [], 'val_dice': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1': [], 'val_f1': []
        }
        
        # Best metrics
        self.best_val_dice = 0.0
        self.patience_counter = 0
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=LOGS_DIR)
        
        # Unlabeled iterator
        if self.use_unlabeled and unlabeled_loader:
            self.unlabeled_iter = iter(unlabeled_loader)
        else:
            self.unlabeled_iter = None
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["num_epochs"]} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Get unlabeled data if using semi-supervised
            if self.use_unlabeled and self.unlabeled_iter:
                try:
                    unlabeled_batch = next(self.unlabeled_iter)
                except StopIteration:
                    self.unlabeled_iter = iter(self.unlabeled_loader)
                    unlabeled_batch = next(self.unlabeled_iter)
                
                unlabeled_images = unlabeled_batch['image'].to(self.device)
                
                # Apply mixup with unlabeled
                images, masks, is_labeled = self.mixup(
                    images, masks, unlabeled_images, training=True
                )
            elif self.mixup:
                # Apply regular mixup
                images, masks = self.mixup(images, masks, training=True)
                is_labeled = torch.ones(images.size(0), dtype=torch.bool)
            else:
                is_labeled = torch.ones(images.size(0), dtype=torch.bool)
            
            # Forward
            outputs = self.model(images)
            
            # Compute loss only on labeled data
            if self.use_unlabeled:
                labeled_outputs = outputs[is_labeled]
                labeled_masks = masks[is_labeled]
                if labeled_outputs.size(0) > 0:
                    loss = self.criterion(labeled_outputs, labeled_masks)
                else:
                    continue
            else:
                loss = self.criterion(outputs, masks)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics (only on labeled data)
            if self.use_unlabeled:
                if labeled_outputs.size(0) > 0:
                    self.train_metrics.update(labeled_outputs, labeled_masks)
            else:
                self.train_metrics.update(outputs, masks)
            
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = epoch_loss / len(self.train_loader)
        train_metrics = self.train_metrics.get_metrics()
        
        return avg_loss, train_metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        self.val_metrics.reset()
        
        epoch_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.config["num_epochs"]} [Val]')
        
        # Collect samples for visualization
        vis_images, vis_masks, vis_preds = [], [], []
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            epoch_loss += loss.item()
            self.val_metrics.update(outputs, masks)
            
            # Collect samples for visualization
            if len(vis_images) < 8:
                vis_images.append(images[:4])
                vis_masks.append(masks[:4])
                vis_preds.append(outputs[:4])
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(self.val_loader)
        val_metrics = self.val_metrics.get_metrics()
        
        # Visualize predictions
        if epoch % 5 == 0 or epoch == 1:
            vis_imgs = torch.cat(vis_images[:2], dim=0)
            vis_msks = torch.cat(vis_masks[:2], dim=0)
            vis_prds = torch.cat(vis_preds[:2], dim=0)
            
            fig = visualize_batch(
                vis_imgs, vis_msks, vis_prds,
                num_samples=min(4, vis_imgs.size(0)),
                save_path=VISUALIZATIONS_DIR / f'epoch_{epoch}_predictions.png'
            )
            import matplotlib.pyplot as plt
            plt.close(fig)
        
        return avg_loss, val_metrics
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"Model: ResNet50-UNet")
        print(f"Mixup: {self.config['use_mixup']} (alpha={self.config.get('mixup_alpha', 0)})")
        print(f"Use Unlabeled: {self.use_unlabeled}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Learning Rate: {self.config['learning_rate']}")
        print(f"Epochs: {self.config['num_epochs']}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for key in ['iou', 'dice', 'precision', 'recall', 'f1']:
                self.history[f'train_{key}'].append(train_metrics[key])
                self.history[f'val_{key}'].append(val_metrics[key])
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            for key in ['iou', 'dice', 'precision', 'recall', 'f1']:
                self.writer.add_scalar(f'Metrics/train_{key}', train_metrics[key], epoch)
                self.writer.add_scalar(f'Metrics/val_{key}', val_metrics[key], epoch)
            
            # Print epoch results
            print(f"\nEpoch {epoch}/{self.config['num_epochs']}:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
            print(f"  Train IoU:  {train_metrics['iou']:.4f} | Val IoU:  {val_metrics['iou']:.4f}")
            
            # Learning rate scheduling
            if self.config['scheduler'] == 'cosine':
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics['dice'])
            
            # Save best model
            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"  ✓ New best model saved! (Dice: {self.best_val_dice:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            # Save checkpoint periodically
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')
        
        # Plot training history
        print("\nTraining completed!")
        plot_training_history(
            self.history,
            save_path=VISUALIZATIONS_DIR / 'training_history.png'
        )
        
        self.writer.close()
        return self.history
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'history': self.history,
            'config': self.config
        }
        
        save_path = CHECKPOINTS_DIR / filename
        torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser(description='Train ResNet50-UNet with Mixup')
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['num_epochs'])
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['learning_rate'])
    parser.add_argument('--mixup_alpha', type=float, default=TRAIN_CONFIG['mixup_alpha'])
    parser.add_argument('--use_mixup', action='store_true', default=TRAIN_CONFIG['use_mixup'])
    parser.add_argument('--use_unlabeled', action='store_true', default=TRAIN_CONFIG['use_unlabeled'])
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name (V2.0 feature)')
    
    args = parser.parse_args()
    
    # Setup experiment manager if using V2

    experiment_manager = ExperimentManager('training', experiment_name=args.experiment_name)
        
    # Update directories to use experiment-specific paths
    global CHECKPOINTS_DIR, LOGS_DIR, VISUALIZATIONS_DIR
    exp_dirs = experiment_manager.exp_dir
    CHECKPOINTS_DIR = exp_dirs / 'checkpoints'
    LOGS_DIR = exp_dirs / 'logs'
    VISUALIZATIONS_DIR = exp_dirs / 'visualizations'
        
    # Ensure directories exist
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
        
    print(f"Using Experiment Manager: {args.experiment_name}")
    print(f"Results will be saved to: {exp_dirs}")
    
    # Update config
    config = TRAIN_CONFIG.copy()
    config['batch_size'] = args.batch_size
    config['num_epochs'] = args.epochs
    config['learning_rate'] = args.lr
    config['mixup_alpha'] = args.mixup_alpha
    config['use_mixup'] = args.use_mixup
    config['use_unlabeled'] = args.use_unlabeled
    
    # Get dataloaders
    train_loader, val_loader, test_loader, unlabeled_loader = get_dataloaders(
        batch_size=config['batch_size']
    )
    
    # Create model
    model = ResNet50UNet(in_channels=3, out_channels=1, pretrained=True)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, unlabeled_loader, config)
    
    # Save config if using experiment manager
    if experiment_manager:
        experiment_manager.save_config(config)
        experiment_manager.update_status('running')
    
    try:
        # Train
        history = trainer.train()
        
        # Save results if using experiment manager
        if experiment_manager:
            results = {
                'best_val_dice': float(trainer.best_val_dice),
                'final_train_loss': float(history['train_loss'][-1]) if history['train_loss'] else None,
                'final_val_loss': float(history['val_loss'][-1]) if history['val_loss'] else None,
                'history': {k: [float(v) for v in vals] for k, vals in history.items()}
            }
            experiment_manager.save_results(results)
            experiment_manager.update_status('completed')
        
        print("\nBest validation Dice score:", trainer.best_val_dice)
        
    except Exception as e:
        if experiment_manager:
            experiment_manager.handle_error(e, context="Training")
        raise


if __name__ == '__main__':
    main()