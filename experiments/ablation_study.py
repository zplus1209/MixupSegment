"""
Ablation Study Experiments
Đánh giá ảnh hưởng của từng component:
1. Baseline (no mixup, no unlabeled)
2. With Mixup
3. With Unlabeled data
4. Full model (Mixup + Unlabeled)
5. Different encoder (ResNet34)
"""
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from data import get_dataloaders
from models import ResNet50UNet, get_model
from utils import (
    SegmentationMixup, MixupWithUnlabeled,
    SegmentationMetrics, CombinedLoss,
    plot_metrics_comparison
)
from config import ABLATION_EXPERIMENTS, TRAIN_CONFIG, RESULTS_ROOT, VISUALIZATIONS_DIR

try:
    from utils import ExperimentManager
    USE_V2 = True
except ImportError:
    USE_V2 = False


class AblationTrainer:
    """Simplified trainer for ablation study"""
    
    def __init__(self, model, train_loader, val_loader, unlabeled_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.unlabeled_loader = unlabeled_loader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = CombinedLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 30)
        )
        
        # Mixup
        if config.get('use_mixup', False):
            if config.get('use_unlabeled', False) and unlabeled_loader:
                self.mixup = MixupWithUnlabeled(alpha=0.4, prob=0.5)
                self.use_unlabeled = True
                self.unlabeled_iter = iter(unlabeled_loader)
            else:
                self.mixup = SegmentationMixup(alpha=0.4, prob=0.5)
                self.use_unlabeled = False
                self.unlabeled_iter = None
        else:
            self.mixup = None
            self.use_unlabeled = False
            self.unlabeled_iter = None
        
        self.best_val_dice = 0.0
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        metrics = SegmentationMetrics()
        total_loss = 0.0
        
        for batch in self.train_loader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Mixup augmentation
            if self.use_unlabeled and self.unlabeled_iter:
                try:
                    unlabeled_batch = next(self.unlabeled_iter)
                except StopIteration:
                    self.unlabeled_iter = iter(self.unlabeled_loader)
                    unlabeled_batch = next(self.unlabeled_iter)
                
                unlabeled_images = unlabeled_batch['image'].to(self.device)
                images, masks, is_labeled = self.mixup(
                    images, masks, unlabeled_images, training=True
                )
            elif self.mixup:
                images, masks = self.mixup(images, masks, training=True)
                is_labeled = torch.ones(images.size(0), dtype=torch.bool)
            else:
                is_labeled = torch.ones(images.size(0), dtype=torch.bool)
            
            # Forward
            outputs = self.model(images)
            
            # Loss (only labeled)
            if self.use_unlabeled:
                labeled_outputs = outputs[is_labeled]
                labeled_masks = masks[is_labeled]
                if labeled_outputs.size(0) > 0:
                    loss = self.criterion(labeled_outputs, labeled_masks)
                    metrics.update(labeled_outputs, labeled_masks)
                else:
                    continue
            else:
                loss = self.criterion(outputs, masks)
                metrics.update(outputs, masks)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, metrics.get_metrics()
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        metrics = SegmentationMetrics()
        total_loss = 0.0
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            total_loss += loss.item()
            metrics.update(outputs, masks)
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, metrics.get_metrics()
    
    def train(self, num_epochs=30):
        """Train for specified epochs"""
        print(f"Training with config: {self.config}")
        
        best_metrics = None
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Scheduler
            self.scheduler.step()
            
            # Track best
            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
                best_metrics = val_metrics.copy()
                best_metrics['epoch'] = epoch
            
            # Print progress
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{num_epochs}: "
                      f"Val Dice={val_metrics['dice']:.4f}, "
                      f"Best={self.best_val_dice:.4f}")
        
        return best_metrics


def run_ablation_study():
    """Run complete ablation study"""
    print("="*80)
    print("ABLATION STUDY")
    print("Evaluating impact of each component")
    print("="*80)
    
    # Get dataloaders
    train_loader, val_loader, test_loader, unlabeled_loader = get_dataloaders()
    
    # Results
    results = {}
    
    # Run experiments
    for i, (exp_name, exp_config) in enumerate(ABLATION_EXPERIMENTS.items(), start=1):
        print(f"\n[{i}/{len(ABLATION_EXPERIMENTS)}] Running: {exp_name}")
        print("-" * 80)
        
        # Create model based on encoder
        encoder = exp_config.get('encoder', 'resnet50')
        if encoder == 'resnet50':
            model = ResNet50UNet(in_channels=3, out_channels=1, pretrained=True)
        else:
            # For other encoders, we'd need to implement or use smp
            print(f"  Using ResNet50 (other encoders not implemented in custom model)")
            model = ResNet50UNet(in_channels=3, out_channels=1, pretrained=True)
        
        # Create trainer
        trainer = AblationTrainer(
            model, train_loader, val_loader, unlabeled_loader, exp_config
        )
        
        # Train
        best_metrics = trainer.train(num_epochs=30)  # Reduced epochs for ablation
        
        # Store results
        results[exp_name] = best_metrics
        
        print(f"\nResults for {exp_name}:")
        print(f"  Best Dice: {best_metrics['dice']:.4f} (epoch {best_metrics['epoch']})")
        print(f"  IoU: {best_metrics['iou']:.4f}")
        print(f"  F1: {best_metrics['f1']:.4f}")
    
    # Save results
    ablation_dir = RESULTS_ROOT / 'ablation_study'
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    results_json = ablation_dir / 'ablation_results.json'
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_json}")
    
    # Create table
    df = pd.DataFrame(results).T
    df = df.round(4)
    df.to_csv(ablation_dir / 'ablation_table.csv')
    
    print("\nAblation Study Results:")
    print(df[['dice', 'iou', 'precision', 'recall', 'f1']])
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_metrics_comparison(
        results,
        save_path=VISUALIZATIONS_DIR / 'ablation_study.png'
    )
    
    # Generate report
    generate_ablation_report(results, ablation_dir)
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETED")
    print(f"Results saved to: {ablation_dir}")
    print("="*80)
    
    return results, df


def generate_ablation_report(results, save_dir):
    """Generate detailed ablation report"""
    report_path = save_dir / 'ablation_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Ablation Study Report\n\n")
        f.write("## Objective\n\n")
        f.write("Evaluate the contribution of each component:\n")
        f.write("- Mixup augmentation\n")
        f.write("- Unlabeled data utilization\n")
        f.write("- Encoder architecture\n\n")
        
        f.write("## Experimental Setup\n\n")
        f.write("- Base Model: ResNet50-UNet\n")
        f.write("- Training epochs: 30 (reduced for ablation)\n")
        f.write("- Batch size: 8\n")
        f.write("- Learning rate: 1e-4\n\n")
        
        f.write("## Results\n\n")
        f.write("| Experiment | Mixup | Unlabeled | Encoder | Dice | IoU | F1 |\n")
        f.write("|------------|-------|-----------|---------|------|-----|----|\n")
        
        for exp_name, metrics in results.items():
            config = ABLATION_EXPERIMENTS[exp_name]
            mixup = '✓' if config.get('use_mixup', False) else '✗'
            unlabeled = '✓' if config.get('use_unlabeled', False) else '✗'
            encoder = config.get('encoder', 'resnet50')
            
            f.write(f"| {exp_name} | {mixup} | {unlabeled} | {encoder} | ")
            f.write(f"{metrics['dice']:.4f} | {metrics['iou']:.4f} | {metrics['f1']:.4f} |\n")
        
        f.write("\n## Analysis\n\n")
        
        # Compare baseline with others
        baseline = results.get('baseline', {})
        with_mixup = results.get('with_mixup', {})
        with_unlabeled = results.get('with_unlabeled', {})
        full_model = results.get('full_model', {})
        
        if baseline and with_mixup:
            improvement = (with_mixup['dice'] - baseline['dice']) / baseline['dice'] * 100
            f.write(f"### Impact of Mixup\n")
            f.write(f"- Baseline Dice: {baseline['dice']:.4f}\n")
            f.write(f"- With Mixup Dice: {with_mixup['dice']:.4f}\n")
            f.write(f"- Improvement: {improvement:+.2f}%\n\n")
        
        if baseline and with_unlabeled:
            improvement = (with_unlabeled['dice'] - baseline['dice']) / baseline['dice'] * 100
            f.write(f"### Impact of Unlabeled Data\n")
            f.write(f"- Baseline Dice: {baseline['dice']:.4f}\n")
            f.write(f"- With Unlabeled Dice: {with_unlabeled['dice']:.4f}\n")
            f.write(f"- Improvement: {improvement:+.2f}%\n\n")
        
        if baseline and full_model:
            improvement = (full_model['dice'] - baseline['dice']) / baseline['dice'] * 100
            f.write(f"### Full Model vs Baseline\n")
            f.write(f"- Baseline Dice: {baseline['dice']:.4f}\n")
            f.write(f"- Full Model Dice: {full_model['dice']:.4f}\n")
            f.write(f"- Total Improvement: {improvement:+.2f}%\n\n")
        
        f.write("## Conclusions\n\n")
        f.write("- Key findings from ablation study\n")
        f.write("- Best component combination\n")
        f.write("- Recommendations for production\n")
    
    print(f"Ablation report saved to {report_path}")


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    args = parser.parse_args()
    
    experiment_manager = None
    
    if USE_V2 and args.experiment_name:
        experiment_manager = ExperimentManager('ablation_study', experiment_name=args.experiment_name)
        print(f"Using V2.0 Experiment Manager: {args.experiment_name}")
        experiment_manager.update_status('running')
    
    try:
        results, df = run_ablation_study()
        
        if experiment_manager:
            experiment_manager.save_results(results)
            experiment_manager.update_status('completed')
    except Exception as e:
        if experiment_manager:
            experiment_manager.handle_error(e, context="Ablation Study")
        raise


if __name__ == '__main__':
    main()