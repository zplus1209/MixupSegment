"""
Focus on two main questions:
1. Impact of ResNet50 encoder (vs other encoders)
2. Impact of Mixup augmentation
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
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from data import get_dataloaders
from models import get_model
from utils import (
    SegmentationMixup, MixupWithUnlabeled,
    SegmentationMetrics, CombinedLoss,
    plot_metrics_comparison, ExperimentManager,
    visualize_batch
)
from config import ABLATION_EXPERIMENTS, TRAIN_CONFIG, RESULTS_ROOT, VISUALIZATIONS_DIR


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
                self.mixup = MixupWithUnlabeled(alpha=config.get('mixup_alpha', 0.4), prob=config.get('mixup_prob', 0.5))
                self.use_unlabeled = True
                self.unlabeled_iter = iter(unlabeled_loader)
            else:
                self.mixup = SegmentationMixup(alpha=config.get('mixup_alpha', 0.4), prob=config.get('mixup_prob', 0.5))
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
                is_labeled = torch.ones(images.size(0), dtype=torch.bool, device=self.device)
            else:
                is_labeled = torch.ones(images.size(0), dtype=torch.bool, device=self.device)
            
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
    
    @torch.no_grad()
    def save_visualization(self, save_path: Path, num_samples=4):
        self.model.eval()
        first_batch = next(iter(self.val_loader))
        images = first_batch['image'].to(self.device)
        masks = first_batch['mask'].to(self.device)
        preds = self.model(images)
        fig = visualize_batch(
            images.detach().cpu(),
            masks.detach().cpu(),
            preds.detach().cpu(),
            num_samples=min(num_samples, images.size(0)),
            save_path=save_path
        )
        plt.close(fig)

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
                print(f"  Epoch {epoch}/{num_epochs}: Val Dice={val_metrics['dice']:.4f}, Best={self.best_val_dice:.4f}")
        
        return best_metrics


def run_ablation_study():
    """Run complete ablation study"""
    print("="*80)
    print("ABLATION STUDY")
    print("Evaluating impact of ResNet50 encoder and Mixup")
    print("="*80)
    
    # Get dataloaders
    train_loader, val_loader, test_loader, unlabeled_loader = get_dataloaders()
    
    # Results
    results = {}
    
    # Run experiments
    ablation_dir = RESULTS_ROOT / 'ablation_study'
    ablation_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = ablation_dir / 'sample_predictions'
    vis_dir.mkdir(parents=True, exist_ok=True)


    for i, (exp_name, exp_config) in enumerate(ABLATION_EXPERIMENTS.items(), start=1):
        print(f"\n[{i}/{len(ABLATION_EXPERIMENTS)}] Running: {exp_name}")
        print("-" * 80)

        try:
            encoder = exp_config.get('encoder', 'resnet50')
            model = get_model(encoder=encoder, pretrained=True, in_channels=3, out_channels=1)

            trainer = AblationTrainer(
                model, train_loader, val_loader, unlabeled_loader, exp_config
            )

            best_metrics = trainer.train(num_epochs=exp_config.get('num_epochs', 30))
            trainer.save_visualization(vis_dir / f'{exp_name}_predictions.png')

            best_metrics['encoder'] = encoder
            best_metrics['use_mixup'] = exp_config.get('use_mixup', False)
            best_metrics['use_unlabeled'] = exp_config.get('use_unlabeled', False)
            best_metrics['mixup_alpha'] = exp_config.get('mixup_alpha', TRAIN_CONFIG.get('mixup_alpha', 0.4))
            best_metrics['mixup_prob'] = exp_config.get('mixup_prob', TRAIN_CONFIG.get('mixup_prob', 0.5))

            results[exp_name] = best_metrics

            print(f"\nResults for {exp_name}:")
            print(f"  Best Dice: {best_metrics['dice']:.4f} (epoch {best_metrics['epoch']})")
            print(f"  IoU: {best_metrics['iou']:.4f}")
            print(f"  F1: {best_metrics['f1']:.4f}")
        except Exception as exc:
            print(f"Skipping {exp_name} due to error: {exc}")

    if not results:
        raise RuntimeError('No ablation experiment completed successfully.')
    
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
    available_cols = [c for c in ['dice', 'iou', 'precision', 'recall', 'f1', 'encoder', 'use_mixup', 'use_unlabeled', 'mixup_alpha'] if c in df.columns]
    print(df[available_cols])

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
        f.write("Evaluate two major factors:\n")
        f.write("- Impact of ResNet50 encoder\n")
        f.write("- Impact of Mixup augmentation\n\n")
        
        f.write("## Experimental Setup\n\n")
        f.write("- Base Model: ResNet50-UNet\n")
        f.write("- Training epochs: 30 (reduced for ablation)\n")
        f.write("- Batch size: 8\n")
        f.write("- Learning rate: 1e-4\n\n")

        f.write("## Commands Used (reproducibility)\n\n")
        f.write("```bash\n")
        f.write("python experiments/ablation_study.py --experiment_name ablation_main\n")
        f.write("python experiments/train.py --num_epochs 100 --batch_size 8 --use_mixup --mixup_alpha 0.4\n")
        f.write("python experiments/train.py --num_epochs 100 --batch_size 8 --use_mixup --mixup_alpha 0.4 --use_unlabeled\n")
        f.write("```\n\n")

        f.write("## Results\n\n")
        f.write("| Experiment | Mixup | Unlabeled | mixup_alpha | Encoder | Dice | IoU | Precision | Recall | F1 |\n")
        f.write("|------------|-------|-----------|-------------|---------|------|-----|-----------|--------|----|\n")

        for exp_name, metrics in results.items():
            mixup = '✓' if metrics.get('use_mixup', False) else '✗'
            unlabeled = '✓' if metrics.get('use_unlabeled', False) else '✗'
            encoder = metrics.get('encoder', 'resnet50')
            alpha = metrics.get('mixup_alpha', '-')
            
            f.write(f"| {exp_name} | {mixup} | {unlabeled} | {alpha} | {encoder} | ")
            f.write(f"{metrics['dice']:.4f} | {metrics['iou']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |\n")
            
        f.write("\n## Analysis\n\n")
        
        mixup_off = results.get('mixup_off', {})
        mixup_on = results.get('mixup_on', {})

        if mixup_off and mixup_on:
            improvement = (mixup_on['dice'] - mixup_off['dice']) / max(mixup_off['dice'], 1e-8) * 100
            f.write("### Impact of Mixup\n")
            f.write(f"- No Mixup Dice: {mixup_off['dice']:.4f}\n")
            f.write(f"- With Mixup Dice: {mixup_on['dice']:.4f}\n")
            f.write(f"- Relative change: {improvement:+.2f}%\n\n")

        encoder_keys = [k for k in results if k.startswith('encoder_')]
        if encoder_keys:
            best_encoder = max(encoder_keys, key=lambda k: results[k]['dice'])
            f.write("### Impact of ResNet50 Encoder\n")
            for key in sorted(encoder_keys):
                f.write(
                    f"- {results[key]['encoder']}: Dice={results[key]['dice']:.4f}, "
                    f"IoU={results[key]['iou']:.4f}\n"
                )
            f.write(f"- Best encoder setup: {results[best_encoder]['encoder']}\n\n")

        f.write("## Visualization\n\n")
        f.write("- Qualitative predictions are saved in `results/ablation_study/sample_predictions/`.\n")
    
    print(f"Ablation report saved to {report_path}")


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    args = parser.parse_args()

    experiment_manager = ExperimentManager('ablation_study', experiment_name=args.experiment_name)
    print(f"Using Experiment Manager: {args.experiment_name}")
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