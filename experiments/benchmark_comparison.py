"""
Benchmark Comparison Experiments
So sánh ResNet50-UNet với các model SOTA khác
"""
import os
import sys
import re
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from data import get_dataloaders
from models import ResNet50UNet, get_all_benchmark_models, get_model_info
from utils import (
    SegmentationMetrics, CombinedLoss,
    plot_metrics_comparison, create_comparison_table,
    ExperimentManager, visualize_batch
)
from config import CHECKPOINTS_DIR, VISUALIZATIONS_DIR, RESULTS_ROOT

def _slugify(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def _find_checkpoint_for_model(model_name: str, checkpoints_dir: Path):
    """Find checkpoint by explicit model_name metadata or filename fallback."""
    if not checkpoints_dir.exists():
        return None

    candidates = sorted(checkpoints_dir.rglob('*.pth'))
    if not candidates:
        return None

    model_slug = _slugify(model_name)
    # 1) metadata match
    for ckpt in candidates:
        try:
            obj = torch.load(ckpt, map_location='cpu')
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get('model_name') == model_name:
            return ckpt

    # 2) filename match fallback
    for ckpt in candidates:
        if model_slug in _slugify(ckpt.stem):
            return ckpt

    return None


class BenchmarkEvaluator:
    """Evaluator for benchmark comparison"""
    
    def __init__(self, test_loader, vis_dir: Path):
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = CombinedLoss()

        self.vis_dir = vis_dir
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate_model(self, model, model_name):
        """Evaluate a single model"""
        print(f"\nEvaluating {model_name}...")
        
        model.to(self.device)
        model.eval()
        
        metrics = SegmentationMetrics()
        total_loss = 0.0
        
        vis_images, vis_masks, vis_preds = [], [], []

        pbar = tqdm(self.test_loader, desc=f'Evaluating {model_name}')
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            outputs = model(images)
            loss = self.criterion(outputs, masks)
            
            total_loss += loss.item()
            metrics.update(outputs, masks)

            if len(vis_images) < 2:
                vis_images.append(images[:4].detach().cpu())
                vis_masks.append(masks[:4].detach().cpu())
                vis_preds.append(outputs[:4].detach().cpu())


            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Get metrics
        avg_loss = total_loss / len(self.test_loader)
        result_metrics = metrics.get_metrics()
        result_metrics['loss'] = avg_loss
        
        # Add model info
        model_info = get_model_info(model)
        result_metrics.update(model_info)

        if vis_images:
            vis_img = torch.cat(vis_images, dim=0)
            vis_msk = torch.cat(vis_masks, dim=0)
            vis_prd = torch.cat(vis_preds, dim=0)
            vis_path = self.vis_dir / f'{_slugify(model_name)}_predictions.png'
            fig = visualize_batch(
                vis_img, vis_msk, vis_prd,
                num_samples=min(4, vis_img.size(0)),
                save_path=vis_path
            )
            plt.close(fig)
            result_metrics['visualization_path'] = str(vis_path)

        print(f"Results for {model_name}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  IoU: {result_metrics['iou']:.4f}")
        print(f"  Dice: {result_metrics['dice']:.4f}")
        print(f"  Precision: {result_metrics['precision']:.4f}")
        print(f"  Recall: {result_metrics['recall']:.4f}")
        print(f"  F1: {result_metrics['f1']:.4f}")
        print(f"  Parameters: {result_metrics['total_params_M']:.2f}M")
        
        return result_metrics


def run_benchmark_comparison(checkpoints_dir=CHECKPOINTS_DIR, allow_random_ours=False):
    """Run complete benchmark comparison"""
    print("="*80)
    print("BENCHMARK COMPARISON")
    print("Comparing our model with requested segmentation baselines")
    print("="*80)
    
    # Get test dataloader
    _, _, test_loader, _ = get_dataloaders()

    benchmark_results_dir = RESULTS_ROOT / 'benchmark_comparison'
    benchmark_results_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = benchmark_results_dir / 'sample_predictions'

    # Initialize evaluator
    evaluator = BenchmarkEvaluator(test_loader, vis_dir=vis_dir)
    
    # Results dictionary
    results = {}
    
    # 1. Evaluate our ResNet50-UNet (load best checkpoint)
    print("\n[1/N] Evaluating ResNet50-UNet (Ours)")
    our_model = ResNet50UNet(in_channels=3, out_channels=1, pretrained=False)
    
    ours_ckpt = _find_checkpoint_for_model('ResNet50-UNet (Ours)', checkpoints_dir)
    if ours_ckpt is None:
        legacy_ckpt = checkpoints_dir / 'best_model.pth'
        if legacy_ckpt.exists():
            ours_ckpt = legacy_ckpt

    if ours_ckpt is not None:
        checkpoint = torch.load(ours_ckpt, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        our_model.load_state_dict(state_dict)
        print(f"Loaded checkpoint for Ours: {ours_ckpt}")
        results['ResNet50-UNet (Ours)'] = evaluator.evaluate_model(our_model, 'ResNet50-UNet (Ours)')
    elif allow_random_ours:
        print("Warning: No trained checkpoint found for Ours. Using random initialization (--allow_random_ours).")
        results['ResNet50-UNet (Ours)'] = evaluator.evaluate_model(our_model, 'ResNet50-UNet (Ours)')
    else:
        print("Skipping Ours: checkpoint not found. Run train.py first or pass --allow_random_ours.")

    # 2. Evaluate benchmark models (checkpoint-required)
    benchmark_models = get_all_benchmark_models()
    
    for i, (model_name, model) in enumerate(benchmark_models.items(), start=2):
        print(f"\n[{i}/{len(benchmark_models)+1}] Evaluating {model_name}")

        ckpt = _find_checkpoint_for_model(model_name, checkpoints_dir)
        if ckpt is None:
            print(f"Skipping {model_name}: checkpoint not found in {checkpoints_dir}")
            continue

        try:
            checkpoint = torch.load(ckpt, map_location='cpu')
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint: {ckpt}")
            results[model_name] = evaluator.evaluate_model(model, model_name)
        except Exception as exc:
            print(f"Skipping {model_name} due to runtime error: {exc}")

    if not results:
        raise RuntimeError(
            "No model was evaluated. Please train models first and ensure checkpoints exist."
        )
    
    # Save results as JSON
    results_json = benchmark_results_dir / 'benchmark_results.json'
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_json}")
    
    # Create comparison table
    df = create_comparison_table(
        results,
        save_path=benchmark_results_dir / 'benchmark_table.csv'
    )
    print("\nComparison Table:")
    print(df)
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_metrics_comparison(
        results,
        save_path=VISUALIZATIONS_DIR / 'benchmark_comparison.png'
    )
    
    # Generate detailed report
    generate_detailed_report(results, benchmark_results_dir)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON COMPLETED")
    print(f"Results saved to: {benchmark_results_dir}")
    print("="*80)
    
    return results, df


def generate_detailed_report(results, save_dir):
    """Generate detailed markdown report"""
    report_path = save_dir / 'benchmark_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Benchmark Comparison Report\n\n")
        f.write("## Overview\n\n")
        f.write("Comparison of our model with requested segmentation baselines ")
        f.write("on HyperKvasir/Kvasir-style polyp segmentation data.\n\n")
        
        f.write("## Results\n\n")
        f.write("### Performance Metrics\n\n")
        
        # Create markdown table
        f.write("| Model | IoU | Dice | Precision | Recall | F1 | Params (M) | Visualize |\n")
        f.write("|-------|-----|------|-----------|--------|----|-----------|-----------|\n")
        
        for model_name, metrics in results.items():
            f.write(f"| {model_name} | ")
            f.write(f"{metrics['iou']:.4f} | ")
            f.write(f"{metrics['dice']:.4f} | ")
            f.write(f"{metrics['precision']:.4f} | ")
            f.write(f"{metrics['recall']:.4f} | ")
            f.write(f"{metrics['f1']:.4f} | ")
            f.write(f"{metrics['total_params_M']:.2f} | ")
            f.write(f"{metrics.get('visualization_path', 'N/A')} |\n")
        
        f.write("\n### Key Findings\n\n")
        
        # Find best model for each metric
        best_iou = max(results.items(), key=lambda x: x[1]['iou'])
        best_dice = max(results.items(), key=lambda x: x[1]['dice'])
        best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
        smallest = min(results.items(), key=lambda x: x[1]['total_params_M'])
        
        f.write(f"- **Best IoU**: {best_iou[0]} ({best_iou[1]['iou']:.4f})\n")
        f.write(f"- **Best Dice**: {best_dice[0]} ({best_dice[1]['dice']:.4f})\n")
        f.write(f"- **Best F1**: {best_f1[0]} ({best_f1[1]['f1']:.4f})\n")
        f.write(f"- **Smallest Model**: {smallest[0]} ({smallest[1]['total_params_M']:.2f}M params)\n")
        
        f.write("\n### Analysis\n\n")
        
        if 'ResNet50-UNet (Ours)' in results:
            our_results = results['ResNet50-UNet (Ours)']
            f.write(f"ResNet50-UNet achieves:\n")
            f.write(f"- IoU: {our_results['iou']:.4f}\n")
            f.write(f"- Dice: {our_results['dice']:.4f}\n")
            f.write(f"- Parameters: {our_results['total_params_M']:.2f}M\n")
        
    print(f"Detailed report saved to {report_path}")


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark Comparison')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name (V2.0 feature)')
    parser.add_argument('--checkpoints_dir', type=str, default=str(CHECKPOINTS_DIR), help='Directory containing model checkpoints')
    parser.add_argument('--allow_random_ours', action='store_true', help='Allow evaluating our model without checkpoint')
    args = parser.parse_args()
    

    experiment_manager = ExperimentManager('benchmark_comparison', experiment_name=args.experiment_name)
    print(f"Using Experiment Manager: {args.experiment_name}")
    experiment_manager.update_status('running')
    
    try:
        results, df = run_benchmark_comparison(
            checkpoints_dir=Path(args.checkpoints_dir),
            allow_random_ours=args.allow_random_ours
        )
        
        if experiment_manager:
            experiment_manager.save_results(results)
            experiment_manager.update_status('completed')
    except Exception as e:
        if experiment_manager:
            experiment_manager.handle_error(e, context="Benchmark Comparison")
        raise


if __name__ == '__main__':
    main()