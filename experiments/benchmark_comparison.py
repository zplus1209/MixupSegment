"""
Benchmark Comparison Experiments
So sánh ResNet50-UNet với các model SOTA khác
"""
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json

sys.path.append(str(Path(__file__).parent.parent))

from data import get_dataloaders
from models import ResNet50UNet, get_all_benchmark_models, get_model_info, count_parameters
from utils import (
    SegmentationMetrics, CombinedLoss,
    plot_metrics_comparison, create_comparison_table,
    ExperimentManager
)
from config import CHECKPOINTS_DIR, VISUALIZATIONS_DIR, RESULTS_ROOT


class BenchmarkEvaluator:
    """Evaluator for benchmark comparison"""
    
    def __init__(self, test_loader):
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = CombinedLoss()
        
    @torch.no_grad()
    def evaluate_model(self, model, model_name):
        """Evaluate a single model"""
        print(f"\nEvaluating {model_name}...")
        
        model.to(self.device)
        model.eval()
        
        metrics = SegmentationMetrics()
        total_loss = 0.0
        
        pbar = tqdm(self.test_loader, desc=f'Evaluating {model_name}')
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            outputs = model(images)
            loss = self.criterion(outputs, masks)
            
            total_loss += loss.item()
            metrics.update(outputs, masks)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Get metrics
        avg_loss = total_loss / len(self.test_loader)
        result_metrics = metrics.get_metrics()
        result_metrics['loss'] = avg_loss
        
        # Add model info
        model_info = get_model_info(model)
        result_metrics.update(model_info)
        
        print(f"Results for {model_name}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  IoU: {result_metrics['iou']:.4f}")
        print(f"  Dice: {result_metrics['dice']:.4f}")
        print(f"  Precision: {result_metrics['precision']:.4f}")
        print(f"  Recall: {result_metrics['recall']:.4f}")
        print(f"  F1: {result_metrics['f1']:.4f}")
        print(f"  Parameters: {result_metrics['total_params_M']:.2f}M")
        
        return result_metrics


def run_benchmark_comparison():
    """Run complete benchmark comparison"""
    print("="*80)
    print("BENCHMARK COMPARISON")
    print("Comparing ResNet50-UNet with other segmentation models")
    print("="*80)
    
    # Get test dataloader
    _, _, test_loader, _ = get_dataloaders()
    
    # Initialize evaluator
    evaluator = BenchmarkEvaluator(test_loader)
    
    # Results dictionary
    results = {}
    
    # 1. Evaluate our ResNet50-UNet (load best checkpoint)
    print("\n[1/6] Evaluating ResNet50-UNet (Ours)")
    our_model = ResNet50UNet(in_channels=3, out_channels=1, pretrained=False)
    
    # Try to load best checkpoint
    best_checkpoint = CHECKPOINTS_DIR / 'best_model.pth'
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location='cpu')
        our_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print("Warning: No trained checkpoint found. Using random initialization.")
    
    results['ResNet50-UNet (Ours)'] = evaluator.evaluate_model(our_model, 'ResNet50-UNet')
    
    # 2. Evaluate benchmark models
    benchmark_models = get_all_benchmark_models()
    
    for i, (model_name, model) in enumerate(benchmark_models.items(), start=2):
        print(f"\n[{i}/{len(benchmark_models)+1}] Evaluating {model_name}")
        results[model_name] = evaluator.evaluate_model(model, model_name)
    
    # Create results directory
    benchmark_results_dir = RESULTS_ROOT / 'benchmark_comparison'
    benchmark_results_dir.mkdir(parents=True, exist_ok=True)
    
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
        f.write("Comparison of ResNet50-UNet with other state-of-the-art segmentation models ")
        f.write("on HyperKvasir dataset.\n\n")
        
        f.write("## Results\n\n")
        f.write("### Performance Metrics\n\n")
        
        # Create markdown table
        f.write("| Model | IoU | Dice | Precision | Recall | F1 | Params (M) |\n")
        f.write("|-------|-----|------|-----------|--------|----|-----------|\n")
        
        for model_name, metrics in results.items():
            f.write(f"| {model_name} | ")
            f.write(f"{metrics['iou']:.4f} | ")
            f.write(f"{metrics['dice']:.4f} | ")
            f.write(f"{metrics['precision']:.4f} | ")
            f.write(f"{metrics['recall']:.4f} | ")
            f.write(f"{metrics['f1']:.4f} | ")
            f.write(f"{metrics['total_params_M']:.2f} |\n")
        
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
        
        # Compare our model with others
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
    args = parser.parse_args()
    

    experiment_manager = ExperimentManager('benchmark_comparison', experiment_name=args.experiment_name)
    print(f"Using Experiment Manager: {args.experiment_name}")
    experiment_manager.update_status('running')
    
    try:
        results, df = run_benchmark_comparison()
        
        if experiment_manager:
            experiment_manager.save_results(results)
            experiment_manager.update_status('completed')
    except Exception as e:
        if experiment_manager:
            experiment_manager.handle_error(e, context="Benchmark Comparison")
        raise


if __name__ == '__main__':
    main()