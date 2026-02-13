import sys
import argparse
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))

from data import get_dataloaders
from models import get_all_benchmark_models
import experiments.train as train_module
from experiments.train import Trainer
from utils import ExperimentManager
from config import TRAIN_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Train all benchmark models')
    parser.add_argument('--experiment_name', type=str, default='benchmark_training')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)
    parser.add_argument('--use_mixup', action='store_true', default=False)
    parser.add_argument('--use_unlabeled', action='store_true', default=False)
    parser.add_argument('--only', type=str, default=None, help='Comma separated model display names')
    args = parser.parse_args()

    em = ExperimentManager('benchmark_training', experiment_name=args.experiment_name)
    em.update_status('running')

    train_loader, val_loader, _, unlabeled_loader = get_dataloaders(batch_size=args.batch_size)
    models = get_all_benchmark_models()

    only_set = None
    if args.only:
        only_set = {x.strip() for x in args.only.split(',') if x.strip()}

    summary = {}
    for model_name, model in models.items():
        if only_set and model_name not in only_set:
            continue

        print('\n' + '=' * 80)
        print(f'Training benchmark model: {model_name}')
        print('=' * 80)


        # isolate output folders per benchmark model
        model_slug = ''.join(c.lower() if c.isalnum() else '_' for c in model_name).strip('_')
        out_root = em.exp_dir / model_slug
        train_module.CHECKPOINTS_DIR = out_root / 'checkpoints'
        train_module.LOGS_DIR = out_root / 'logs'
        train_module.VISUALIZATIONS_DIR = out_root / 'visualizations'
        train_module.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        train_module.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        train_module.VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

        cfg = TRAIN_CONFIG.copy()
        cfg.update({
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'use_mixup': args.use_mixup,
            'use_unlabeled': args.use_unlabeled,
            'mixup_alpha': args.mixup_alpha,
            'model_name': model_name,
        })

        trainer = Trainer(model, train_loader, val_loader, unlabeled_loader, cfg)
        history = trainer.train()

        summary[model_name] = {
            'best_val_dice': float(trainer.best_val_dice),
            'epochs': int(args.epochs),
            'mixup_alpha': float(args.mixup_alpha),
            'use_mixup': bool(args.use_mixup),
            'use_unlabeled': bool(args.use_unlabeled),
            'final_train_loss': float(history['train_loss'][-1]) if history['train_loss'] else None,
            'final_val_loss': float(history['val_loss'][-1]) if history['val_loss'] else None,
        }

    em.save_results(summary)
    em.update_status('completed')


if __name__ == '__main__':
    main()