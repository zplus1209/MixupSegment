"""
Configuration file cho HyperKvasir Segmentation Project
VERSION 2.0 - With timestamped experiments and better organization
"""
import os
from pathlib import Path
from datetime import datetime

def get_experiment_name(prefix="train"):
    """Generate unique experiment name with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data" / "hyper-kvasir"
RESULTS_ROOT = PROJECT_ROOT / "results"

# Dataset paths
LABELED_IMAGES = DATA_ROOT / "segmented-images" / "images"
LABELED_MASKS = DATA_ROOT / "segmented-images" / "masks"
UNLABELED_IMAGES = DATA_ROOT / "unlabeled-images"

# Base results paths
CHECKPOINTS_DIR = RESULTS_ROOT / "checkpoints"
LOGS_DIR = RESULTS_ROOT / "logs"
VISUALIZATIONS_DIR = RESULTS_ROOT / "visualizations"

# Create base directories
for path in [CHECKPOINTS_DIR, LOGS_DIR, VISUALIZATIONS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

def get_experiment_dirs(experiment_name):
    """Get experiment-specific directories"""
    exp_checkpoints = CHECKPOINTS_DIR / experiment_name
    exp_logs = LOGS_DIR / experiment_name
    exp_vis = VISUALIZATIONS_DIR / experiment_name
    
    # Create directories
    for path in [exp_checkpoints, exp_logs, exp_vis]:
        path.mkdir(parents=True, exist_ok=True)
    
    return {
        'checkpoints': exp_checkpoints,
        'logs': exp_logs,
        'visualizations': exp_vis
    }


MODEL_CONFIG = {
    'encoder': 'resnet50',
    'encoder_weights': 'imagenet',
    'in_channels': 3,
    'classes': 1,  # Binary segmentation
    'activation': 'sigmoid'
}


TRAIN_CONFIG = {
    # Basic training
    'batch_size': 8,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    
    # Mixup configuration
    'use_mixup': True,
    'mixup_alpha': 0.4,
    'mixup_prob': 0.5,
    
    # Unlabeled data
    'use_unlabeled': True,
    'unlabeled_ratio': 0.3,
    
    # Optimization
    'optimizer': 'adam',
    'scheduler': 'cosine',
    'warmup_epochs': 5,
    
    # Early stopping
    'patience': 15,
    'min_delta': 1e-4,
    
    # Checkpointing
    'save_every_n_epochs': 10,
    'keep_last_n_checkpoints': 3,
    
    # Hardware
    'num_workers': 4,
    'pin_memory': True,
    
    # Error recovery
    'auto_resume': True,  # Automatically resume from last checkpoint
    'backup_checkpoints': True,  # Keep backup of checkpoints
}


AUGMENTATION_CONFIG = {
    'train': {
        'resize': (384, 384),
        'horizontal_flip': 0.5,
        'vertical_flip': 0.5,
        'rotate_limit': 30,
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
    },
    'val': {
        'resize': (384, 384),
    }
}


EVAL_CONFIG = {
    'metrics': ['iou', 'dice', 'precision', 'recall', 'f1'],
    'save_predictions': True,
    'visualize_samples': 10,
}


BENCHMARK_MODELS = {
    'unet': {
        'display_name': 'U-Net',
        'encoder': 'resnet50',
        'encoder_weights': 'imagenet',
    },
    'unetplusplus': {
        'display_name': 'U-Net++',
        'encoder': 'resnet50',
        'encoder_weights': 'imagenet',
    },
    'resunetplusplus': {
        'display_name': 'ResU-Net++',
        'encoder': 'resnet50',
        'encoder_weights': 'imagenet',
    },
    'segformer': {
        'display_name': 'SegFormer',
        'encoder': 'mit_b2',
        'encoder_weights': 'imagenet',
    },
    'swin_unet': {
        'display_name': 'Swin Transformer',
        'encoder': 'tu-swin_base_patch4_window7_224',
        'encoder_weights': 'imagenet',
    },
    'deeplabv3plus': {
        'display_name': 'DeepLabV3+',
        'encoder': 'resnet50',
        'encoder_weights': 'imagenet',
    },
    'fpn': {
        'display_name': 'FPN',
        'encoder': 'resnet50',
        'encoder_weights': 'imagenet',
    }
}


ABLATION_EXPERIMENTS = {
    'mixup_off': {
        'use_mixup': False,
        'use_unlabeled': False,
        'mixup_alpha': 0.4,
        'mixup_prob': 0.5,
        'encoder': 'resnet50',
    },
    'mixup_on': {
        'use_mixup': True,
        'use_unlabeled': False,
        'mixup_alpha': 0.4,
        'mixup_prob': 0.5,
        'encoder': 'resnet50',
    },
    'encoder_vgg16': {
        'use_mixup': True,
        'use_unlabeled': False,
        'mixup_alpha': 0.4,
        'mixup_prob': 0.5,
        'encoder': 'vgg16',
    },
    'encoder_resnet50': {
        'use_mixup': True,
        'use_unlabeled': False,
        'mixup_alpha': 0.4,
        'mixup_prob': 0.5,
        'encoder': 'resnet50',
    },
    'encoder_inceptionv4': {
        'use_mixup': True,
        'use_unlabeled': False,
        'mixup_alpha': 0.4,
        'mixup_prob': 0.5,
        'encoder': 'inceptionv4',
    },
    'encoder_efficientnetb0': {
        'use_mixup': True,
        'use_unlabeled': False,
        'mixup_alpha': 0.4,
        'mixup_prob': 0.5,
        'encoder': 'efficientnet-b0',
    },
}


VIDEO_CONFIG = {
    'fps': 30,
    'output_format': 'mp4',
    'codec': 'mp4v',
    'overlay_alpha': 0.5,
    'color_map': 'jet',
}


VIS_CONFIG = {
    'dpi': 150,
    'figsize': (15, 5),
    'cmap': 'jet',
    'alpha': 0.5,
}


LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_logs': True,
}


EXPERIMENT_PRESETS = {
    'quick_test': {
        'num_epochs': 5,
        'batch_size': 4,
        'save_every_n_epochs': 1,
    },
    'full_training': {
        'num_epochs': 100,
        'batch_size': 8,
        'save_every_n_epochs': 10,
    },
    'ablation': {
        'num_epochs': 30,
        'batch_size': 8,
        'save_every_n_epochs': 5,
    }
}