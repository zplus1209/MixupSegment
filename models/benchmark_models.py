"""
Benchmark Models for Comparison
Sử dụng segmentation_models_pytorch library
"""
import segmentation_models_pytorch as smp
from config import BENCHMARK_MODELS


def get_benchmark_model(model_name, encoder_name=None, encoder_weights='imagenet', 
                        in_channels=3, classes=1):
    """
    Get benchmark model for comparison
    
    Args:
        model_name: 'unet', 'deeplabv3plus', 'fpn', 'pspnet', 'linknet', 'pan'
        encoder_name: backbone encoder (e.g., 'resnet34', 'resnet50', 'efficientnet-b0')
        encoder_weights: pretrained weights ('imagenet' or None)
        in_channels: number of input channels
        classes: number of output classes
    """
    if encoder_name is None:
        encoder_name = BENCHMARK_MODELS.get(model_name, {}).get('encoder', 'resnet34')
    
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid'
        )
    
    elif model_name == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid'
        )
    
    elif model_name == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid'
        )
    
    elif model_name == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid'
        )
    
    elif model_name == 'linknet':
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid'
        )
    
    elif model_name == 'pan':
        model = smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid'
        )
    
    else:
        raise ValueError(f"Model {model_name} not supported. "
                        f"Choose from: unet, deeplabv3plus, fpn, pspnet, linknet, pan")
    
    return model


def get_all_benchmark_models():
    """Get all benchmark models for comparison"""
    models = {}
    
    for model_name, config in BENCHMARK_MODELS.items():
        encoder = config.get('encoder', 'resnet34')
        weights = config.get('encoder_weights', 'imagenet')
        
        models[f"{model_name}_{encoder}"] = get_benchmark_model(
            model_name=model_name,
            encoder_name=encoder,
            encoder_weights=weights
        )
    
    return models


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model):
    """Get model information"""
    total_params = count_parameters(model)
    return {
        'total_params': total_params,
        'total_params_M': total_params / 1e6,
    }