"""
Benchmark Models for Comparison
Sử dụng segmentation_models_pytorch library
"""
import segmentation_models_pytorch as smp
from config import BENCHMARK_MODELS


MODEL_BUILDERS = {
    'unet': 'Unet',
    'unetplusplus': 'UnetPlusPlus',
    'resunetplusplus': 'UnetPlusPlus',
    'deeplabv3plus': 'DeepLabV3Plus',
    'fpn': 'FPN',
    'pspnet': 'PSPNet',
    'linknet': 'Linknet',
    'pan': 'PAN',
    'segformer': 'Segformer',
    'swin_unet': 'Unet',
}


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
    
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Model {model_name} not supported. "
            f"Choose from: {', '.join(sorted(MODEL_BUILDERS.keys()))}"
        )
    
    builder_name = MODEL_BUILDERS[model_name]
    if not hasattr(smp, builder_name):
        raise ValueError(f"Model builder {builder_name} unavailable in segmentation_models_pytorch version")

    model_builder = getattr(smp, builder_name)
    model = model_builder(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation='sigmoid'
    )

    return model


def get_all_benchmark_models():
    """Get all benchmark models for comparison"""
    models = {}
    
    for model_name, config in BENCHMARK_MODELS.items():
        encoder = config.get('encoder', 'resnet34')
        weights = config.get('encoder_weights', 'imagenet')
        
        display_name = config.get('display_name', f"{model_name}_{encoder}")

        models[display_name] = get_benchmark_model(
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