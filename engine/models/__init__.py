from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from torchvision.models import \
    resnet50, \
    resnet18, \
    vgg19, \
    alexnet, \
    mobilenet_v2, \
    efficientnet_v2_l, \
    MaxVit
from .backbones.tokenizer import (
    conv_tokenizer_tiny,
    conv_tokenizer_small,
    hybrid_convvit_tiny,
)

import torchvision.models as tvm
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2
import torch.nn as nn

_TORCHVISION_WEIGHTS = {
    "resnet18":      lambda pre: tvm.ResNet18_Weights.DEFAULT if pre else None,
    "resnet50":      lambda pre: tvm.ResNet50_Weights.DEFAULT if pre else None,
    "vgg19":         lambda pre: tvm.VGG19_Weights.DEFAULT if pre else None,
    "alexnet":       lambda pre: tvm.AlexNet_Weights.DEFAULT if pre else None,
    "mobilenet_v2":  lambda pre: tvm.MobileNet_V2_Weights.DEFAULT if pre else None,
    "efficientnet_v2_l": lambda pre: tvm.EfficientNet_V2_L_Weights.DEFAULT if pre else None,
    # Nếu bạn dùng MaxViT của torchvision (maxvit_t), map thêm ở đây:
    # "maxvit_t":     lambda pre: tvm.MaxVit_T_Weights.DEFAULT if pre else None,
}

# Với một số backbone tuỳ biến (cifar variant) thường không có weights sẵn:
_CUSTOM_NO_PRETRAIN = {
    "resnet18_cifar_variant1", "resnet18_cifar_variant2",
    "conv_tokenizer_tiny", "conv_tokenizer_small", "hybrid_convvit_tiny"
}


def _build_torchvision_backbone(name: str, pretrained: bool) -> nn.Module:
    """Tạo backbone torchvision với/không pretrained một cách an toàn."""
    lname = name.lower()
    if lname not in _TORCHVISION_WEIGHTS and not hasattr(tvm, lname):
        raise ValueError(f"Không tìm thấy backbone '{name}' trong torchvision.models")

    # Lấy constructor từ torchvision.models theo tên
    ctor = getattr(tvm, lname, None)
    if ctor is None or not callable(ctor):
        raise ValueError(f"Constructor cho backbone '{name}' không hợp lệ trong torchvision")

    weights = _TORCHVISION_WEIGHTS.get(lname, lambda pre: None)(pretrained)
    try:
        # Hầu hết các model torchvision hiện đại đều nhận tham số 'weights='
        net = ctor(weights=weights)
    except TypeError:
        # Phòng trường hợp phiên bản torchvision cũ: dùng 'pretrained='
        net = ctor(pretrained=bool(pretrained))
    return net


_CUSTOM_REGISTRY = {
    "resnet18_cifar_variant1": resnet18_cifar_variant1,
    "resnet18_cifar_variant2": resnet18_cifar_variant2,
    "conv_tokenizer_tiny":     conv_tokenizer_tiny,
    "conv_tokenizer_small":    conv_tokenizer_small,
    "hybrid_convvit_tiny":     hybrid_convvit_tiny,
}

def get_backbone(backbone: str, castrate: bool = True, pretrained: bool = False) -> nn.Module:
    lname = backbone.lower()

    # custom qua registry (không dùng eval)
    if lname in _CUSTOM_REGISTRY:
        if pretrained:
            print(f"[WARN] '{backbone}' không hỗ trợ pretrained; sẽ khởi tạo ngẫu nhiên.")
        net = _CUSTOM_REGISTRY[lname]()
    else:
        net = _build_torchvision_backbone(lname, pretrained=pretrained)

    if not castrate:
        return net

    # 1) ResNet-style
    if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):
        net.output_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net

    # 2) Classifier = Linear
    if hasattr(net, "classifier") and isinstance(net.classifier, nn.Linear):
        net.output_dim = net.classifier.in_features
        net.classifier = nn.Identity()
        return net

    # 3) Classifier = Sequential (lấy Linear cuối)
    if hasattr(net, "classifier") and isinstance(net.classifier, nn.Sequential):
        for i in reversed(range(len(net.classifier))):
            if isinstance(net.classifier[i], nn.Linear):
                net.output_dim = net.classifier[i].in_features
                net.classifier = nn.Sequential(*list(net.classifier.children())[:i])
                return net

    # 4) Một số mô hình dùng .head hoặc .heads (vit-like)
    if hasattr(net, "head") and isinstance(getattr(net, "head"), nn.Linear):
        net.output_dim = net.head.in_features
        net.head = nn.Identity()
        return net
    if hasattr(net, "heads") and isinstance(getattr(net, "heads"), nn.Sequential):
        # lấy Linear cuối trong heads
        for i in reversed(range(len(net.heads))):
            if isinstance(net.heads[i], nn.Linear):
                net.output_dim = net.heads[i].in_features
                net.heads = nn.Sequential(*list(net.heads.children())[:i])
                return net

    raise ValueError(f"Backbone {backbone} chưa được hỗ trợ trong get_backbone()")


def get_model(model_cfg):    
    use_pretrained = bool(getattr(model_cfg, "pretrained_backbone", False))
    
    if model_cfg.name == 'simsiam':
        model =  SimSiam(get_backbone(model_cfg.backbone, castrate=True, pretrained=use_pretrained))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone, castrate=True, pretrained=use_pretrained))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone, castrate=True, pretrained=use_pretrained))
    elif model_cfg.name == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model






