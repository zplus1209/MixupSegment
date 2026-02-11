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
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2
import torch.nn as nn

# def get_backbone(backbone, castrate=True):
#     backbone = eval(f"{backbone}()")

#     if castrate:
#         backbone.output_dim = backbone.fc.in_features
#         backbone.fc = torch.nn.Identity()

#     return backbone

def get_backbone(backbone, castrate=True):
    # khởi tạo model theo tên cấu hình (resnet18, resnet50, vgg19, …)
    net = eval(f"{backbone}()")

    if not castrate:
        return net

    # Trường hợp ResNet (có .fc)
    if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):
        net.output_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net

    # Trường hợp VGG / các backbone có .classifier (Sequential)
    if hasattr(net, "classifier"):
        # Tìm Linear cuối trong classifier (ví dụ vgg19: 4096->1000)
        last_linear_idx = None
        for i in reversed(range(len(net.classifier))):
            if isinstance(net.classifier[i], nn.Linear):
                last_linear_idx = i
                break
        if last_linear_idx is None:
            raise ValueError(f"Không tìm thấy Linear cuối trong classifier của backbone {backbone}")

        last_linear = net.classifier[last_linear_idx]
        # Đặt output_dim = in_features của Linear cuối (ví dụ 4096)
        net.output_dim = last_linear.in_features
        # Loại bỏ Linear cuối: giữ lại các lớp trước đó (ReLU/Dropout/Linear 4096, v.v.)
        net.classifier = nn.Sequential(*list(net.classifier.children())[:last_linear_idx])
        return net

    # Nếu rơi vào backbone khác chưa hỗ trợ
    raise ValueError(f"Backbone {backbone} chưa được hỗ trợ trong get_backbone()")

def get_model(model_cfg):    
    use_pretrained = bool(getattr(model_cfg, "pretrained_backbone", False))
    
    if model_cfg.name == 'simsiam':
        model =  SimSiam(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model






