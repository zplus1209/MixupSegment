from .cifar_resnet_1 import resnet18 as resnet18_cifar_variant1
from .cifar_resnet_2 import ResNet18 as resnet18_cifar_variant2






_CUSTOM_NO_PRETRAIN = {
    "resnet18_cifar_variant1", "resnet18_cifar_variant2",
    "conv_tokenizer_tiny", "conv_tokenizer_small", "hybrid_convvit_tiny"
}
