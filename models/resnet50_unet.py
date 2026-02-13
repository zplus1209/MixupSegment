"""
ResNet50-UNet Model for Medical Image Segmentation
Sử dụng ResNet50 pretrained encoder kết hợp với UNet decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision import models
from torchvision.models import ResNet50_Weights

from config import MODEL_CONFIG


class ConvBlock(nn.Module):
    """Basic convolutional block: Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """UNet decoder block with skip connections"""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatch
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class ResNet50UNet(nn.Module):
    """
    ResNet50-UNet Architecture
    - Encoder: ResNet50 pretrained trên ImageNet
    - Decoder: UNet-style với skip connections
    """
    
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()
        
        # ResNet50 encoder
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Encoder layers
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # 64 channels
        self.pool1 = resnet.maxpool
        
        self.encoder2 = resnet.layer1  # 256 channels
        self.encoder3 = resnet.layer2  # 512 channels
        self.encoder4 = resnet.layer3  # 1024 channels
        self.encoder5 = resnet.layer4  # 2048 channels
        
        # Bottleneck
        self.bottleneck = ConvBlock(2048, 2048)
        
        # Decoder layers
        self.decoder5 = DecoderBlock(2048, 1024, 1024)
        self.decoder4 = DecoderBlock(1024, 512, 512)
        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 64, 128)
        
        # Final upsampling và output
        self.final_up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Activation
        self.activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # 64, H/2, W/2
        enc1_pool = self.pool1(enc1)  # 64, H/4, W/4
        
        enc2 = self.encoder2(enc1_pool)  # 256, H/4, W/4
        enc3 = self.encoder3(enc2)  # 512, H/8, W/8
        enc4 = self.encoder4(enc3)  # 1024, H/16, W/16
        enc5 = self.encoder5(enc4)  # 2048, H/32, W/32
        
        # Bottleneck
        bottleneck = self.bottleneck(enc5)
        
        # Decoder với skip connections
        dec5 = self.decoder5(bottleneck, enc4)  # 1024, H/16, W/16
        dec4 = self.decoder4(dec5, enc3)  # 512, H/8, W/8
        dec3 = self.decoder3(dec4, enc2)  # 256, H/4, W/4
        dec2 = self.decoder2(dec3, enc1)  # 128, H/2, W/2
        
        # Final output
        out = self.final_up(dec2)  # 64, H, W
        out = self.final_conv(out)  # out_channels, H, W
        out = self.activation(out)
        
        return out
    
    def get_encoder_params(self):
        """Get encoder parameters for differential learning rates"""
        return list(self.encoder1.parameters()) + \
               list(self.encoder2.parameters()) + \
               list(self.encoder3.parameters()) + \
               list(self.encoder4.parameters()) + \
               list(self.encoder5.parameters())
    
    def get_decoder_params(self):
        """Get decoder parameters for differential learning rates"""
        return list(self.bottleneck.parameters()) + \
               list(self.decoder5.parameters()) + \
               list(self.decoder4.parameters()) + \
               list(self.decoder3.parameters()) + \
               list(self.decoder2.parameters()) + \
               list(self.final_up.parameters()) + \
               list(self.final_conv.parameters())


def get_model(encoder='resnet50', pretrained=True, in_channels=3, out_channels=1):
    """Factory function to create segmentation model."""
    if encoder == 'resnet50':
        return ResNet50UNet(in_channels, out_channels, pretrained)

    # SMP does not support `inceptionv3`; map to closest available backbone.
    encoder_alias = {
        'inceptionv3': 'inceptionv4',
    }
    resolved_encoder = encoder_alias.get(encoder, encoder)

    encoder_weights = 'imagenet' if pretrained else None
    return smp.Unet(
        encoder_name=resolved_encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
        activation='sigmoid'
    )

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)