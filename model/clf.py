import os, sys
import torch
from torch import nn
from torchvision import models
from .resnet_sc import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


def CLF(args):
    if args.model == 'resnet18':
        return models.resnet18(num_classes = args.n_classes)
    elif args.model == 'base':
        return Base(num_classes=args.n_classes)
    elif args.model == 'resnet18_scratch':
        return ResNet18(num_classes = args.n_classes)
    else:
        raise ValueError(f"the backbone {args.model} is not supported in classification experiments")

class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        
        self.args = args
        self.num_classes = self.args.n_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(4 * 4 * 48, self.num_classes)


    def forward(self, x):
        x = self.layer1(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x

def get_resnet18(args):
    return models.resnet18(num_classes = args.n_classes)

def get_resnet18_scratch(args):
    return ResNet18(num_classes = args.n_classes)