import os, sys
import torch
from torch import nn
import torch.nn.functional as F

class Recurrent_block(nn.Module):
    def __init__(self, out_ch, t = 2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out
    
class RRCNN_block(nn.Module):
    def __init__(self, in_ch, out_ch, t = 2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t = t),
            Recurrent_block(out_ch, t = t)
        )
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out
    
class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.up(x)
        return x