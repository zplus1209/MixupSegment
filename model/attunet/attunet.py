from lzma import FILTER_X86
import torch
from torch import nn 

from .attunet_core import *

class AttUnet(nn.Module):
    def __init__(self, args):
        super(AttUnet, self).__init__()

        self.seg_n_classes = args.seg_n_classes
        self.init_ch = args.init_ch

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.encoder = nn.ModuleList([            
            ConvBlock(3, self.init_ch), 
            ConvBlock(self.init_ch, self.init_ch*2),
            ConvBlock(self.init_ch*2, self.init_ch*4),
            ConvBlock(self.init_ch*4, self.init_ch*8),
            ConvBlock(self.init_ch*8, self.init_ch*16)
        ])

        self.decoder = nn.ModuleList([
            UpConv(self.init_ch*16, self.init_ch*8),
            ConvBlock(self.init_ch*16, self.init_ch*8),

            UpConv(self.init_ch*8, self.init_ch*4),
            ConvBlock(self.init_ch*8, self.init_ch*4),

            UpConv(self.init_ch*4, self.init_ch*2),
            ConvBlock(self.init_ch*4, self.init_ch*2),

            UpConv(self.init_ch*2, self.init_ch),
            ConvBlock(self.init_ch*2, self.init_ch),            
        ])

        self.Att1 = AttentionBlock(F_g=self.init_ch*8, F_l=self.init_ch*8, F_int=self.init_ch*4)
        self.Att2 = AttentionBlock(F_g=self.init_ch*4, F_l=self.init_ch*4, F_int=self.init_ch*2)
        self.Att3 = AttentionBlock(F_g=self.init_ch*2, F_l=self.init_ch*2, F_int=self.init_ch)
        self.Att4 = AttentionBlock(F_g=self.init_ch, F_l=self.init_ch, F_int=32)

        self.Conv = nn.Conv2d(self.init_ch, self.seg_n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e1 = self.encoder[0](x)

        e2 = self.Maxpool(e1)
        e2 = self.encoder[1](e2)    

        e3 = self.Maxpool(e2)
        e3 = self.encoder[2](e3)

        e4 = self.Maxpool(e3)
        e4 = self.encoder[3](e4)

        e5 = self.Maxpool(e4)
        e5 = self.encoder[4](e5)

        d5 = self.decoder[0](e5)     
        x4 = self.Att1(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.decoder[1](d5)

        d4 = self.decoder[2](d5)
        x3 = self.Att2(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.decoder[3](d4)

        d3 = self.decoder[4](d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.decoder[5](d3)

        d2 = self.decoder[6](d3)
        x1 = self.Att4(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.decoder[7](d2)

        out = self.Conv(d2)

        return out