import torch
from torch import nn 

from .runet_core import *

class RUnet(nn.Module):
    def __init__(self, args):
        super(RUnet, self).__init__()

        self.seg_n_classes = args.seg_n_classes
        self.init_ch = args.init_ch

        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.upsample = nn.Upsample(scale_factor = 2)
  
        self.encoder = nn.ModuleList([            
            RRCNN_block(3, self.init_ch, 2),
            RRCNN_block(self.init_ch, self.init_ch * 2, 2),
            RRCNN_block(self.init_ch * 2, self.init_ch * 4, 2),
            RRCNN_block(self.init_ch * 4 , self.init_ch * 8, 2),
            RRCNN_block(self.init_ch * 8 , self.init_ch * 16, 2)
        ])

        self.decoder = nn.ModuleList([
            UpConv(self.init_ch*16, self.init_ch*8),
            RRCNN_block(self.init_ch *16 , self.init_ch * 8, 2),

            UpConv(self.init_ch*8, self.init_ch*4),
            RRCNN_block(self.init_ch *8 , self.init_ch * 4, 2),

            UpConv(self.init_ch*4, self.init_ch*2),
            RRCNN_block(self.init_ch *4 , self.init_ch * 2, 2),
                
            UpConv(self.init_ch*2, self.init_ch),
            RRCNN_block(self.init_ch *2 , self.init_ch, 2),
        ])

        self.conv = nn.Conv2d(self.init_ch, self.seg_n_classes, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x1 = self.encoder[0](x)

        x2 = self.maxpool(x1)
        x2 = self.encoder[1](x2)

        x3 = self.maxpool(x2)
        x3 = self.encoder[2](x3)
        
        x4 = self.maxpool(x3)
        x4 = self.encoder[3](x4)

        x5 = self.maxpool(x4)
        x5 = self.encoder[4](x5)
        
        d5 = self.decoder[0](x5)
        d5 = torch.cat((x4, d5), dim = 1)
        d5 = self.decoder[1](d5)

        d4 = self.decoder[2](d5)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.decoder[3](d4)   

        d3 = self.decoder[4](d4)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.decoder[5](d3)   

        d2 = self.decoder[6](d3)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.decoder[7](d2)  

        out = self.conv(d2)   

        return out