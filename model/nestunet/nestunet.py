import torch
from torch import nn

from .nestunet_core import *

class NestedUNet(nn.Module):
    def __init__(self, args):
        super(NestedUNet, self).__init__()

        self.seg_n_classes = args.seg_n_classes
        self.init_ch = args.init_ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = Conv_Block_Nested(3, self.init_ch, self.init_ch)
        self.conv1_0 = Conv_Block_Nested(self.init_ch, self.init_ch*2, self.init_ch*2)
        self.conv2_0 = Conv_Block_Nested(self.init_ch*2, self.init_ch*4, self.init_ch*4)
        self.conv3_0 = Conv_Block_Nested(self.init_ch*4, self.init_ch*8, self.init_ch*8)
        self.conv4_0 = Conv_Block_Nested(self.init_ch*8, self.init_ch*16, self.init_ch*16)

        self.conv0_1 = Conv_Block_Nested(self.init_ch + self.init_ch*2, self.init_ch, self.init_ch)
        self.conv1_1 = Conv_Block_Nested(self.init_ch*2 + self.init_ch*4, self.init_ch*2, self.init_ch*2)
        self.conv2_1 = Conv_Block_Nested(self.init_ch*4 + self.init_ch*8, self.init_ch*4, self.init_ch*4)
        self.conv3_1 = Conv_Block_Nested(self.init_ch*8 + self.init_ch*16, self.init_ch*8, self.init_ch*8)

        self.conv0_2 = Conv_Block_Nested(self.init_ch*2 + self.init_ch*2, self.init_ch, self.init_ch)
        self.conv1_2 = Conv_Block_Nested(self.init_ch*2*2 + self.init_ch*4, self.init_ch*2, self.init_ch*2)
        self.conv2_2 = Conv_Block_Nested(self.init_ch*4*2 + self.init_ch*8, self.init_ch*4, self.init_ch*4)

        self.conv0_3 = Conv_Block_Nested(self.init_ch*3 + self.init_ch*2, self.init_ch, self.init_ch)
        self.conv1_3 = Conv_Block_Nested(self.init_ch*2*3 + self.init_ch*4, self.init_ch*2, self.init_ch*2)

        self.conv0_4 = Conv_Block_Nested(self.init_ch*4 + self.init_ch*2, self.init_ch, self.init_ch)

        self.final = nn.Conv2d(self.init_ch, self.seg_n_classes, kernel_size=1)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output