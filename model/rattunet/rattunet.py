import torch
from torch import nn

from .rattunet_core import *


class RATTUnet(nn.Module):
    def __init__(self, args):
        super(RATTUnet, self).__init__()

        self.seg_n_classes = args.seg_n_classes
        self.init_ch = args.init_ch

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNN_block(3, self.init_ch, t=2)
        self.RRCNN2 = RRCNN_block(self.init_ch, self.init_ch*2, t=2)
        self.RRCNN3 = RRCNN_block(self.init_ch*2, self.init_ch*4, t=2)
        self.RRCNN4 = RRCNN_block(self.init_ch*4, self.init_ch*8, t=2)
        self.RRCNN5 = RRCNN_block(self.init_ch*8, self.init_ch*16, t=2)

        self.Up5 = UpConv(self.init_ch*16, self.init_ch*8)
        self.Att5 = Attention_block(F_g=self.init_ch*8, F_l=self.init_ch*8, F_int=self.init_ch*4)
        self.Up_RRCNN5 = RRCNN_block(self.init_ch*16, self.init_ch*8, t=2)

        self.Up4 = UpConv(self.init_ch*8, self.init_ch*4)
        self.Att4 = Attention_block(F_g=self.init_ch*4, F_l=self.init_ch*4, F_int=self.init_ch*2)
        self.Up_RRCNN4 = RRCNN_block(self.init_ch*8, self.init_ch*4, t=2)

        self.Up3 = UpConv(self.init_ch*4, self.init_ch*2)
        self.Att3 = Attention_block(F_g=self.init_ch*2, F_l=self.init_ch*2, F_int=self.init_ch)
        self.Up_RRCNN3 = RRCNN_block(self.init_ch*4, self.init_ch*2, t=2)

        self.Up2 = UpConv(self.init_ch*2, self.init_ch)
        self.Att2 = Attention_block(F_g=self.init_ch, F_l=self.init_ch, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(self.init_ch*2, self.init_ch, t=2)

        self.Conv = nn.Conv2d(self.init_ch, self.seg_n_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        return out