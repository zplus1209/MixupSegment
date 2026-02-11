import os, sys
import torch
from torch import nn 
import torch.nn.functional as F


class NaClassifierV0(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.temp = args.na_alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = args.idx)
        self.weight = nn.Parameter(torch.tensor([self.temp]*self.args.n_classes, device=self.device))

    def forward(self, pred, target) -> torch.Tensor:
        loss = F.cross_entropy(input=pred, target=target, weight=1/(2*self.weight.exp()))
        return loss

class NaSegmenterV0(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = args.idx)
        self.weight = nn.Parameter(torch.tensor([args.na_alpha]*args.seg_n_classes, device=self.device))

    def forward(self, pred, target) -> torch.Tensor:
        loss = F.cross_entropy(input=pred, target=target, weight=1/(2*self.weight.exp()))
        return loss