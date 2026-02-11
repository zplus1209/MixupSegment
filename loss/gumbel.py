from typing import Any

from torch._tensor import Tensor
from .vanilla_clf_stable import VanillaClassifierStableV0
from .vanilla_seg_stable import VanillaSegmenterStableV0
import torch
from torch import nn
import torch.nn.functional as F


class GumbelSoftmax(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard
    
    def __call__(self, x) -> Any:
        return F.gumbel_softmax(x, tau=self.tau, hard=self.hard, dim=1)


class GumbelClassifierV0(VanillaClassifierStableV0):
    def __init__(self, args):
        super().__init__(args)

        self.act = GumbelSoftmax(args=args)
    
    def forward(self, pred, target) -> Tensor:
        return super().forward(pred, target)


class GumbelSegmenterV0(VanillaSegmenterStableV0):
    def __init__(self, args):
        super().__init__(args)

        self.act = GumbelSoftmax(args=args)
    
    def forward(self, pred, target) -> Tensor:
        return super().forward(pred, target)