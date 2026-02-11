import os, sys
import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import normal


class BLVSegmenterV0(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.sampler = normal.Normal(0, args.blv_s)

    def forward(self, pred, target) -> torch.Tensor:

        m_list = torch.sum(target, dim=[0, 2, 3]) + 0.0001

        frequency_list = torch.log(m_list)

        viariation = self.sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)

        pred = pred + (viariation.abs().permute(0, 2, 3, 1) / frequency_list.max() * frequency_list).permute(0, 3, 1, 2)

        loss = F.cross_entropy(pred, target)

        return loss