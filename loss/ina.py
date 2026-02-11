import os, sys
import torch
from torch import nn 
import torch.nn.functional as F

class INASegmenterV0(nn.Module):
    def __init__(self, args) -> None:
        super(INASegmenterV0, self).__init__()

        self.args = args
    
    def forward(self, pred, target) -> torch.Tensor:
        log_prob = F.log_softmax(pred/self.args.gumbel_tau, dim=1)

        inv_prob = (log_prob.exp() - 1) * self.args.seg_n_classes

        entropy = -(log_prob * target)/(2 * inv_prob.exp())

        return torch.mean(entropy)

class INAPSegmenterV0(nn.Module):
    def __init__(self, args) -> None:
        super(INAPSegmenterV0, self).__init__()

        self.args = args
    
    def forward(self, pred, target) -> torch.Tensor:
        log_prob = F.log_softmax(pred/self.args.gumbel_tau, dim=1)

        inv_prob = (log_prob.exp() - 1)

        entropy = -(log_prob * target)/(2 * inv_prob.exp())

        return torch.mean(entropy)