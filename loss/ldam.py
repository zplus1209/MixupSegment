import os, sys
import torch
from torch import nn 
import torch.nn.functional as F


class LDAMSegmenterV0(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.max_m = args.ldam_max_m
        self.s = args.ldam_s

    def forward(self, pred, target) -> torch.Tensor:
        m_list = torch.sum(target, dim=[0, 2, 3]) + 0.0001
        m_list = 1.0 / torch.sqrt(torch.sqrt(m_list))
        m_list = m_list * (self.max_m / torch.max(m_list))

        _pred = pred.permute(0, 2, 3, 1).flatten(0, -2)
        _target = target.permute(0, 2, 3, 1).flatten(0, -2).argmax(1).long()

        index = torch.zeros_like(_pred, dtype=torch.uint8)
        index.scatter_(1, _target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = _pred - batch_m
    
        output = torch.where(index, x_m, _pred)
        return F.cross_entropy(self.s*output, _target)