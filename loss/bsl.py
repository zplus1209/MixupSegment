import os, sys
import torch
from torch import nn 
import torch.nn.functional as F

from .vanilla_clf_stable import VanillaClassifierStableV0


class BSLClassifierV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)
        
    def forward(self, pred, target) -> torch.Tensor:
        
        pred_exp = torch.exp(pred)
        
        B, C = tuple(pred.size())
        
        class_freq = torch.zeros(C).to(pred.device)

        for idx, _cls in enumerate(range(C)):
            cnt = (target == _cls).sum()
            class_freq[idx] = cnt

        class_scaled_pred = pred_exp * class_freq
        
        logits = self.act(class_scaled_pred)

        entropy = logits * F.one_hot(target, num_classes=C).float()

        return (-1 / B) * torch.sum(entropy)

class BSLSegmenterV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)

    def forward(self, pred, target) -> torch.Tensor:
        pred_exp = torch.exp(pred)

        B, C, H, W = tuple(pred.size())
        
        _pred_exp = pred_exp.permute(0, 2, 3, 1).flatten(0, -2)
        _target = target.permute(0, 2, 3, 1).flatten(0, -2)
        
        class_freq = torch.zeros(C).to(pred.device)

        for idx, _cls in enumerate(range(C)):
            cnt = (target.argmax(1) == _cls).sum()
            class_freq[idx] = cnt
        
        class_scaled_pred = _pred_exp * class_freq
        
        logits = self.act(class_scaled_pred)

        entropy = logits * _target

        return (-1 / (B * H * W)) * torch.sum(entropy)