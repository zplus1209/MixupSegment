import torch
from torch import nn
import torch.nn.functional as F

class FocalClassifierV0(nn.Module):
    def __init__(self, args) -> None:
        super(FocalClassifierV0, self).__init__()
        
        self.gamma = args.gamma
    
    def forward(self, pred, target) -> torch.Tensor:

        logits = F.log_softmax(pred, dim=1)

        B, C = tuple(logits.size())

        entropy = torch.pow(1 - logits.exp(), self.gamma) * logits * F.one_hot(target, num_classes=C).float()

        loss = (-1) * entropy.mean()

        return loss

class FocalSegmenterV0(nn.Module):
    def __init__(self, args) -> None:
        super(FocalSegmenterV0, self).__init__()
        
        self.gamma = args.gamma

    def forward(self, pred, target) -> torch.Tensor:
        logits = F.log_softmax(pred, dim=1)

        entropy = torch.pow(1 - logits.exp(), self.gamma) * logits * target

        loss = (-1) * entropy.mean()

        return loss