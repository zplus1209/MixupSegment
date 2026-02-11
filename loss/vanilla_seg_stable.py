from .vanilla_clf_stable import VanillaClassifierStableV0
import torch
from torch import nn

class VanillaSegmenterStableV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)

    def forward(self, pred, target) -> torch.Tensor:
        logits = self.act(pred)

        B, _, H, W = tuple(logits.size())

        entropy = logits * target

        return (-1 / (B * H * W)) * torch.sum(entropy)
    

class VanillaSegmenterStableV1(VanillaSegmenterStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)
    
    def forward(self, pred, target) -> torch.Tensor:

        cls_loss = {}

        logits = self.act(pred)

        B, C, H, W = tuple(logits.size())

        _logits = logits.permute(0, 2, 3, 1).flatten(0, -2)
        _target = target.permute(0, 2, 3, 1).flatten(0, -2)

        for cidx in range(C):
            c_logits = _logits[_target[:, cidx] == 1]
            c_target = _target[_target[:, cidx] == 1]

            entropy = torch.sum(c_logits * c_target)

            cls_loss[cidx] = entropy

        return (-1 / (B * H * W)) * sum(list(cls_loss.values()))
    
class VanillaSegmenterStableV2(VanillaSegmenterStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        losses = torch.zeros(self.args.seg_n_classes).to(pred.device)

        _, C, _, _ = tuple(pred.size())

        for cidx in range(C):
            c_pred = pred[:, cidx]
            c_target = target[:, cidx]

            c_loss = self.loss_fn(c_pred, c_target)

            losses[cidx] = c_loss
        
        batch_weight = torch.ones_like(losses).to(pred.device)
        
        loss = torch.mul(losses, batch_weight).sum()

        return loss