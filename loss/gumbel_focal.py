from .vanilla_clf_stable import VanillaClassifierStableV0
import torch
import torch.nn.functional as F

class GumbelFocalClassifierV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.gamma = args.gamma
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard
    
    def forward(self, pred, target) -> torch.Tensor:

        logits = F.gumbel_softmax(pred, tau=self.tau, hard=self.hard, dim=1)

        B, C = tuple(logits.size())

        entropy = torch.pow(1 - logits, self.gamma) * torch.log(logits) * F.one_hot(target, num_classes=C).float()

        return (-1 / B) * torch.sum(entropy)

class GumbelFocalSegmenterV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.gamma = args.gamma
        self.tau = args.gumbel_tau
        self.hard = args.gumbel_hard

    def forward(self, pred, target) -> torch.Tensor:
        logits = F.gumbel_softmax(pred, tau=self.tau, hard=self.hard, dim=1)

        B, C, H, W = tuple(logits.size())

        entropy = torch.pow(1 - logits, self.gamma) * torch.log(logits) * target

        return (-1 / (B * H * W)) * torch.sum(entropy)