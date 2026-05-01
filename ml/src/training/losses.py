"""Loss functions for VTON training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class PerceptualLoss(nn.Module):
    """VGG-16 feature-space perceptual loss."""

    def __init__(self, layers: tuple = (3, 8, 15, 22)):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT).features
        self.slices = nn.ModuleList()
        prev = 0
        for l in layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:l]))
            prev = l
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        x, y = pred, target
        for s in self.slices:
            x, y = s(x), s(y)
            loss = loss + F.l1_loss(x, y)
        return loss


class WarpingLoss(nn.Module):
    """Regularisation on TPS theta to prevent extreme deformations."""

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.mean(theta ** 2)


class VTONLoss(nn.Module):
    def __init__(self, lambda_l1: float = 1.0, lambda_perc: float = 0.1, lambda_warp: float = 0.01):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()
        self.warp_reg = WarpingLoss()
        self.lw = {"l1": lambda_l1, "perc": lambda_perc, "warp": lambda_warp}

    def forward(self, pred: torch.Tensor, target: torch.Tensor, theta: torch.Tensor) -> dict:
        l1 = self.l1(pred, target)
        perc = self.perceptual(pred, target)
        warp = self.warp_reg(theta)
        total = self.lw["l1"] * l1 + self.lw["perc"] * perc + self.lw["warp"] * warp
        return {"total": total, "l1": l1, "perceptual": perc, "warp_reg": warp}
