"""Garment feature encoder — extracts appearance features for the try-on pipeline."""
import torch
import torch.nn as nn
import torchvision.models as tvm


class GarmentEncoder(nn.Module):
    def __init__(self, out_dim: int = 512, freeze_backbone: bool = True):
        super().__init__()
        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # (B, 2048, H/32, W/32)
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.proj(feats)
