"""End-to-end VTON pipeline: warping + generation."""
import torch
import torch.nn as nn

from .garment_encoder import GarmentEncoder
from .warping import CorrelationNet, warp_garment
from .unet import UNet


class VTONPipeline(nn.Module):
    def __init__(self, garment_feat_dim: int = 512):
        super().__init__()
        self.garment_encoder = GarmentEncoder(out_dim=garment_feat_dim)
        self.person_encoder = GarmentEncoder(out_dim=garment_feat_dim, freeze_backbone=False)
        self.warp_net = CorrelationNet(feature_dim=garment_feat_dim)
        # UNet takes: person (3) + warped garment (3) + agnostic mask (1) = 7 channels
        self.generator = UNet(in_channels=7, out_channels=3)

    def forward(
        self,
        person: torch.Tensor,       # (B, 3, H, W)
        garment: torch.Tensor,      # (B, 3, H, W)
        agnostic_mask: torch.Tensor # (B, 1, H, W)
    ) -> dict:
        p_feat = self.person_encoder(person)
        g_feat = self.garment_encoder(garment)

        theta = self.warp_net(p_feat, g_feat)
        warped = warp_garment(garment, theta)

        gen_input = torch.cat([person, warped, agnostic_mask], dim=1)
        result = self.generator(gen_input)

        return {"result": result, "warped_garment": warped, "theta": theta}
