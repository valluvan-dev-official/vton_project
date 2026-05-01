"""Thin-plate spline warping module for garment deformation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrelationNet(nn.Module):
    """Predicts TPS control-point offsets from (person, garment) feature pairs."""

    def __init__(self, feature_dim: int = 512, num_control_points: int = 25):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_control_points * 2),
        )
        self.num_cp = num_control_points

    def forward(self, person_feat: torch.Tensor, garment_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([person_feat, garment_feat], dim=1)
        theta = self.regressor(combined)
        return theta.view(-1, self.num_cp, 2)


def warp_garment(garment: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Apply affine-approximated warp (placeholder for full TPS).
    theta: (B, 2, 3) affine parameters.
    """
    if theta.shape[1:] != (2, 3):
        # Reduce control-point offsets to a single affine matrix for the placeholder
        B = garment.size(0)
        theta = theta.mean(dim=1).view(B, -1)[:, :6].view(B, 2, 3)

    grid = F.affine_grid(theta, garment.size(), align_corners=False)
    return F.grid_sample(garment, grid, align_corners=False, padding_mode="border")
