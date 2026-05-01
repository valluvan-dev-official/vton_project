"""Validation loop and metrics."""
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import numpy as np


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) normalised tensor to uint8 HWC array."""
    img = (t.detach().cpu().clamp(-1, 1) * 0.5 + 0.5).numpy()
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


@torch.no_grad()
def validate(model, dataloader: DataLoader, loss_fn, device: str) -> dict:
    model.eval()
    total_loss, total_ssim, n = 0.0, 0.0, 0

    for batch in dataloader:
        person = batch["person"].to(device)
        garment = batch["garment"].to(device)
        result = batch["result"].to(device)
        mask = batch["agnostic_mask"].to(device)

        out = model(person, garment, mask)
        losses = loss_fn(out["result"], result, out["theta"])
        total_loss += losses["total"].item()

        for i in range(person.size(0)):
            pred_np = tensor_to_numpy(out["result"][i])
            gt_np = tensor_to_numpy(result[i])
            score, _ = ssim(pred_np, gt_np, full=True, channel_axis=2, data_range=255)
            total_ssim += score
        n += person.size(0)

    return {
        "val_loss": total_loss / max(len(dataloader), 1),
        "val_ssim": total_ssim / max(n, 1),
    }
