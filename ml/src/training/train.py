"""
Main training loop.
Reads training pairs from storage/training_pairs/ (written by the API Celery worker).
Saves checkpoints to ml/checkpoints/.

Usage:
    python -m src.training.train --config configs/train_full.yaml
"""
import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.data.dataset import VTONDataset
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.models.pipeline import VTONPipeline
from src.training.losses import VTONLoss
from src.training.validation import validate
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_dataloaders(cfg: dict):
    pairs_root = cfg["data"]["training_pairs_path"]
    image_size = cfg["data"].get("image_size", 512)
    val_split = cfg["data"].get("val_split", 0.1)
    batch_size = cfg["training"].get("batch_size", 4)

    full_dataset = VTONDataset(
        pairs_root=pairs_root,
        image_size=image_size,
        min_ssim=cfg["data"].get("min_ssim", 0.0),
    )
    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def train(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device}")

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "ml/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = VTONPipeline().to(device)
    loss_fn = VTONLoss(
        lambda_l1=cfg["training"].get("lambda_l1", 1.0),
        lambda_perc=cfg["training"].get("lambda_perc", 0.1),
        lambda_warp=cfg["training"].get("lambda_warp", 0.01),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"].get("lr", 1e-4),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"].get("epochs", 50)
    )

    start_epoch = 0
    resume = cfg.get("resume_checkpoint")
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer)
        logger.info(f"Resumed from epoch {start_epoch}")

    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    for epoch in range(start_epoch, cfg["training"].get("epochs", 50)):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            person = batch["person"].to(device)
            garment = batch["garment"].to(device)
            result = batch["result"].to(device)
            mask = batch["agnostic_mask"].to(device)

            optimizer.zero_grad()
            out = model(person, garment, mask)
            losses = loss_fn(out["result"], result, out["theta"])
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += losses["total"].item()
            if step % 50 == 0:
                logger.info(
                    f"Epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={losses['total'].item():.4f} "
                    f"l1={losses['l1'].item():.4f} "
                    f"perc={losses['perceptual'].item():.4f}"
                )

        scheduler.step()
        val_metrics = validate(model, val_loader, loss_fn, device)
        logger.info(
            f"Epoch {epoch} done. avg_loss={epoch_loss/len(train_loader):.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"val_ssim={val_metrics['val_ssim']:.4f}"
        )

        save_checkpoint(
            checkpoint_dir / f"epoch_{epoch:04d}.pt",
            model, optimizer, epoch, val_metrics
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_full.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
