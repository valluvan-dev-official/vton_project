"""PyTorch Dataset that reads training pairs saved by the API worker."""
import json
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class VTONDataset(Dataset):
    """
    Reads training pairs from storage/training_pairs/.
    Each pair is a directory containing person.jpg, garment.jpg, result.jpg, meta.json.
    """

    def __init__(
        self,
        pairs_root: str,
        image_size: int = 512,
        transform: Callable | None = None,
        min_ssim: float = 0.0,
    ):
        self.root = Path(pairs_root)
        self.transform = transform or T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        self.pairs = self._discover_pairs(min_ssim)

    def _discover_pairs(self, min_ssim: float) -> list[dict]:
        pairs = []
        for meta_file in sorted(self.root.rglob("meta.json")):
            meta = json.loads(meta_file.read_text())
            if meta.get("ssim_score", 0) >= min_ssim:
                pairs.append({"dir": meta_file.parent, "meta": meta})
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        d = pair["dir"]

        person = self.transform(Image.open(next(d.glob("person.*"))).convert("RGB"))
        garment = self.transform(Image.open(next(d.glob("garment.*"))).convert("RGB"))
        result = self.transform(Image.open(next(d.glob("result.*"))).convert("RGB"))

        # Agnostic mask: derive from person by zeroing the torso region (placeholder)
        agnostic_mask = torch.ones(1, person.shape[1], person.shape[2])

        return {
            "person": person,
            "garment": garment,
            "result": result,
            "agnostic_mask": agnostic_mask,
            "job_id": pair["meta"].get("job_id", ""),
            "ssim_score": pair["meta"].get("ssim_score", 0.0),
        }
