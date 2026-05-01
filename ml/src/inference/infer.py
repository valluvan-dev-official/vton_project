"""Single-image inference using a trained VTONPipeline checkpoint."""
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from src.models.pipeline import VTONPipeline
from src.utils.checkpoint import load_checkpoint


class VTONInference:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VTONPipeline().to(self.device)
        load_checkpoint(checkpoint_path, self.model)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def run(self, person_path: str, garment_path: str, output_path: str) -> str:
        person = self.transform(Image.open(person_path).convert("RGB")).unsqueeze(0).to(self.device)
        garment = self.transform(Image.open(garment_path).convert("RGB")).unsqueeze(0).to(self.device)
        mask = torch.ones(1, 1, 512, 512, device=self.device)

        out = self.model(person, garment, mask)
        result_tensor = out["result"][0].cpu().clamp(-1, 1) * 0.5 + 0.5
        result_img = T.ToPILImage()(result_tensor)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result_img.save(output_path)
        return output_path
