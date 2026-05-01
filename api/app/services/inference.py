"""
Inference router — pluggable try-on engine.
Currently uses a placeholder renderer. Set use_own_model = True (or call
switch_to_own_model()) once the real model is ready; no callers change.
"""
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


class InferenceRouter:
    def __init__(self):
        self.use_own_model: bool = False
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, person_image_path: str, garment_image_path: str, output_path: str) -> str:
        """
        Run inference. Returns output_path.
        Simulates a 3-second GPU job with a labeled composite image.
        """
        if self.use_own_model and self._model is not None:
            return self._run_own_model(person_image_path, garment_image_path, output_path)
        return self._run_placeholder(person_image_path, garment_image_path, output_path)

    def switch_to_own_model(self, model_path: str) -> None:
        """Load the trained model and activate it for all subsequent runs."""
        self._model = self._load_model(model_path)
        self.use_own_model = True

    # ------------------------------------------------------------------
    # Placeholder implementation
    # ------------------------------------------------------------------

    def _run_placeholder(self, person_path: str, garment_path: str, output_path: str) -> str:
        time.sleep(3)  # simulate GPU processing time

        person_img = Image.open(person_path).convert("RGB").resize((512, 512))
        garment_img = Image.open(garment_path).convert("RGB").resize((256, 256))

        # Paste garment as an overlay to simulate a try-on result
        canvas = person_img.copy()
        canvas.paste(garment_img, (128, 100))

        draw = ImageDraw.Draw(canvas)
        draw.rectangle([(0, 0), (512, 30)], fill=(0, 0, 0, 180))
        draw.text((10, 8), "VTON Placeholder Result", fill="white")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        return output_path

    # ------------------------------------------------------------------
    # Own-model stub (wire up real model here)
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str):
        """Load model weights from model_path. Replace with real loader."""
        raise NotImplementedError("Wire up your trained model here.")

    def _run_own_model(self, person_path: str, garment_path: str, output_path: str) -> str:
        """Run the loaded model. Replace with real inference call."""
        raise NotImplementedError("Wire up your trained model inference here.")


# Module-level singleton
_router: InferenceRouter | None = None


def get_inference_router() -> InferenceRouter:
    global _router
    if _router is None:
        _router = InferenceRouter()
    return _router
