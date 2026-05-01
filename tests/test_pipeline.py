"""Integration tests for the VTON pipeline."""
import os
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_image(path: str, size: tuple = (256, 256), color: tuple = (128, 64, 200)):
    img = Image.fromarray(np.full((*size, 3), color, dtype=np.uint8))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------

class TestLocalStorageBackend:
    def test_save_and_load(self, tmp_path):
        from api.app.services.storage import LocalStorageBackend

        backend = LocalStorageBackend(base_path=str(tmp_path / "store"))
        src = str(tmp_path / "source.jpg")
        make_test_image(src)

        stored = backend.save(src, "inputs/test.jpg")
        assert backend.exists("inputs/test.jpg")

        dest = str(tmp_path / "loaded.jpg")
        backend.load("inputs/test.jpg", dest)
        assert Path(dest).exists()

    def test_delete(self, tmp_path):
        from api.app.services.storage import LocalStorageBackend

        backend = LocalStorageBackend(base_path=str(tmp_path / "store"))
        src = str(tmp_path / "img.jpg")
        make_test_image(src)
        backend.save(src, "del/img.jpg")
        assert backend.exists("del/img.jpg")
        backend.delete("del/img.jpg")
        assert not backend.exists("del/img.jpg")


# ---------------------------------------------------------------------------
# Inference tests
# ---------------------------------------------------------------------------

class TestInferenceRouter:
    def test_placeholder_run(self, tmp_path):
        from api.app.services.inference import InferenceRouter

        router = InferenceRouter()
        person = make_test_image(str(tmp_path / "person.jpg"))
        garment = make_test_image(str(tmp_path / "garment.jpg"), color=(200, 100, 50))
        output = str(tmp_path / "output.jpg")

        with patch("time.sleep"):  # skip 3-sec delay in tests
            result = router.run(person, garment, output)

        assert result == output
        assert Path(output).exists()

    def test_switch_to_own_model_raises_not_implemented(self, tmp_path):
        from api.app.services.inference import InferenceRouter

        router = InferenceRouter()
        with pytest.raises(NotImplementedError):
            router.switch_to_own_model("nonexistent.pt")


# ---------------------------------------------------------------------------
# SSIM quality threshold test
# ---------------------------------------------------------------------------

class TestSSIMThreshold:
    def test_identical_images_high_ssim(self, tmp_path):
        from skimage.metrics import structural_similarity as ssim
        import numpy as np

        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        score, _ = ssim(img, img, full=True, channel_axis=2, data_range=255)
        assert score >= 0.65

    def test_random_noise_low_ssim(self, tmp_path):
        from skimage.metrics import structural_similarity as ssim
        import numpy as np

        rng = np.random.default_rng(42)
        a = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        b = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        score, _ = ssim(a, b, full=True, channel_axis=2, data_range=255)
        assert score < 0.65


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestVTONDataset:
    def test_discovers_pairs(self, tmp_path):
        pair_dir = tmp_path / "training_pairs" / "job_001"
        pair_dir.mkdir(parents=True)
        make_test_image(str(pair_dir / "person.jpg"))
        make_test_image(str(pair_dir / "garment.jpg"))
        make_test_image(str(pair_dir / "result.jpg"))
        (pair_dir / "meta.json").write_text(json.dumps({"job_id": "job_001", "ssim_score": 0.8}))

        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
        from src.data.dataset import VTONDataset

        ds = VTONDataset(str(tmp_path / "training_pairs"), image_size=64)
        assert len(ds) == 1
        sample = ds[0]
        assert "person" in sample
        assert "garment" in sample
        assert "result" in sample
