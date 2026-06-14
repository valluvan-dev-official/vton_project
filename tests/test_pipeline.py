"""
Phase 1 integration tests — run with placeholder inference (no DCI-VTON needed).

TC-001  Health check
TC-003  Missing person image validation
TC-004  File too large validation
TC-005  Wrong file type validation
TC-008  Local storage save / retrieve
TC-009  Training pair auto-save (quality_score >= 0.65)
TC-010  Low quality result NOT saved
TC-012  Background async (placeholder always passes SSIM >= 0.65 for identical images)
"""
import os
import json
import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_test_image(path: str, size: tuple = (256, 256), color=(128, 64, 200)) -> str:
    img = Image.fromarray(np.full((*size, 3), color, dtype=np.uint8))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ── TC-008 / TC-009 / TC-010  Storage ─────────────────────────────────────────

class TestLocalStorageBackend:
    def test_save_load_exists_delete(self, tmp_path):
        from api.app.services.storage import LocalStorageBackend

        backend = LocalStorageBackend(base_path=str(tmp_path / "store"))
        src = make_test_image(str(tmp_path / "src.jpg"))

        stored = backend.save(src, "inputs/test.jpg")
        assert backend.exists("inputs/test.jpg")

        dest = str(tmp_path / "loaded.jpg")
        backend.load("inputs/test.jpg", dest)
        assert Path(dest).exists()

        backend.delete("inputs/test.jpg")
        assert not backend.exists("inputs/test.jpg")

    def test_url_returns_absolute_path(self, tmp_path):
        from api.app.services.storage import LocalStorageBackend

        backend = LocalStorageBackend(base_path=str(tmp_path / "store"))
        src = make_test_image(str(tmp_path / "img.jpg"))
        backend.save(src, "outputs/x.jpg")
        url = backend.url("outputs/x.jpg")
        assert "outputs/x.jpg" in url


# ── TC-001  Inference placeholder ─────────────────────────────────────────────

class TestInferencePlaceholder:
    def test_placeholder_creates_output_file(self, tmp_path):
        from api.app.services.inference import InferenceRouter

        router = InferenceRouter()
        person  = make_test_image(str(tmp_path / "person.jpg"))
        garment = make_test_image(str(tmp_path / "garment.jpg"), color=(200, 100, 50))
        output  = str(tmp_path / "output.jpg")

        with patch("time.sleep"):  # skip 3-sec delay
            result = router.run(person, garment, output)

        assert result == output
        assert Path(output).exists()
        img = Image.open(output)
        assert img.size == (512, 512)

    def test_own_model_not_loaded_by_default(self):
        from api.app.services.inference import InferenceRouter
        router = InferenceRouter()
        assert router.use_own_model is False

    def test_switch_to_own_model_raises_not_implemented(self):
        from api.app.services.inference import InferenceRouter
        router = InferenceRouter()
        with pytest.raises(NotImplementedError):
            router.switch_to_own_model("nonexistent.pt")


# ── TC-003 / TC-004 / TC-005  Route validation ────────────────────────────────

class TestTryonValidation:
    """
    These test the validation helpers directly — no running server needed.
    Full HTTP-level tests belong in an async httpx test suite against Docker.
    """

    def test_allowed_extensions(self):
        from api.app.routes.tryon import ALLOWED_EXTENSIONS
        assert ".jpg" in ALLOWED_EXTENSIONS
        assert ".jpeg" in ALLOWED_EXTENSIONS
        assert ".png" in ALLOWED_EXTENSIONS
        assert ".pdf" not in ALLOWED_EXTENSIONS
        assert ".gif" not in ALLOWED_EXTENSIONS

    def test_max_bytes_respects_env(self):
        from api.app.routes.tryon import MAX_BYTES
        from api.app.config import get_settings
        s = get_settings()
        assert MAX_BYTES == s.MAX_UPLOAD_SIZE_MB * 1024 * 1024


# ── TC-009 / TC-010  SSIM quality gate ────────────────────────────────────────

class TestSSIMQualityGate:
    def test_identical_images_score_high(self):
        from skimage.metrics import structural_similarity as ssim
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        score, _ = ssim(img, img, full=True, channel_axis=2, data_range=255)
        assert score >= 0.65, "Identical images must score above threshold"

    def test_noise_images_score_low(self):
        from skimage.metrics import structural_similarity as ssim
        rng = np.random.default_rng(42)
        a = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        b = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        score, _ = ssim(a, b, full=True, channel_axis=2, data_range=255)
        assert score < 0.65, "Random noise images must score below threshold"

    def test_training_pair_saved_when_score_high(self, tmp_path):
        """TC-009: high-quality result → saved_as_training = True"""
        from api.app.workers.tasks import _compute_ssim, _save_training_pair
        from api.app.config import get_settings

        person  = make_test_image(str(tmp_path / "person.jpg"))
        garment = make_test_image(str(tmp_path / "garment.jpg"))
        result  = make_test_image(str(tmp_path / "result.jpg"))  # identical → high SSIM

        score = _compute_ssim(person, result)
        assert score >= 0.65

        pairs_root = tmp_path / "storage"
        pairs_root.mkdir()
        with patch("api.app.workers.tasks.settings") as mock_cfg, \
             patch("api.app.workers.tasks.get_storage") as mock_storage:

            mock_cfg.LOCAL_STORAGE_PATH = str(pairs_root)
            mock_cfg.MIN_QUALITY_SCORE  = 0.65

            from api.app.services.storage import LocalStorageBackend
            mock_storage.return_value = LocalStorageBackend(str(pairs_root))

            _save_training_pair("job_test", person, garment, result, score)

        pair_dir = pairs_root / "training_pairs" / "job_test"
        assert pair_dir.exists()
        meta = json.loads((pair_dir / "meta.json").read_text())
        assert meta["quality_score"] >= 0.65

        pairs_json = pairs_root / "training_pairs" / "pairs.json"
        assert pairs_json.exists()
        entries = json.loads(pairs_json.read_text())
        assert any(e["job_id"] == "job_test" for e in entries)


# ── TC-019  Dataset discovery ─────────────────────────────────────────────────

class TestVTONDataset:
    def test_discovers_pairs_from_training_dir(self, tmp_path):
        pair_dir = tmp_path / "training_pairs" / "job_001"
        pair_dir.mkdir(parents=True)
        make_test_image(str(pair_dir / "person.jpg"))
        make_test_image(str(pair_dir / "garment.jpg"))
        make_test_image(str(pair_dir / "result.jpg"))
        (pair_dir / "meta.json").write_text(
            json.dumps({"job_id": "job_001", "ssim_score": 0.8, "quality_score": 0.8})
        )

        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
        from src.data.dataset import VTONDataset

        ds = VTONDataset(str(tmp_path / "training_pairs"), image_size=64)
        assert len(ds) == 1
        sample = ds[0]
        assert "person" in sample
        assert "garment" in sample
        assert "result" in sample
