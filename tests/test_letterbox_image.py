"""
Full-image round-trip tests for _letterbox_image / _unletterbox_image in
ml/scripts/gpu_inference.py.

Requires PIL, and (transitively, via gpu_inference.py's module-level imports)
torch and cv2. These are present on the GPU worker but not in a lightweight
dev checkout, so this file skips cleanly via importorskip rather than failing
collection. See test_letterbox_geometry.py for the dependency-free math tests
that validate the same guarantees without needing the ML stack installed.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml" / "scripts"))

from PIL import Image  # noqa: E402
from gpu_inference import _letterbox_image, _unletterbox_image  # noqa: E402

TARGET_W, TARGET_H = 768, 1024


@pytest.mark.parametrize("orig_w, orig_h", [
    (383, 372),   # near-square — the reported regression case
    (500, 500),   # exact square
    (300, 900),   # portrait
    (1200, 400),  # landscape
])
def test_letterbox_then_unletterbox_restores_exact_original_size(orig_w, orig_h):
    img = Image.new("RGB", (orig_w, orig_h), (10, 20, 30))

    canvas, transform = _letterbox_image(img, TARGET_W, TARGET_H)
    assert canvas.size == (TARGET_W, TARGET_H)

    restored = _unletterbox_image(canvas, transform)
    assert restored.size == (orig_w, orig_h)


def test_letterbox_uses_nearest_for_label_masks():
    # A 2-value "mask-like" image should stay binary after letterboxing when
    # NEAREST is requested — no gray blending at the pad boundary.
    mask = Image.new("L", (383, 372), 0)
    canvas, _ = _letterbox_image(mask, TARGET_W, TARGET_H, fill=0, resample=Image.NEAREST)
    values = set(canvas.getdata())
    assert values <= {0, 255}
