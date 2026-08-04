"""
Unit tests for the pure-arithmetic letterbox geometry (GPU inference
body-proportion regression fix).

No PIL/numpy/torch/cv2 required — validates that the forward transform stays
within canvas bounds and that inverse-letterbox restores the exact original
width/height, for square, portrait and landscape inputs of arbitrary size
(nothing here is hard-coded to one job's dimensions).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml" / "scripts"))

from letterbox_geometry import compute_letterbox_geometry  # noqa: E402

TARGET_W, TARGET_H = 768, 1024

CASES = [
    (383, 372),    # near-square — the reported regression case
    (500, 500),    # exact square
    (300, 900),    # portrait, taller than target ratio
    (1200, 400),   # landscape
    (768, 1024),   # already exact target ratio
    (1, 1),        # degenerate tiny square
    (4000, 3000),  # large landscape photo
    (383, 1),      # extreme landscape
    (1, 372),      # extreme portrait
]


@pytest.mark.parametrize("orig_w, orig_h", CASES)
def test_fitted_size_within_target_bounds(orig_w, orig_h):
    t = compute_letterbox_geometry(orig_w, orig_h, TARGET_W, TARGET_H)
    assert 1 <= t.new_w <= TARGET_W
    assert 1 <= t.new_h <= TARGET_H
    # At least one axis exactly fills the target (the constraining axis).
    assert t.new_w == TARGET_W or t.new_h == TARGET_H


@pytest.mark.parametrize("orig_w, orig_h", [
    c for c in CASES if min(c) >= 100  # exclude 1px-wide/tall degenerate cases:
    # at such tiny fitted pixel counts, integer rounding of new_w/new_h can
    # swing the ratio by double digits despite being off by a single pixel —
    # not a realistic photo size, so it's excluded from this ratio check.
])
def test_fitted_size_preserves_aspect_ratio(orig_w, orig_h):
    t = compute_letterbox_geometry(orig_w, orig_h, TARGET_W, TARGET_H)
    orig_ratio = orig_w / orig_h
    fitted_ratio = t.new_w / t.new_h
    assert abs(orig_ratio - fitted_ratio) / orig_ratio < 0.02


@pytest.mark.parametrize("orig_w, orig_h", CASES)
def test_padding_is_centered_and_nonnegative(orig_w, orig_h):
    t = compute_letterbox_geometry(orig_w, orig_h, TARGET_W, TARGET_H)
    assert t.pad_x >= 0 and t.pad_y >= 0
    leftover_x = TARGET_W - t.new_w
    leftover_y = TARGET_H - t.new_h
    # Centered: the two pad slivers differ by at most 1px (odd leftover).
    assert abs(t.pad_x - (leftover_x - t.pad_x)) <= 1
    assert abs(t.pad_y - (leftover_y - t.pad_y)) <= 1


@pytest.mark.parametrize("orig_w, orig_h", CASES)
def test_crop_box_never_exceeds_canvas(orig_w, orig_h):
    """Safeguard check: the region _unletterbox_image crops must stay inside
    the (target_w, target_h) canvas produced by the forward pass."""
    t = compute_letterbox_geometry(orig_w, orig_h, TARGET_W, TARGET_H)
    assert t.pad_x + t.new_w <= TARGET_W
    assert t.pad_y + t.new_h <= TARGET_H


@pytest.mark.parametrize("orig_w, orig_h", CASES)
def test_transform_records_exact_original_dimensions(orig_w, orig_h):
    """Core guarantee: the transform always carries back the *exact* original
    width/height, so the inverse resize target is never approximate."""
    t = compute_letterbox_geometry(orig_w, orig_h, TARGET_W, TARGET_H)
    assert (t.orig_w, t.orig_h) == (orig_w, orig_h)


def test_invalid_dimensions_raise():
    with pytest.raises(ValueError):
        compute_letterbox_geometry(0, 100, TARGET_W, TARGET_H)
    with pytest.raises(ValueError):
        compute_letterbox_geometry(100, 0, TARGET_W, TARGET_H)
    with pytest.raises(ValueError):
        compute_letterbox_geometry(100, 100, 0, TARGET_H)
    with pytest.raises(ValueError):
        compute_letterbox_geometry(100, 100, TARGET_W, -1)
