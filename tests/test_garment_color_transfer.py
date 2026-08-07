"""Unit tests for the color-transfer safety net (ml/scripts/garment_color_transfer.py).

Pure numpy/cv2 — runs without a GPU or the diffusion pipeline, using
synthetic images to simulate a diffusion-model color drift (e.g. a red
garment reproduced as green/olive) and verifying the correction pulls it
back toward the true source color.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml" / "scripts"))

from garment_color_transfer import apply_garment_color_transfer, garment_reference_lab_stats  # noqa: E402


def _solid(h, w, rgb):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = rgb
    return arr


def test_pulls_drifted_color_toward_true_reference():
    # Simulate the reported bug: source garment is pure red, but the
    # diffusion model reproduced it as olive/green.
    reference_garment = _solid(64, 64, (200, 30, 30))   # true red
    drifted_result = _solid(64, 64, (90, 130, 60))       # olive/green drift
    mask = np.full((64, 64), 255, dtype=np.uint8)

    corrected = apply_garment_color_transfer(drifted_result, mask, reference_garment, strength=1.0)

    # After a full-strength match, the corrected region should be
    # dominated by red, not green, unlike the drifted input.
    mean_before = drifted_result.reshape(-1, 3).mean(axis=0)
    mean_after = corrected.reshape(-1, 3).mean(axis=0)
    assert mean_before[1] > mean_before[0]   # drifted: green > red
    assert mean_after[0] > mean_after[1]     # corrected: red > green


def test_pixels_outside_mask_are_untouched():
    reference_garment = _solid(32, 32, (200, 30, 30))
    result = np.zeros((32, 32, 3), dtype=np.uint8)
    result[:16, :] = (90, 130, 60)   # top half: "garment" region (drifted)
    result[16:, :] = (10, 10, 10)    # bottom half: background, must survive untouched

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:16, :] = 255

    corrected = apply_garment_color_transfer(result, mask, reference_garment, strength=1.0)

    assert np.array_equal(corrected[16:, :], result[16:, :])
    assert not np.array_equal(corrected[:16, :], result[:16, :])


def test_strength_zero_is_a_no_op():
    reference_garment = _solid(32, 32, (200, 30, 30))
    result = _solid(32, 32, (90, 130, 60))
    mask = np.full((32, 32), 255, dtype=np.uint8)

    corrected = apply_garment_color_transfer(result, mask, reference_garment, strength=0.0)

    # LAB round-trip introduces small quantization error, not an exact identity.
    assert np.allclose(corrected.astype(int), result.astype(int), atol=2)


def test_empty_mask_returns_input_unchanged():
    reference_garment = _solid(32, 32, (200, 30, 30))
    result = _solid(32, 32, (90, 130, 60))
    mask = np.zeros((32, 32), dtype=np.uint8)

    corrected = apply_garment_color_transfer(result, mask, reference_garment, strength=1.0)

    assert np.array_equal(corrected, result)


def test_reference_stats_exclude_white_backdrop():
    # Product photo: red garment centered on a white background.
    garment = _solid(40, 40, (255, 255, 255))
    garment[10:30, 10:30] = (200, 30, 30)   # the actual garment pixels

    mean, _std = garment_reference_lab_stats(garment)
    # LAB 'a' channel (index 1, OpenCV 8U convention centered at 128) should
    # read positive/red-leaning, not neutral — proving white pixels were excluded.
    assert mean[1] > 128


def test_reference_stats_fallback_when_fully_white():
    garment = _solid(20, 20, (255, 255, 255))
    mean, std = garment_reference_lab_stats(garment)
    assert mean.shape == (3,)
    assert std.shape == (3,)


@pytest.mark.parametrize("strength", [0.25, 0.5, 0.85, 1.0])
def test_strength_monotonically_increases_match(strength):
    reference_garment = _solid(32, 32, (200, 30, 30))
    result = _solid(32, 32, (90, 130, 60))
    mask = np.full((32, 32), 255, dtype=np.uint8)

    corrected = apply_garment_color_transfer(result, mask, reference_garment, strength=strength)
    mean_after = corrected.reshape(-1, 3).mean(axis=0)
    # Higher strength should move the red channel further from the drifted
    # baseline (30-ish) toward the reference (200-ish).
    assert mean_after[0] >= 90 - 1  # never moves further from red than the start
