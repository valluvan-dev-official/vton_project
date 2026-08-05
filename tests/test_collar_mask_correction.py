"""
Unit tests for ml/scripts/collar_mask_correction.py — the leftover-collar
mask correction.

Uses small synthetic parser-label arrays + OpenPose keypoints (no PIL, no
torch, no GPU model, no real image files). Requires only numpy + opencv,
already pinned in api/requirements.txt / requirements-gpu.txt.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml" / "scripts"))

from collar_mask_correction import (  # noqa: E402
    correct_collar_mask,
    BACKGROUND_LABEL,
    UPPER_CLOTHES_LABEL,
    DRESS_LABEL,
    LEFT_ARM_LABEL,
    RIGHT_ARM_LABEL,
    NECK_SKIN_LABEL,
)

W, H = 200, 300


def _kp(neck=None, r_shoulder=None, l_shoulder=None):
    pts = [[0, 0] for _ in range(18)]
    if neck is not None:
        pts[1] = list(neck)
    if r_shoulder is not None:
        pts[2] = list(r_shoulder)
    if l_shoulder is not None:
        pts[5] = list(l_shoulder)
    return {"pose_keypoints_2d": pts}


def make_scene(neck_y=100, shoulder_y=130, shoulder_half=35,
               dark_garment=False, side_pose=False,
               garment_band=True, collar_style="round",
               mask_already_covers_band=False):
    """Build a synthetic torso scene.

    - neck at (cx, neck_y); shoulders at (cx -/+ shoulder_half, shoulder_y).
      By construction neck_y < shoulder_y (neck above the shoulder line),
      matching the reported bug: leftover garment lives in the neck-to-
      shoulder band that a naive shoulder-line clamp would exclude.
    - `garment_band`: if True, the parser still shows old T-shirt fabric
      (label 4) in a thin band between the neck and shoulders, and the mask
      does NOT already cover it (the actual bug being fixed) unless
      `mask_already_covers_band` is set.
    - `dark_garment` doesn't change parser labels (SCHP is intensity-
      independent) but documents the scenario the fix targets.
    - `side_pose` shifts the neck slightly off the shoulder midpoint,
      simulating a slight-side pose.
    - `collar_style`: "round" leaves a full band; "collared" leaves an
      uneven, partially-covered band, closer to a shirt collar leftover.
    """
    cx = W // 2
    parse = np.full((H, W), BACKGROUND_LABEL, dtype=np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)

    torso_top = shoulder_y
    torso_bot = int(0.85 * H)
    torso_left = cx - shoulder_half
    torso_right = cx + shoulder_half
    parse[torso_top:torso_bot, torso_left:torso_right] = UPPER_CLOTHES_LABEL
    mask[torso_top:torso_bot, torso_left:torso_right] = 255

    neck_x = cx + (12 if side_pose else 0)
    neck = (neck_x, neck_y)
    r_shoulder = (cx - shoulder_half, shoulder_y)
    l_shoulder = (cx + shoulder_half, shoulder_y)

    # Exposed neck skin directly at/around the neck point.
    skin_r = max(3, shoulder_half // 6)
    cv2.circle(parse, (int(neck_x), int(neck_y)), skin_r, NECK_SKIN_LABEL, -1)

    if garment_band:
        # Leftover old-garment fabric in the neck-to-shoulder band.
        band_top = neck_y + 2
        band_bot = shoulder_y - 2
        if collar_style == "round":
            parse[band_top:band_bot, torso_left + 4: torso_right - 4] = UPPER_CLOTHES_LABEL
        else:  # "collared" -> uneven strip, e.g. only near the sides (collar points)
            mid = (torso_left + torso_right) // 2
            parse[band_top:band_bot, torso_left + 4: mid - 5] = UPPER_CLOTHES_LABEL
            parse[band_top:band_bot, mid + 5: torso_right - 4] = UPPER_CLOTHES_LABEL
        # Re-stamp the skin circle on top so it stays skin (parser wouldn't
        # double-label a pixel; this keeps the synthetic scene consistent).
        cv2.circle(parse, (int(neck_x), int(neck_y)), skin_r, NECK_SKIN_LABEL, -1)

        if mask_already_covers_band:
            mask[band_top:band_bot, torso_left:torso_right] = 255
        # else: mask stays 0 there -> this is the leftover-collar bug.

    kp = _kp(neck=neck, r_shoulder=r_shoulder, l_shoulder=l_shoulder)
    return parse, mask, kp, {
        "neck": neck, "r_shoulder": r_shoulder, "l_shoulder": l_shoulder,
        "band_top": neck_y + 2, "band_bot": shoulder_y - 2,
        "torso_left": torso_left, "torso_right": torso_right,
    }


# ── Core "applied" cases ─────────────────────────────────────────────────────

def test_round_neck_tshirt_collar_band_added():
    parse, mask, kp, geo = make_scene(collar_style="round")
    corrected, cand, prot, diag = correct_collar_mask(parse, mask, kp)
    assert diag["skipped"] is False
    assert diag["reason"] == "applied"
    assert diag["added_px"] > 0
    # The band that was previously black (0) in the mask is now white (255).
    band = mask[geo["band_top"]:geo["band_bot"], geo["torso_left"] + 4: geo["torso_right"] - 4]
    band_corrected = corrected[geo["band_top"]:geo["band_bot"], geo["torso_left"] + 4: geo["torso_right"] - 4]
    assert (band == 0).any()
    assert (band_corrected == 255).sum() > (band_corrected == 0).sum()


def test_collared_half_sleeve_shirt_uneven_band_added():
    parse, mask, kp, geo = make_scene(collar_style="collared")
    corrected, cand, prot, diag = correct_collar_mask(parse, mask, kp)
    assert diag["skipped"] is False
    assert diag["added_px"] > 0


def test_full_sleeve_shirt_scenario_collar_band_added():
    # Sleeve type doesn't affect this module at all (it runs before the
    # sleeve-specific carve-out in gpu_inference.py); verify it behaves
    # identically regardless — use a wider torso to emulate a looser shirt.
    parse, mask, kp, geo = make_scene(shoulder_half=45, collar_style="round")
    corrected, cand, prot, diag = correct_collar_mask(parse, mask, kp)
    assert diag["skipped"] is False
    assert diag["added_px"] > 0


def test_straight_front_pose():
    parse, mask, kp, geo = make_scene(side_pose=False, collar_style="round")
    corrected, cand, prot, diag = correct_collar_mask(parse, mask, kp)
    assert diag["skipped"] is False


def test_slight_side_pose():
    parse, mask, kp, geo = make_scene(side_pose=True, collar_style="round")
    corrected, cand, prot, diag = correct_collar_mask(parse, mask, kp)
    assert diag["skipped"] is False
    assert diag["added_px"] > 0


def test_dark_original_tshirt_scenario():
    # SCHP parsing is label-based, not intensity-based, so a "dark" garment
    # produces the same label=4 leftover band; this documents/locks that the
    # fix is intensity-independent (it would also fire for a dark shirt).
    parse, mask, kp, geo = make_scene(dark_garment=True, collar_style="round")
    corrected, cand, prot, diag = correct_collar_mask(parse, mask, kp)
    assert diag["skipped"] is False
    assert diag["added_px"] > 0


# ── Required regression test: neck-to-shoulder band inclusion ──────────────

def test_regression_neck_to_shoulder_band_added_without_touching_protected_pixels():
    """The exact bug reported: original-garment pixels between neck.y and
    shoulder_y (i.e. ABOVE the shoulder line, since neck.y < shoulder_y in
    image coordinates) must be added to the editable mask. Neck skin,
    background, and arm pixels elsewhere must be provably unchanged.
    """
    parse, mask, kp, geo = make_scene(neck_y=100, shoulder_y=130,
                                       shoulder_half=35, collar_style="round")

    # Sanity: the band genuinely sits strictly between neck.y and shoulder.y.
    assert geo["neck"][1] < geo["band_top"] < geo["band_bot"] < geo["r_shoulder"][1]
    # Sanity: that band is old-garment-labelled but NOT yet editable.
    band_slice = (slice(geo["band_top"], geo["band_bot"]),
                  slice(geo["torso_left"] + 4, geo["torso_right"] - 4))
    assert (parse[band_slice] == UPPER_CLOTHES_LABEL).any()
    assert (mask[band_slice] == 0).all()

    # Add some arm and background pixels adjacent to the band to prove they
    # survive untouched.
    arm_slice = (slice(geo["band_top"], geo["band_bot"]), slice(0, 5))
    parse[arm_slice] = RIGHT_ARM_LABEL
    mask[arm_slice] = 255  # arm already correctly editable
    bg_before = mask[0:5, 0:5].copy()
    assert (parse[0:5, 0:5] == BACKGROUND_LABEL).all()

    neck_skin_mask_before = mask.copy()
    neck_x, neck_y = int(geo["neck"][0]), int(geo["neck"][1])
    assert parse[neck_y, neck_x] == NECK_SKIN_LABEL

    corrected, candidate_debug, protection_debug, diag = correct_collar_mask(parse, mask, kp)

    assert diag["skipped"] is False
    assert diag["added_px"] > 0

    # 1) The neck-to-shoulder band's garment pixels are now editable.
    assert (corrected[band_slice] == 255).sum() > 0

    # 2) Neck skin pixel itself must never be flipped to editable by this
    #    correction (it was 0 before and must remain 0).
    assert neck_skin_mask_before[neck_y, neck_x] == 0
    assert corrected[neck_y, neck_x] == 0

    # 3) Background pixels untouched.
    assert np.array_equal(corrected[0:5, 0:5], bg_before)

    # 4) Arm pixels untouched (were already 255, must remain exactly 255 —
    #    not re-derived/altered by this correction).
    assert (corrected[arm_slice] == 255).all()

    # 5) Every newly-added pixel is parser-confirmed old garment fabric.
    added = (corrected == 255) & (mask == 0)
    assert added.any()
    assert np.isin(parse[added], (UPPER_CLOTHES_LABEL, DRESS_LABEL)).all()

    # 6) No newly-added pixel is background/arm/neck-skin.
    assert not np.isin(parse[added], (BACKGROUND_LABEL, LEFT_ARM_LABEL,
                                       RIGHT_ARM_LABEL, NECK_SKIN_LABEL)).any()


# ── Skip conditions ──────────────────────────────────────────────────────────

def test_missing_neck_keypoint_skips():
    parse, mask, kp, geo = make_scene(collar_style="round")
    kp["pose_keypoints_2d"][1] = [0, 0]
    corrected, cand, prot, diag = correct_collar_mask(parse, mask, kp)
    assert diag["skipped"] is True
    assert diag["reason"] == "missing_or_invalid_joints"
    assert np.array_equal(corrected, (mask > 0).astype(np.uint8) * 255)


def test_missing_shoulder_keypoint_skips():
    parse, mask, kp, geo = make_scene(collar_style="round")
    kp["pose_keypoints_2d"][2] = [0, 0]  # right shoulder missing
    corrected, cand, prot, diag = correct_collar_mask(parse, mask, kp)
    assert diag["skipped"] is True
    assert diag["reason"] == "missing_or_invalid_joints"


def test_low_confidence_keypoints_skip():
    parse, mask, kp, geo = make_scene(collar_style="round")
    nx, ny = kp["pose_keypoints_2d"][1]
    kp["pose_keypoints_2d"][1] = [nx, ny, 0.01]  # below confidence floor
    corrected, cand, prot, diag = correct_collar_mask(parse, mask, kp)
    assert diag["skipped"] is True
    assert diag["reason"] == "missing_or_invalid_joints"


def test_no_garment_pixels_in_region_skips():
    parse, mask, kp, geo = make_scene(garment_band=False)
    corrected, cand, prot, diag = correct_collar_mask(parse, mask, kp)
    assert diag["skipped"] is True
    assert diag["reason"] == "no_original_garment_pixels"
    assert diag["added_px"] == 0


def test_already_covered_band_skips_as_no_garment_pixels():
    parse, mask, kp, geo = make_scene(collar_style="round", mask_already_covers_band=True)
    corrected, cand, prot, diag = correct_collar_mask(parse, mask, kp)
    assert diag["skipped"] is True
    assert diag["added_px"] == 0


# ── Protection guarantees ────────────────────────────────────────────────────

def test_face_chin_hair_region_never_touched_geometrically():
    """No confirmed face/hair/chin labels exist, so this is enforced purely
    geometrically: nothing above/around the small neck margin can ever be
    added, because the candidate ellipse's vertical extent stops a bounded
    distance above neck.y regardless of what (unlabeled) content sits there."""
    parse, mask, kp, geo = make_scene(collar_style="round")
    # Simulate a "face-like" garment-labelled region far above the neck
    # (which should never happen from a real parser, but proves the
    # geometric ceiling holds even if it did).
    far_above = slice(0, max(1, geo["neck"][1] - 40)), slice(0, W)
    parse[far_above] = UPPER_CLOTHES_LABEL
    mask[far_above] = 0

    corrected, candidate_debug, protection_debug, diag = correct_collar_mask(parse, mask, kp)
    assert (corrected[far_above] == 0).all()
    assert (candidate_debug[far_above] == 0).all()


def test_upper_chest_boundary_not_exceeded():
    parse, mask, kp, geo = make_scene(collar_style="round")
    # Put garment-labelled pixels far below the intended chest boundary
    # (simulating waist-level fabric) and confirm they're never added.
    waist_y = int(0.9 * H)
    parse[waist_y:waist_y + 5, geo["torso_left"]:geo["torso_right"]] = UPPER_CLOTHES_LABEL
    mask[waist_y:waist_y + 5, geo["torso_left"]:geo["torso_right"]] = 0

    corrected, candidate_debug, protection_debug, diag = correct_collar_mask(parse, mask, kp)
    assert (corrected[waist_y:waist_y + 5, geo["torso_left"]:geo["torso_right"]] == 0).all()
    assert (candidate_debug[waist_y:waist_y + 5, :] == 0).all()


def test_protection_debug_is_255_for_protected_0_elsewhere():
    parse, mask, kp, geo = make_scene(collar_style="round")
    _, _, protection_debug, _ = correct_collar_mask(parse, mask, kp)
    assert set(np.unique(protection_debug)).issubset({0, 255})
    # Background must be marked protected.
    assert protection_debug[0, 0] == 255
    # Neck skin pixel must be marked protected.
    neck_x, neck_y = int(geo["neck"][0]), int(geo["neck"][1])
    assert protection_debug[neck_y, neck_x] == 255


def test_candidate_debug_represents_raw_labels_before_protection():
    parse, mask, kp, geo = make_scene(collar_style="round")
    _, candidate_debug, protection_debug, _ = correct_collar_mask(parse, mask, kp)
    # candidate_debug is old-garment-labelled pixels within the geometric
    # region; it is allowed to overlap protection_debug's "outside bounds"
    # component being empty there, but must never include background pixels.
    assert candidate_debug[0, 0] == 0  # background never candidate


# ── Output contract ──────────────────────────────────────────────────────────

def test_output_strictly_binary():
    parse, mask, kp, geo = make_scene(collar_style="round")
    corrected, candidate_debug, protection_debug, diag = correct_collar_mask(parse, mask, kp)
    assert set(np.unique(corrected)).issubset({0, 255})
    assert set(np.unique(candidate_debug)).issubset({0, 255})
    assert set(np.unique(protection_debug)).issubset({0, 255})


def test_skip_returns_exact_input_mask_content():
    parse, mask, kp, geo = make_scene(garment_band=False)
    corrected, _, _, diag = correct_collar_mask(parse, mask, kp)
    assert diag["skipped"] is True
    # Content-identical to the (already-binary) input mask.
    assert np.array_equal(corrected, mask)


# ── Debug visualization on/off (module itself always returns debug arrays;
#    gpu_inference.py gates whether they get saved to disk via DEBUG_VISUALIZATION) ──

def test_debug_arrays_always_returned_regardless_of_visualization_flag():
    """The correction module has no notion of DEBUG_VISUALIZATION itself —
    gpu_inference.py decides whether to persist the returned debug arrays to
    disk. Confirm the function contract always returns them so that gating
    lives entirely in the caller (matches mask_gap_correction's pattern)."""
    parse, mask, kp, geo = make_scene(collar_style="round")
    result = correct_collar_mask(parse, mask, kp)
    assert len(result) == 4
    corrected, candidate_debug, protection_debug, diag = result
    assert corrected.shape == mask.shape
    assert candidate_debug.shape == mask.shape
    assert protection_debug.shape == mask.shape


# ── Independence from arm-gap correction ────────────────────────────────────

def test_does_not_mutate_input_mask_array_in_place():
    """gpu_inference.py passes the arm-gap-corrected mask in; this module
    must not mutate that array in place, so the arm-gap output stays
    verifiably unchanged for anyone still holding a reference to it."""
    parse, mask, kp, geo = make_scene(collar_style="round")
    original = mask.copy()
    correct_collar_mask(parse, mask, kp)
    assert np.array_equal(mask, original)
