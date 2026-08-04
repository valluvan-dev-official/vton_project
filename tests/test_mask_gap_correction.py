"""
Unit tests for ml/scripts/mask_gap_correction.py — the bent-arm/torso
background-gap correction for the IDM-VTON inpainting mask.

Uses small synthetic parser-label arrays + OpenPose keypoints (no PIL, no
torch, no GPU model, no real image files). Requires only numpy + opencv,
which are already pinned in api/requirements.txt / requirements-gpu.txt.

Scene construction mirrors get_mask_location()'s actual bug: a torso
rectangle (label 4), a shoulder->elbow->wrist arm (label 15/14), and a mask
that (like the live implementation) dilates the arm line so it bridges into
real background near the elbow when the arm is bent. Tests are deterministic
— no randomness.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml" / "scripts"))

from mask_gap_correction import (  # noqa: E402
    correct_agnostic_mask_gap,
    scale_keypoints,
    BACKGROUND_LABEL,
)

UPPER_CLOTHES = 4
RIGHT_ARM = 15
LEFT_ARM = 14


def _empty_keypoints():
    return {"pose_keypoints_2d": [[0, 0] for _ in range(18)]}


def make_scene(W=200, H=300, right_bent=False, left_bent=False,
               right_present=True, left_present=True, bridge_gap=True):
    """Build a synthetic (parse_array, mask_array, keypoints) scene.

    A torso occupies the center column. Each present arm goes
    shoulder -> elbow -> wrist. A *bent* arm's wrist folds back toward the
    torso centerline (creating a real background wedge between forearm and
    torso); a *straight* arm continues outward/down (no wedge). When
    `bridge_gap` is True, the mask (but not the parser) dilates the arm line
    the way the live get_mask_location() does, bridging that wedge's
    background pixels into the mask — this is the bug under test.
    """
    parse = np.full((H, W), BACKGROUND_LABEL, dtype=np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)

    cx = W // 2
    torso_half_w = int(0.13 * W)
    torso_top = int(0.15 * H)
    torso_bot = int(0.78 * H)
    parse[torso_top:torso_bot, cx - torso_half_w: cx + torso_half_w] = UPPER_CLOTHES
    mask[torso_top:torso_bot, cx - torso_half_w: cx + torso_half_w] = 255

    shoulder_y = torso_top + int(0.04 * H)
    r_shoulder = (cx - torso_half_w - 2, shoulder_y)
    l_shoulder = (cx + torso_half_w + 2, shoulder_y)

    kp = _empty_keypoints()
    kp["pose_keypoints_2d"][1] = [cx, shoulder_y]  # neck
    kp["pose_keypoints_2d"][2] = list(r_shoulder)
    kp["pose_keypoints_2d"][5] = list(l_shoulder)

    thickness = max(2, int(0.025 * min(W, H)))

    def draw_arm(shoulder, bent, sign, label, elbow_idx, wrist_idx):
        elbow_y = shoulder[1] + int(0.22 * H)
        if bent:
            elbow = (shoulder[0] + sign * int(0.16 * W), elbow_y)
            # wrist folds back toward the torso centerline -> real wedge gap
            wrist = (cx + sign * int(0.02 * W), elbow_y + int(0.20 * H))
        else:
            elbow = (shoulder[0] + sign * int(0.05 * W), elbow_y)
            wrist = (shoulder[0] + sign * int(0.10 * W), elbow_y + int(0.20 * H))

        cv2.line(parse, shoulder, elbow, label, thickness)
        cv2.line(parse, elbow, wrist, label, thickness)

        arm_line = np.zeros((H, W), dtype=np.uint8)
        cv2.line(arm_line, shoulder, elbow, 255, thickness)
        cv2.line(arm_line, elbow, wrist, 255, thickness)

        if bridge_gap:
            dpx = max(3, int(0.05 * min(W, H)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dpx + 1, 2 * dpx + 1))
            arm_line = cv2.dilate(arm_line, kernel, iterations=4)

        mask[:] = np.maximum(mask, arm_line)
        kp["pose_keypoints_2d"][elbow_idx] = list(elbow)
        kp["pose_keypoints_2d"][wrist_idx] = list(wrist)

    if right_present:
        draw_arm(r_shoulder, right_bent, sign=-1, label=RIGHT_ARM, elbow_idx=3, wrist_idx=4)
    if left_present:
        draw_arm(l_shoulder, left_bent, sign=+1, label=LEFT_ARM, elbow_idx=6, wrist_idx=7)

    return parse, mask, kp


# ── Core gap-removal behaviour ──────────────────────────────────────────────

def test_right_bent_arm_gap_removed():
    parse, mask, kp = make_scene(right_bent=True, left_present=False)
    corrected, diag, gap_debug = correct_agnostic_mask_gap(parse, mask, kp)

    right_diag = next(d for d in diag if d["side"] == "right")
    assert right_diag["skipped"] is False
    assert right_diag["removed_px"] > 0
    assert gap_debug.any()
    # Every removed pixel must be real background per the parser.
    removed = (mask > 0) & (corrected == 0)
    assert removed.any()
    assert np.all(parse[removed] == BACKGROUND_LABEL)


def test_left_bent_arm_gap_removed():
    parse, mask, kp = make_scene(left_bent=True, right_present=False)
    corrected, diag, gap_debug = correct_agnostic_mask_gap(parse, mask, kp)

    left_diag = next(d for d in diag if d["side"] == "left")
    assert left_diag["skipped"] is False
    assert left_diag["removed_px"] > 0
    removed = (mask > 0) & (corrected == 0)
    assert removed.any()
    assert np.all(parse[removed] == BACKGROUND_LABEL)


def test_both_arms_bent_both_gaps_removed():
    parse, mask, kp = make_scene(right_bent=True, left_bent=True)
    corrected, diag, gap_debug = correct_agnostic_mask_gap(parse, mask, kp)

    by_side = {d["side"]: d for d in diag}
    assert by_side["right"]["skipped"] is False
    assert by_side["left"]["skipped"] is False
    assert by_side["right"]["removed_px"] > 0
    assert by_side["left"]["removed_px"] > 0


# ── Skip conditions ──────────────────────────────────────────────────────────

def test_straight_arm_mask_unchanged():
    parse, mask, kp = make_scene(right_bent=False, left_present=False, bridge_gap=False)
    corrected, diag, gap_debug = correct_agnostic_mask_gap(parse, mask, kp)

    right_diag = next(d for d in diag if d["side"] == "right")
    assert right_diag["skipped"] is True
    assert "straight" in right_diag["reason"]
    assert np.array_equal(corrected, (mask > 0).astype(np.uint8) * 255)
    assert not gap_debug.any()


def test_missing_joints_unchanged():
    parse, mask, kp = make_scene(right_present=False, left_present=False)
    corrected, diag, gap_debug = correct_agnostic_mask_gap(parse, mask, kp)

    assert all(d["skipped"] for d in diag)
    assert np.array_equal(corrected, (mask > 0).astype(np.uint8) * 255)


def test_invalid_zero_zero_joints_treated_as_missing():
    parse, mask, kp = make_scene(right_bent=True, left_present=False)
    kp["pose_keypoints_2d"][4] = [0, 0]  # blank out the right wrist
    corrected, diag, gap_debug = correct_agnostic_mask_gap(parse, mask, kp)

    right_diag = next(d for d in diag if d["side"] == "right")
    assert right_diag["skipped"] is True
    assert right_diag["reason"] == "missing_or_invalid_joints"


def test_low_confidence_joint_treated_as_missing():
    parse, mask, kp = make_scene(right_bent=True, left_present=False)
    x, y = kp["pose_keypoints_2d"][4]
    kp["pose_keypoints_2d"][4] = [x, y, 0.01]  # below the 0.05 confidence floor
    corrected, diag, gap_debug = correct_agnostic_mask_gap(parse, mask, kp)

    right_diag = next(d for d in diag if d["side"] == "right")
    assert right_diag["skipped"] is True


def test_no_eligible_component_when_mask_not_bridged():
    # Bent arm, but the mask was never dilated into the gap (bridge_gap=False)
    # -> no background pixels are wrongly marked editable -> nothing to remove.
    parse, mask, kp = make_scene(right_bent=True, left_present=False, bridge_gap=False)
    corrected, diag, gap_debug = correct_agnostic_mask_gap(parse, mask, kp)

    right_diag = next(d for d in diag if d["side"] == "right")
    assert right_diag["skipped"] is True
    assert np.array_equal(corrected, (mask > 0).astype(np.uint8) * 255)


def test_border_connected_background_not_removed():
    """A large background region touching the image border (the open
    background around the person, mistakenly marked editable) must never be
    removed — only enclosed pockets are valid candidates."""
    parse, mask, kp = make_scene(right_bent=True, left_present=False)
    # Mark almost the whole top-left corner (touching the border) as editable
    # background — simulates an unrelated stray white region elsewhere.
    mask[0:20, 0:20] = 255
    assert parse[0, 0] == BACKGROUND_LABEL

    corrected, diag, gap_debug = correct_agnostic_mask_gap(parse, mask, kp)
    # The border-touching corner must survive untouched.
    assert corrected[0, 0] == 255
    assert corrected[10, 10] == 255


# ── Preservation guarantees ─────────────────────────────────────────────────

def test_parsed_arm_pixels_retained():
    parse, mask, kp = make_scene(right_bent=True, left_present=False)
    corrected, _, _ = correct_agnostic_mask_gap(parse, mask, kp)
    arm_px = (parse == RIGHT_ARM) & (mask > 0)
    assert arm_px.any()
    # None of the actual arm-labeled pixels are ever removed by this
    # correction (it only ever touches parser-background pixels).
    assert np.array_equal(corrected[arm_px], mask[arm_px])


def test_parsed_upper_clothes_pixels_retained():
    parse, mask, kp = make_scene(right_bent=True, left_present=False)
    corrected, _, _ = correct_agnostic_mask_gap(parse, mask, kp)
    clothes_px = (parse == UPPER_CLOTHES)
    assert clothes_px.any()
    assert np.array_equal(corrected[clothes_px], mask[clothes_px])


def test_does_not_globally_strip_background_from_mask():
    """The correction must not behave like a blanket 'mask AND clothes-label'
    intersection — legitimate non-gap background left in the mask by the
    caller should be unaffected unless it's the specific bridged component."""
    parse, mask, kp = make_scene(right_bent=True, left_present=False)
    # Add an unrelated, small, enclosed patch of "editable background" far
    # from either arm, fully surrounded by clothes pixels so it isn't
    # border-connected.
    mask[5:9, 5:9] = 255
    parse[5:9, 5:9] = BACKGROUND_LABEL
    corrected, _, _ = correct_agnostic_mask_gap(parse, mask, kp)
    # Nowhere near the arm ROI / not overlapping enough -> should survive.
    assert corrected[6, 6] == 255


# ── Sleeve-type interaction (correction runs before sleeve-specific logic) ──

def test_half_sleeve_downstream_carveout_still_effective():
    """This module doesn't know about sleeve type; it only guarantees it
    doesn't remove the arm pixels that the half-sleeve carve-out (in
    gpu_inference.py, applied afterwards) needs to still be present."""
    parse, mask, kp = make_scene(right_bent=True, left_present=False)
    corrected, _, _ = correct_agnostic_mask_gap(parse, mask, kp)
    # Arm pixels near the wrist remain white and available for the
    # downstream half-sleeve circle carve-out to clear.
    wx, wy = (int(v) for v in kp["pose_keypoints_2d"][4])
    wx, wy = max(0, min(mask.shape[1] - 1, wx)), max(0, min(mask.shape[0] - 1, wy))
    assert corrected[wy, wx] == 255


def test_full_sleeve_arm_coverage_retained():
    """For full sleeves (no carve-out applied downstream), the actual parsed
    arm region must remain part of the mask so the garment can cover it."""
    parse, mask, kp = make_scene(right_bent=True, left_present=False)
    corrected, _, _ = correct_agnostic_mask_gap(parse, mask, kp)
    arm_px = (parse == RIGHT_ARM)
    assert np.array_equal(corrected[arm_px], mask[arm_px])
    assert (corrected[arm_px] == 255).all()


# ── Output contract ──────────────────────────────────────────────────────────

def test_output_is_strictly_binary():
    parse, mask, kp = make_scene(right_bent=True, left_present=False)
    corrected, _, gap_debug = correct_agnostic_mask_gap(parse, mask, kp)
    assert set(np.unique(corrected)).issubset({0, 255})
    assert set(np.unique(gap_debug)).issubset({0, 255})


def test_scale_keypoints_scales_xy_only():
    kp = {"pose_keypoints_2d": [[10, 20], [0, 0], [5, 5, 0.9]]}
    scaled = scale_keypoints(kp, 2.0, 3.0)
    assert scaled["pose_keypoints_2d"][0] == [20, 60]
    assert scaled["pose_keypoints_2d"][1] == [0, 0]
    assert scaled["pose_keypoints_2d"][2] == [10, 15, 0.9]


# ── Resolution independence ─────────────────────────────────────────────────

@pytest.mark.parametrize("W, H", [(200, 300), (400, 600), (100, 150)])
def test_resolution_independent_gap_removal(W, H):
    """The same bent-arm scene, rebuilt at different resolutions, must
    trigger gap removal at every scale — nothing here is a fixed pixel
    threshold tied to one job's dimensions."""
    parse, mask, kp = make_scene(W=W, H=H, right_bent=True, left_present=False)
    corrected, diag, _ = correct_agnostic_mask_gap(parse, mask, kp)
    right_diag = next(d for d in diag if d["side"] == "right")
    assert right_diag["skipped"] is False
    assert right_diag["removed_px"] > 0
