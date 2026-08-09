"""Turns a photo (+ optional pose landmarks, + optional height) into a
BodyMeasurementEstimate. Pure math — no I/O, no model calls. Landmark
extraction (which does need a model call) lives in pose_adapter.py and is
injected in, so this class is trivially unit-testable with plain fake
PoseLandmarks values.

Anthropometric constants below are the same well-known, publicly-documented
ratios ml/scripts/gpu_inference.py's own _estimate_person_size() uses
(torso-to-height ~3.05x) — reimplemented independently here (not imported)
so this package has zero import-time coupling to ml/scripts, per Phase-1
spec item 11. Circumference (chest/waist/hip) approximations are
deliberately labeled as rough: a single frontal 2D photo has no depth
information, so any circumference figure here is a coarse estimate, never
a measurement to be taken at face value — confidence is capped accordingly
regardless of how good the pose landmarks are (see confidence.py).
"""
from typing import Optional

from .confidence import estimate_body_confidence, estimate_uncertainty_pct
from .models import MEASUREMENT_SOURCE_ESTIMATED, BodyMeasurementEstimate
from .pose_adapter import PoseLandmarks

# Standard torso-length : standing-height ratio used to back out an overall
# body-height-in-pixels figure from torso length alone (neck to mid-hip).
TORSO_TO_HEIGHT_RATIO = 3.05

# Rough width -> circumference multipliers for a frontal silhouette. These
# assume a roughly elliptical cross-section (front width ~= 0.75 of the true
# depth-inclusive width) and are only ever applied when a real height
# reference is available to establish a pixel-to-cm scale in the first
# place. Treat as a coarse approximation, not an anatomical model.
# Chest circumference from shoulder (biacromial) width.
CHEST_WIDTH_TO_CIRCUMFERENCE = 2.2
# Hip circumference from hip-JOINT width (much narrower than the true hip
# circumference, unlike shoulder width which already approximates chest span).
HIP_WIDTH_TO_CIRCUMFERENCE = 3.3
# Waist has no direct landmark in the COCO-18 layout; approximate as a
# fraction of the chest/hip circumference average (waist is typically
# narrower than both for a standard build).
WAIST_TO_CHEST_HIP_AVERAGE_RATIO = 0.85

# Body-shape classification ratio boundaries — same spirit as
# app/services/fit_calculator.py's BODY_SHAPE_THRESHOLDS, reimplemented
# independently since that module operates on a different (user-entered)
# dataclass and belongs to an unrelated, pre-existing manual-entry feature.
BODY_SHAPE_BUST_HIP_LOW = 0.95
BODY_SHAPE_BUST_HIP_HIGH = 1.05
BODY_SHAPE_WAIST_HIP_HOURGLASS_MAX = 0.75


def _classify_body_shape(chest_cm: Optional[float], waist_cm: Optional[float],
                          hip_cm: Optional[float]) -> Optional[str]:
    """Descriptive only — see models.py docstring: never fed into sizing math."""
    if not (chest_cm and waist_cm and hip_cm):
        return None
    bhr = chest_cm / hip_cm
    whr = waist_cm / hip_cm
    balanced = BODY_SHAPE_BUST_HIP_LOW <= bhr <= BODY_SHAPE_BUST_HIP_HIGH
    if balanced and whr <= BODY_SHAPE_WAIST_HIP_HOURGLASS_MAX:
        return "hourglass"
    if bhr <= BODY_SHAPE_BUST_HIP_LOW:
        return "pear"
    if bhr >= BODY_SHAPE_BUST_HIP_HIGH:
        return "inverted_triangle"
    if whr >= 0.90:
        return "apple"
    return "rectangle"


class BodyAnalyzer:
    """analyze() is the single entry point. Never raises for "missing data"
    situations — a photo with no usable landmarks and no height still
    produces a valid (very low confidence) BodyMeasurementEstimate rather
    than an exception, since shadow-mode analysis must never be allowed to
    fail a try-on job."""

    def analyze(
        self,
        height_cm: Optional[float] = None,
        landmarks: Optional[PoseLandmarks] = None,
    ) -> BodyMeasurementEstimate:
        notes: list[str] = []
        has_height = height_cm is not None and height_cm > 0
        has_landmarks = landmarks is not None and (
            landmarks.shoulder_width_px or landmarks.torso_length_px
        )

        if not has_landmarks:
            notes.append("no_pose_landmarks")
        if not has_height:
            notes.append("no_height_reference")

        confidence = estimate_body_confidence(
            has_pose_landmarks=has_landmarks,
            has_height=has_height,
            pose_quality=landmarks.quality if landmarks else None,
        )
        # Every *_cm value below is a rough, image-derived estimate — never
        # an exact measurement. uncertainty_pct drives the (low, high) range
        # exposed via BodyMeasurementEstimate.range_cm() / *_cm_range
        # properties, so callers never have to treat a bare point value as
        # precise.
        uncertainty_pct = estimate_uncertainty_pct(confidence)

        if not has_landmarks:
            # No usable pixel landmarks at all — nothing to scale, even if
            # height was given. Return height as a passthrough reference
            # only; every derived field stays None rather than guessed.
            return BodyMeasurementEstimate(
                height_cm=height_cm,
                confidence=confidence,
                measurement_source=MEASUREMENT_SOURCE_ESTIMATED,
                uncertainty_pct=uncertainty_pct,
                notes=notes,
            )

        shoulder_width_cm = chest_cm = waist_cm = hip_cm = torso_length_cm = arm_length_cm = None

        if has_height and landmarks.torso_length_px:
            body_height_px = landmarks.torso_length_px * TORSO_TO_HEIGHT_RATIO
            if body_height_px > 0:
                px_to_cm = height_cm / body_height_px

                # Rounded to the nearest whole centimetre, not a decimal —
                # a coarse frontal-photo estimate does not warrant
                # mm-level-looking precision (see uncertainty_pct above for
                # the actual honest error band).
                if landmarks.shoulder_width_px:
                    shoulder_width_cm = float(round(landmarks.shoulder_width_px * px_to_cm))
                    chest_cm = float(round(shoulder_width_cm * CHEST_WIDTH_TO_CIRCUMFERENCE))
                if landmarks.torso_length_px:
                    torso_length_cm = float(round(landmarks.torso_length_px * px_to_cm))
                if landmarks.arm_length_px:
                    arm_length_cm = float(round(landmarks.arm_length_px * px_to_cm))
                if landmarks.hip_width_px:
                    hip_width_cm = landmarks.hip_width_px * px_to_cm
                    hip_cm = float(round(hip_width_cm * HIP_WIDTH_TO_CIRCUMFERENCE))
                    if chest_cm:
                        waist_cm = float(round(
                            ((chest_cm + hip_cm) / 2) * WAIST_TO_CHEST_HIP_AVERAGE_RATIO
                        ))
        elif not has_height:
            notes.append("landmarks_present_but_unscaled_without_height")
        else:
            notes.append("landmarks_present_but_missing_torso_length")

        body_shape = _classify_body_shape(chest_cm, waist_cm, hip_cm)

        return BodyMeasurementEstimate(
            height_cm=height_cm,
            shoulder_width_cm=shoulder_width_cm,
            chest_cm=chest_cm,
            waist_cm=waist_cm,
            hip_cm=hip_cm,
            torso_length_cm=torso_length_cm,
            arm_length_cm=arm_length_cm,
            body_shape=body_shape,
            confidence=confidence,
            measurement_source=MEASUREMENT_SOURCE_ESTIMATED,
            uncertainty_pct=uncertainty_pct,
            notes=notes,
        )
