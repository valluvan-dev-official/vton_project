"""Unit tests for BodyAnalyzer + confidence.py — covers Phase-1 spec item 3
(height optional, never required) and item 7 (confidence tiers). Pure
Python: PoseLandmarks is constructed directly, no OpenPose/SCHP/GPU
involved, matching the dependency-injection design in pose_adapter.py.
"""
import pytest

from api.app.services.fit_analysis.body_analyzer import BodyAnalyzer
from api.app.services.fit_analysis.confidence import (
    CONFIDENCE_NO_POSE_NO_HEIGHT,
    CONFIDENCE_NO_POSE_WITH_HEIGHT,
    CONFIDENCE_POSE_NO_HEIGHT,
    CONFIDENCE_POSE_WITH_HEIGHT,
    CONFIDENCE_POSE_WITH_HEIGHT_HIGH_QUALITY,
    POSE_QUALITY_HIGH_THRESHOLD,
    estimate_body_confidence,
)
from api.app.services.fit_analysis.pose_adapter import PoseLandmarks


def _landmarks(quality=0.9, **overrides):
    defaults = dict(
        shoulder_width_px=140.0, torso_length_px=180.0,
        hip_width_px=95.0, arm_length_px=190.0,
        image_width=384, image_height=512, quality=quality,
    )
    defaults.update(overrides)
    return PoseLandmarks(**defaults)


class TestConfidenceTiers:
    def test_no_pose_no_height(self):
        assert estimate_body_confidence(False, False) == CONFIDENCE_NO_POSE_NO_HEIGHT

    def test_no_pose_with_height(self):
        assert estimate_body_confidence(False, True) == CONFIDENCE_NO_POSE_WITH_HEIGHT

    def test_pose_no_height(self):
        assert estimate_body_confidence(True, False) == CONFIDENCE_POSE_NO_HEIGHT

    def test_pose_with_height_low_quality(self):
        conf = estimate_body_confidence(True, True, pose_quality=0.4)
        assert conf == CONFIDENCE_POSE_WITH_HEIGHT

    def test_pose_with_height_high_quality(self):
        conf = estimate_body_confidence(True, True, pose_quality=POSE_QUALITY_HIGH_THRESHOLD)
        assert conf == CONFIDENCE_POSE_WITH_HEIGHT_HIGH_QUALITY

    def test_ordering_is_monotonic_increasing(self):
        """More/better input should never reduce confidence."""
        tiers = [
            estimate_body_confidence(False, False),
            estimate_body_confidence(False, True),
            estimate_body_confidence(True, False),
            estimate_body_confidence(True, True, pose_quality=0.5),
            estimate_body_confidence(True, True, pose_quality=0.95),
        ]
        assert tiers == sorted(tiers)

    def test_never_claims_full_certainty(self):
        from api.app.services.fit_analysis.confidence import CONFIDENCE_CEILING
        assert CONFIDENCE_CEILING < 1.0
        assert estimate_body_confidence(True, True, pose_quality=1.0) <= CONFIDENCE_CEILING


class TestBodyAnalyzerMissingHeight:
    def test_missing_height_with_landmarks_yields_relative_but_no_absolute_cm(self):
        """Height absent -> proportions can't be scaled to real cm (item 3:
        'estimate body proportions only and reduce confidence')."""
        body = BodyAnalyzer().analyze(height_cm=None, landmarks=_landmarks())
        assert body.height_cm is None
        assert body.chest_cm is None
        assert body.shoulder_width_cm is None
        assert "no_height_reference" in body.notes
        assert body.confidence == CONFIDENCE_POSE_NO_HEIGHT

    def test_missing_height_and_missing_landmarks(self):
        body = BodyAnalyzer().analyze(height_cm=None, landmarks=None)
        assert body.confidence == CONFIDENCE_NO_POSE_NO_HEIGHT
        assert body.chest_cm is None


class TestBodyAnalyzerHeightSupplied:
    def test_height_supplied_scales_measurements(self):
        # quality below POSE_QUALITY_HIGH_THRESHOLD so this isolates the
        # plain "pose + height" tier from the "high quality" tier below.
        body = BodyAnalyzer().analyze(height_cm=175.0, landmarks=_landmarks(quality=0.5))
        assert body.height_cm == 175.0
        assert body.shoulder_width_cm is not None
        assert body.chest_cm is not None
        # Sanity bounds — a real adult torso, not an absurd figure.
        assert 30 < body.shoulder_width_cm < 60
        assert 60 < body.chest_cm < 150
        assert body.confidence == CONFIDENCE_POSE_WITH_HEIGHT

    def test_height_supplied_high_quality_pose_raises_confidence(self):
        body = BodyAnalyzer().analyze(height_cm=175.0, landmarks=_landmarks(quality=0.95))
        assert body.confidence == CONFIDENCE_POSE_WITH_HEIGHT_HIGH_QUALITY

    def test_measurement_source_is_always_estimated(self):
        """Phase-1 spec item 2: never represent image-derived values as exact."""
        body = BodyAnalyzer().analyze(height_cm=175.0, landmarks=_landmarks())
        assert body.measurement_source == "estimated"

    def test_landmarks_without_torso_length_cannot_scale(self):
        lm = _landmarks(torso_length_px=None)
        body = BodyAnalyzer().analyze(height_cm=175.0, landmarks=lm)
        assert body.chest_cm is None
        assert "landmarks_present_but_missing_torso_length" in body.notes


class TestBodyAnalyzerInvalidInputs:
    def test_negative_height_treated_as_missing(self):
        body = BodyAnalyzer().analyze(height_cm=-10, landmarks=_landmarks())
        assert body.chest_cm is None
        assert body.confidence == CONFIDENCE_POSE_NO_HEIGHT

    def test_zero_height_treated_as_missing(self):
        body = BodyAnalyzer().analyze(height_cm=0, landmarks=_landmarks())
        assert body.confidence == CONFIDENCE_POSE_NO_HEIGHT
