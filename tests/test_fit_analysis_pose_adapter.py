"""Unit tests for pose_adapter.py — the read-only reuse boundary (Phase-1
spec item 4). Uses a fake engine double, never a real OpenPose/SCHP model
or GPU, and never imports/touches ml/scripts/gpu_inference.py.
"""
import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from api.app.services.fit_analysis.pose_adapter import (  # noqa: E402
    EngineReadOnlyPoseAdapter,
    landmarks_from_keypoints,
    try_get_pose_engine,
)

# A full COCO-18 keypoint set with plausible pixel positions on a 384x512 canvas.
_FULL_KEYPOINTS = {
    "pose_keypoints_2d": [
        [192, 60],   # 0 nose
        [192, 90],   # 1 neck
        [160, 100],  # 2 R shoulder
        [150, 160],  # 3 R elbow
        [145, 220],  # 4 R wrist
        [224, 100],  # 5 L shoulder
        [234, 160],  # 6 L elbow
        [239, 220],  # 7 L wrist
        [170, 270],  # 8 R hip
        [0, 0],       # 9 R knee (undetected)
        [0, 0],       # 10 R ankle
        [214, 270],  # 11 L hip
        [0, 0],       # 12 L knee
        [0, 0],       # 13 L ankle
        [0, 0], [0, 0], [0, 0], [0, 0],  # 14-17 eyes/ears
    ]
}


class FakeEngine:
    """Satisfies pose_adapter.PoseParseEngine structurally — no inheritance
    needed, just the two method names, exactly like a duck-typed
    GPUInferenceEngine would."""

    def __init__(self, keypoints=None, raise_on_call=False):
        self._keypoints = keypoints if keypoints is not None else _FULL_KEYPOINTS
        self._raise_on_call = raise_on_call

    def _openpose(self, img):
        if self._raise_on_call:
            raise RuntimeError("simulated model failure")
        return self._keypoints

    def _parser(self, img):
        return (None, None)


class TestLandmarksFromKeypoints:
    def test_full_keypoints_produce_all_landmarks(self):
        lm = landmarks_from_keypoints(_FULL_KEYPOINTS, image_width=768, image_height=1024)
        assert lm.shoulder_width_px == pytest.approx(64.0)  # |224-160|
        assert lm.torso_length_px is not None
        assert lm.hip_width_px == pytest.approx(44.0)  # |214-170|
        assert lm.arm_length_px is not None
        assert lm.image_width == 768 and lm.image_height == 1024

    def test_quality_reflects_missing_landmarks(self):
        lm = landmarks_from_keypoints(_FULL_KEYPOINTS, 384, 512)
        # 9 required landmarks (neck, 2 shoulders, 2 elbows, 2 wrists, 2 hips) — all present here.
        assert lm.quality == 1.0

    def test_empty_keypoints_yields_all_none_and_zero_quality(self):
        lm = landmarks_from_keypoints({"pose_keypoints_2d": []}, 384, 512)
        assert lm.shoulder_width_px is None
        assert lm.torso_length_px is None
        assert lm.quality == 0.0

    def test_missing_keypoints_key_does_not_raise(self):
        lm = landmarks_from_keypoints({}, 384, 512)
        assert lm.shoulder_width_px is None

    def test_zero_coordinate_joints_treated_as_undetected(self):
        """OpenPose reports (0,0) for a joint it couldn't find — must not be
        treated as a real position at the image corner."""
        kp = {"pose_keypoints_2d": [[0, 0]] * 18}
        lm = landmarks_from_keypoints(kp, 384, 512)
        assert lm.shoulder_width_px is None
        assert lm.quality == 0.0


class TestEngineReadOnlyPoseAdapter:
    def test_successful_extraction(self):
        img = Image.new("RGB", (768, 1024), (255, 255, 255))
        adapter = EngineReadOnlyPoseAdapter(FakeEngine())
        lm = adapter.get_landmarks(img)
        assert lm is not None
        assert lm.shoulder_width_px is not None

    def test_engine_failure_returns_none_not_raise(self):
        img = Image.new("RGB", (768, 1024), (255, 255, 255))
        adapter = EngineReadOnlyPoseAdapter(FakeEngine(raise_on_call=True))
        assert adapter.get_landmarks(img) is None

    def test_empty_keypoints_returns_none(self):
        img = Image.new("RGB", (768, 1024), (255, 255, 255))
        adapter = EngineReadOnlyPoseAdapter(FakeEngine(keypoints={}))
        assert adapter.get_landmarks(img) is None


class TestTryGetPoseEngine:
    def test_non_cuda_device_short_circuits_without_importing_gpu_stack(self):
        """Must return None immediately for cpu/placeholder setups — this is
        what keeps fit_analysis from ever trying to load GPU models on a
        non-GPU worker."""
        assert try_get_pose_engine(device="cpu", weights_dir="/some/dir") is None

    def test_missing_weights_dir_short_circuits(self):
        assert try_get_pose_engine(device="cuda", weights_dir="") is None

    def test_cuda_with_weights_dir_but_engine_unavailable_returns_none(self):
        """No app.services.gpu_inference_service module reachable in this
        test environment (or engine not yet built) -> must fail soft."""
        result = try_get_pose_engine(device="cuda", weights_dir="/nonexistent/weights")
        assert result is None or hasattr(result, "_openpose")
