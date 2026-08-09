"""Read-only adapter over an already-loaded pose/parsing engine.

Why this file exists (Phase-1 spec item 4): body_analyzer.py needs pixel
landmarks (shoulder width, torso length, hip width, arm length) to turn a
photo into a BodyMeasurementEstimate. ml/scripts/gpu_inference.py's
GPUInferenceEngine already loads OpenPose + SCHP once per worker process
and calls them internally inside _get_agnostic_mask() — but that method is
private to the try-on render path and returns a mask, not landmarks, so we
can't just import a function from it.

Rather than modify gpu_inference.py to expose more internals (explicitly
out of scope — see Phase-1 spec item 11), this module calls the SAME
already-loaded model objects a second, independent time, read-only:
    engine._openpose(img)  -> dict with "pose_keypoints_2d"
    engine._parser(img)    -> (parse_result, face_mask)
These are plain inference calls (no state mutation) already used exactly
this way by gpu_inference.py itself. Zero lines of gpu_inference.py change
because of this file.

Structural typing (Protocol) means this module never imports
GPUInferenceEngine and has no import-time dependency on torch/diffusers —
it only cares that *some* object exposes these two callables, which keeps
this file trivially unit-testable with a plain fake.

Any failure here (engine unavailable, model call raises, malformed output)
must never propagate — callers treat None as "no pose data available" and
fall back to a lower-confidence, landmarks-free estimate.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

# OpenPose COCO-18 keypoint indices — the same public, standard layout
# ml/scripts/gpu_inference.py itself indexes into (see its
# _estimate_person_size / _render_pose_image). Duplicated here as a plain
# constant, not imported, so this package has no import-time coupling to
# ml/scripts.
KP_NOSE, KP_NECK = 0, 1
KP_R_SHOULDER, KP_R_ELBOW, KP_R_WRIST = 2, 3, 4
KP_L_SHOULDER, KP_L_ELBOW, KP_L_WRIST = 5, 6, 7
KP_R_HIP, KP_L_HIP = 8, 11

# The same resize convention ml/scripts/gpu_inference.py uses before
# handing an image to OpenPose/SCHP (PARSE_W, PARSE_H there). Kept as a
# local constant for the same reason as above: no import coupling.
POSE_INPUT_W, POSE_INPUT_H = 384, 512

# Number of the above landmarks required for a "fully detected" pose —
# drives the quality score fed into confidence.estimate_body_confidence().
_REQUIRED_LANDMARK_INDICES = (
    KP_NECK, KP_R_SHOULDER, KP_L_SHOULDER, KP_R_ELBOW, KP_L_ELBOW,
    KP_R_WRIST, KP_L_WRIST, KP_R_HIP, KP_L_HIP,
)


class PoseParseEngine(Protocol):
    """Structural type for 'something that can run OpenPose + SCHP'.
    GPUInferenceEngine satisfies this without knowing it exists."""

    def _openpose(self, img) -> dict: ...

    def _parser(self, img): ...


@dataclass
class PoseLandmarks:
    """Pixel-space landmarks extracted from one frontal photo, plus a
    quality score (fraction of required joints actually detected)."""
    shoulder_width_px: Optional[float]
    torso_length_px: Optional[float]
    hip_width_px: Optional[float]
    arm_length_px: Optional[float]
    image_width: int
    image_height: int
    quality: float  # 0..1


def _pt(candidate: list, idx: int) -> Optional[tuple[float, float]]:
    if idx >= len(candidate):
        return None
    x, y = candidate[idx][0], candidate[idx][1]
    if x <= 0 and y <= 0:
        return None
    return (float(x), float(y))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def landmarks_from_keypoints(keypoints: dict, image_width: int, image_height: int) -> PoseLandmarks:
    """Pure function: OpenPose keypoints dict -> PoseLandmarks. Split out
    from get_landmarks() so it's testable without any engine/adapter at all."""
    candidate = keypoints.get("pose_keypoints_2d", []) if keypoints else []

    detected = sum(1 for i in _REQUIRED_LANDMARK_INDICES if _pt(candidate, i) is not None)
    quality = detected / len(_REQUIRED_LANDMARK_INDICES)

    r_sh, l_sh = _pt(candidate, KP_R_SHOULDER), _pt(candidate, KP_L_SHOULDER)
    neck = _pt(candidate, KP_NECK)
    r_hip, l_hip = _pt(candidate, KP_R_HIP), _pt(candidate, KP_L_HIP)
    r_elbow, r_wrist = _pt(candidate, KP_R_ELBOW), _pt(candidate, KP_R_WRIST)

    shoulder_width_px = _dist(r_sh, l_sh) if (r_sh and l_sh) else None

    hip_pts = [p for p in (r_hip, l_hip) if p]
    torso_length_px = None
    if neck and hip_pts:
        mid_hip_y = sum(p[1] for p in hip_pts) / len(hip_pts)
        torso_length_px = abs(mid_hip_y - neck[1]) or None

    hip_width_px = _dist(r_hip, l_hip) if (r_hip and l_hip) else None

    arm_length_px = None
    if r_sh and r_elbow and r_wrist:
        arm_length_px = _dist(r_sh, r_elbow) + _dist(r_elbow, r_wrist)

    return PoseLandmarks(
        shoulder_width_px=shoulder_width_px,
        torso_length_px=torso_length_px,
        hip_width_px=hip_width_px,
        arm_length_px=arm_length_px,
        image_width=image_width,
        image_height=image_height,
        quality=quality,
    )


class EngineReadOnlyPoseAdapter:
    """Wraps a PoseParseEngine-shaped object for read-only landmark
    extraction. Never raises — every failure mode returns None so a
    shadow-mode analysis can never break the caller."""

    def __init__(self, engine: PoseParseEngine):
        self._engine = engine

    def get_landmarks(self, person_image) -> Optional[PoseLandmarks]:
        """person_image: a PIL.Image already loaded by the caller."""
        try:
            resized = person_image.resize((POSE_INPUT_W, POSE_INPUT_H))
            keypoints = self._engine._openpose(resized)  # noqa: SLF001 — intentional read-only reuse, see module docstring
        except Exception:
            logger.warning("fit_analysis: pose landmark extraction failed; "
                            "continuing without pose data.", exc_info=True)
            return None

        if not keypoints:
            return None

        try:
            return landmarks_from_keypoints(keypoints, *person_image.size)
        except Exception:
            logger.warning("fit_analysis: failed to derive landmarks from "
                            "pose keypoints; continuing without pose data.", exc_info=True)
            return None


def try_get_pose_engine(device: str, weights_dir: str) -> Optional[PoseParseEngine]:
    """Best-effort fetch of the already-loaded GPUInferenceEngine singleton.

    Mirrors InferenceRouter's own local_gpu selection condition (device ==
    "cuda" and weights_dir configured) rather than importing/introspecting
    InferenceRouter, so this stays a read-only, zero-coupling check. Only
    ever called *after* the main try-on inference has already run (so the
    singleton is already built and this is a cheap cache hit) — never
    triggers a fresh model load or S3 bootstrap on its own initiative.
    """
    if (device or "").strip().lower() != "cuda" or not (weights_dir or "").strip():
        return None
    try:
        from app.services.gpu_inference_service import get_gpu_engine
        return get_gpu_engine()
    except Exception:
        logger.warning("fit_analysis: could not obtain the GPU inference engine "
                        "for pose-landmark reuse; continuing without pose data.", exc_info=True)
        return None
