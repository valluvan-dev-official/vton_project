"""
Lightweight CPU compositing engines for accessory try-on (watches/bracelets,
glasses) — landmark detection + a 2D perspective warp of an already
background-removed product cutout onto the person photo. Deliberately NOT
the same technique as garment try-on (see ml/scripts/gpu_inference.py,
IDM-VTON diffusion) — placing a rigid object like a watch or glasses is a
much simpler problem than generatively re-rendering fabric drape on a body,
so this runs on CPU in the same worker process, no GPU job needed.

Phase 1: WristOverlayEngine (MediaPipe Hands).
Phase 2: GlassesOverlayEngine (MediaPipe Face Mesh) — same shape as Phase 1.
Both registered in ACCESSORY_ENGINES below; the route/task layer dispatches
generically by accessory_type, so adding either required zero changes to
api/app/routes/accessory.py or the process_accessory_job Celery task.
"""
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class LandmarkNotDetectedError(Exception):
    """Raised when the required landmarks (wrist, face, ...) couldn't be found in the person photo."""


class WristOverlayEngine:
    """Watch/bracelet placement via MediaPipe Hands landmarks."""

    # MediaPipe Hands landmark indices.
    WRIST = 0
    INDEX_MCP = 5
    PINKY_MCP = 17

    def __init__(self, min_detection_confidence: float = 0.5):
        self._min_confidence = min_detection_confidence
        self._hands = None  # lazy-init — loading the model isn't free

    def _get_hands(self):
        if self._hands is None:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=self._min_confidence,
            )
        return self._hands

    def detect_wrist(self, person_bgr: np.ndarray):
        """Returns (wrist_xy, forearm_angle_deg, wrist_width_px) or None if no hand found."""
        h, w = person_bgr.shape[:2]
        rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
        result = self._get_hands().process(rgb)
        if not result.multi_hand_landmarks:
            return None

        lm = result.multi_hand_landmarks[0].landmark

        def pt(i):
            return np.array([lm[i].x * w, lm[i].y * h])

        wrist = pt(self.WRIST)
        index_mcp = pt(self.INDEX_MCP)
        pinky_mcp = pt(self.PINKY_MCP)

        # Forearm axis: from the knuckle midpoint back through the wrist —
        # the direction a watch band should align with.
        hand_center = (index_mcp + pinky_mcp) / 2
        direction = wrist - hand_center
        angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])))

        # Wrist width: distance across the two MCP (knuckle) joints, scaled
        # down — the actual wrist is narrower than the knuckle span. This
        # ratio (0.75) is a first-pass estimate, not empirically calibrated —
        # expect to tune against real photos once this ships.
        knuckle_span = float(np.linalg.norm(index_mcp - pinky_mcp))
        wrist_width_px = knuckle_span * 0.75

        return wrist, angle_deg, wrist_width_px

    def validate(self, person_path: str) -> tuple[bool, str | None]:
        """Fast pre-flight check — no compositing, just "can we find a wrist here."""
        person_bgr = cv2.imread(person_path)
        if person_bgr is None:
            return False, "This photo couldn't be read — please choose a different one."
        if self.detect_wrist(person_bgr) is None:
            return False, "No wrist visible in this photo — try a photo showing your hand/wrist clearly."
        return True, None

    def run(self, person_path: str, accessory_path: str, output_path: str) -> None:
        """
        person_path: the profile photo.
        accessory_path: the product's front try-on image — already
            background-removed (transparent PNG) by the same rembg pipeline
            garment images go through, see parik_backend's
            apps/products/utils.py::strip_tryon_background.
        Raises LandmarkNotDetectedError if no wrist is found (caller — the
        Celery task — catches this and fails the job with that message).
        """
        person_bgr = cv2.imread(person_path)
        if person_bgr is None:
            raise ValueError(f"Could not read person image: {person_path}")

        detection = self.detect_wrist(person_bgr)
        if detection is None:
            raise LandmarkNotDetectedError("No wrist/hand detected in this photo.")
        wrist_xy, angle_deg, wrist_width_px = detection

        accessory = Image.open(accessory_path).convert("RGBA")
        # Fit the accessory's own aspect ratio to the detected wrist width —
        # width drives the scale, height follows proportionally.
        scale = wrist_width_px / accessory.width
        new_w = max(1, int(accessory.width * scale))
        new_h = max(1, int(accessory.height * scale))
        accessory_resized = accessory.resize((new_w, new_h), Image.LANCZOS)

        # Rotate to align with the forearm axis. Watch product photos are
        # typically shot "upright" (band vertical), so align the image's
        # vertical axis with the forearm direction (angle_deg - 90).
        accessory_rotated = accessory_resized.rotate(
            -(angle_deg - 90), expand=True, resample=Image.BICUBIC,
        )

        person_rgba = Image.open(person_path).convert("RGBA")
        paste_x = int(wrist_xy[0] - accessory_rotated.width / 2)
        paste_y = int(wrist_xy[1] - accessory_rotated.height / 2)
        person_rgba.alpha_composite(accessory_rotated, (paste_x, paste_y))

        person_rgba.convert("RGB").save(output_path, "JPEG", quality=95)
        logger.info(
            "Wrist overlay: placed accessory at (%.0f, %.0f), angle=%.1f deg, "
            "width=%.0fpx -> %s",
            wrist_xy[0], wrist_xy[1], angle_deg, wrist_width_px, output_path,
        )


_wrist_engine: WristOverlayEngine | None = None


def get_wrist_engine() -> WristOverlayEngine:
    global _wrist_engine
    if _wrist_engine is None:
        _wrist_engine = WristOverlayEngine()
    return _wrist_engine


class GlassesOverlayEngine:
    """Glasses placement via MediaPipe Face Mesh landmarks."""

    # MediaPipe Face Mesh landmark indices (468-point topology).
    RIGHT_EYE_OUTER = 33    # subject's right eye, outer corner (camera-left side of frame)
    LEFT_EYE_OUTER = 263    # subject's left eye, outer corner (camera-right side of frame)
    NOSE_BRIDGE = 168       # between the eyebrows — vertical anchor

    def __init__(self, min_detection_confidence: float = 0.5):
        self._min_confidence = min_detection_confidence
        self._face_mesh = None  # lazy-init — loading the model isn't free

    def _get_face_mesh(self):
        if self._face_mesh is None:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=self._min_confidence,
            )
        return self._face_mesh

    def detect_face(self, person_bgr: np.ndarray):
        """Returns (anchor_xy, roll_angle_deg, glasses_width_px) or None if no face found."""
        h, w = person_bgr.shape[:2]
        rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
        result = self._get_face_mesh().process(rgb)
        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark

        def pt(i):
            return np.array([lm[i].x * w, lm[i].y * h])

        right_eye = pt(self.RIGHT_EYE_OUTER)
        left_eye = pt(self.LEFT_EYE_OUTER)
        nose_bridge = pt(self.NOSE_BRIDGE)

        # Roll: angle of the line between the two outer eye corners — how
        # much the head is tilted in-plane. Glasses rotate to match.
        direction = left_eye - right_eye
        angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])))

        # Width: eye-to-eye span scaled up — glasses extend past both eyes
        # to the temples. This ratio (1.9) is a first-pass estimate, not
        # empirically calibrated — same caveat as wrist_width_px above,
        # expect to tune once real photos run through this.
        eye_span = float(np.linalg.norm(left_eye - right_eye))
        glasses_width_px = eye_span * 1.9

        # Anchor: horizontally centered between the eyes, vertically at the
        # nose bridge (glasses sit ON the nose bridge, not at eye height).
        anchor = np.array([(right_eye[0] + left_eye[0]) / 2, nose_bridge[1]])

        return anchor, angle_deg, glasses_width_px

    def validate(self, person_path: str) -> tuple[bool, str | None]:
        """Fast pre-flight check — no compositing, just "can we find a face here."""
        person_bgr = cv2.imread(person_path)
        if person_bgr is None:
            return False, "This photo couldn't be read — please choose a different one."
        if self.detect_face(person_bgr) is None:
            return False, "No face detected in this photo — try a clearer, front-facing photo."
        return True, None

    def run(self, person_path: str, accessory_path: str, output_path: str) -> None:
        """
        person_path: the profile photo.
        accessory_path: the product's front try-on image — already
            background-removed (transparent PNG), same pipeline as garment
            and wrist accessory images.
        Raises LandmarkNotDetectedError if no face is found (caller — the
        Celery task — catches this and fails the job with that message).
        """
        person_bgr = cv2.imread(person_path)
        if person_bgr is None:
            raise ValueError(f"Could not read person image: {person_path}")

        detection = self.detect_face(person_bgr)
        if detection is None:
            raise LandmarkNotDetectedError("No face detected in this photo.")
        anchor_xy, angle_deg, glasses_width_px = detection

        accessory = Image.open(accessory_path).convert("RGBA")
        # Fit the accessory's own aspect ratio to the detected glasses width —
        # width drives the scale, height follows proportionally.
        scale = glasses_width_px / accessory.width
        new_w = max(1, int(accessory.width * scale))
        new_h = max(1, int(accessory.height * scale))
        accessory_resized = accessory.resize((new_w, new_h), Image.LANCZOS)

        # Rotate to match head roll. Glasses product photos are typically
        # shot level/upright (no rotate() offset needed the way the wrist
        # band image needs a -90deg axis correction), so align directly with
        # the measured roll angle.
        accessory_rotated = accessory_resized.rotate(
            -angle_deg, expand=True, resample=Image.BICUBIC,
        )

        person_rgba = Image.open(person_path).convert("RGBA")
        paste_x = int(anchor_xy[0] - accessory_rotated.width / 2)
        paste_y = int(anchor_xy[1] - accessory_rotated.height / 2)
        person_rgba.alpha_composite(accessory_rotated, (paste_x, paste_y))

        person_rgba.convert("RGB").save(output_path, "JPEG", quality=95)
        logger.info(
            "Glasses overlay: placed accessory at (%.0f, %.0f), angle=%.1f deg, "
            "width=%.0fpx -> %s",
            anchor_xy[0], anchor_xy[1], angle_deg, glasses_width_px, output_path,
        )


_glasses_engine: GlassesOverlayEngine | None = None


def get_glasses_engine() -> GlassesOverlayEngine:
    global _glasses_engine
    if _glasses_engine is None:
        _glasses_engine = GlassesOverlayEngine()
    return _glasses_engine


# Registry keyed by accessory_type, the same string the API/task layer uses —
# adding a new accessory engine is "implement it above + add a line here,"
# not touching the route/task dispatch code (both already iterate this dict).
ACCESSORY_ENGINES = {
    "wrist": get_wrist_engine,
    "glasses": get_glasses_engine,
}
