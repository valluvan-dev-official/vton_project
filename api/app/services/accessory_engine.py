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

    # Raised from MediaPipe's own default (0.5) — a hand tucked in a pocket
    # with only a sliver of fingers visible was still clearing 0.5 and
    # producing a garbage detection (tiny knuckle span -> comically small
    # watch, placed on whatever fragment was actually visible instead of a
    # real wrist). 0.75 filters out these partial/occluded detections;
    # combined with the handedness-score check and the width sanity check
    # below, this is a first-pass threshold, not empirically tuned against
    # a real dataset of hand photos — expect to revisit if it starts
    # rejecting genuinely-visible wrists too.
    MIN_HAND_CONFIDENCE = 0.75
    # A knuckle span narrower than this fraction of the photo's width can't
    # be a real, fully-visible hand in a normal portrait framing — it's a
    # sign the detector only found a small fragment (fingertips peeking out
    # of a pocket/sleeve, not an actual exposed wrist).
    MIN_WRIST_WIDTH_FRACTION = 0.03

    def __init__(self, min_detection_confidence: float = MIN_HAND_CONFIDENCE):
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
        """Returns (wrist_xy, forearm_angle_deg, wrist_width_px) or None if
        no hand found, or if what was found doesn't look like a genuinely
        visible wrist (low handedness confidence, or an implausibly narrow
        knuckle span — see MIN_WRIST_WIDTH_FRACTION)."""
        h, w = person_bgr.shape[:2]
        rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
        result = self._get_hands().process(rgb)
        if not result.multi_hand_landmarks:
            return None

        # multi_handedness carries the model's own confidence for this
        # specific detection — a stricter, explicit check on top of the
        # min_detection_confidence gate already configured above, so we can
        # log/tune this threshold independently of the model's internal one.
        if result.multi_handedness:
            score = result.multi_handedness[0].classification[0].score
            if score < self.MIN_HAND_CONFIDENCE:
                logger.info("Wrist detection rejected: handedness score %.2f below threshold %.2f",
                            score, self.MIN_HAND_CONFIDENCE)
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

        if knuckle_span < w * self.MIN_WRIST_WIDTH_FRACTION:
            logger.info("Wrist detection rejected: knuckle span %.0fpx too narrow for a %dpx-wide photo "
                        "(likely a partial/occluded hand, not a genuinely visible wrist)", knuckle_span, w)
            return None

        return wrist, angle_deg, wrist_width_px

    def validate(self, person_path: str) -> tuple[bool, str | None]:
        """Fast pre-flight check — no compositing, just "can we find a wrist here."""
        person_bgr = cv2.imread(person_path)
        if person_bgr is None:
            return False, "This photo couldn't be read — please choose a different one."
        if self.detect_wrist(person_bgr) is None:
            return False, ("No wrist clearly visible in this photo — try a photo with your hand and "
                            "wrist fully visible, not in a pocket or covered by a sleeve.")
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
            raise LandmarkNotDetectedError(
                "No wrist clearly visible in this photo — try a photo with your hand and wrist "
                "fully visible, not in a pocket or covered by a sleeve."
            )
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
    """
    Glasses placement via MediaPipe Face Mesh landmarks.

    Uses a full affine warp (cv2.warpAffine) derived from a 3-point
    correspondence — both outer eye corners + the nose bridge — rather than
    a rigid rotate+scale+paste. A pure rotate+scale (a "similarity"
    transform: uniform scale, one rotation angle, translate — 4 degrees of
    freedom) can only correct for in-plane head TILT (roll). It cannot
    express the SHEAR that a turned head (yaw) or an off-angle accessory
    product photo introduces, which is what made earlier output look like
    the glasses were simply pasted on flat regardless of the photo's
    geometry. An affine transform (6 DOF: adds shear + independent
    horizontal/vertical scale) captures that, derived directly from where
    the accessory's own lens/bridge points need to land on the face.
    """

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

    def detect_face_points(self, person_bgr: np.ndarray):
        """
        Returns (image_left_eye_xy, image_right_eye_xy, nose_bridge_xy) in
        person-image pixel coordinates, or None if no face found.
        "left"/"right" here mean image-left/image-right (camera view) —
        the same convention the accessory product photo is naturally laid
        out in — not anatomical left/right.
        """
        h, w = person_bgr.shape[:2]
        rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
        result = self._get_face_mesh().process(rgb)
        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark

        def pt(i):
            return (lm[i].x * w, lm[i].y * h)

        image_left_eye = pt(self.RIGHT_EYE_OUTER)    # camera-left side of frame
        image_right_eye = pt(self.LEFT_EYE_OUTER)    # camera-right side of frame
        nose_bridge = pt(self.NOSE_BRIDGE)
        return image_left_eye, image_right_eye, nose_bridge

    @staticmethod
    def _derive_accessory_anchor_points(accessory_rgba: np.ndarray):
        """
        Derives (image_left_point, image_right_point, top_center_point)
        directly from the accessory's OWN alpha-channel silhouette, instead
        of assuming every product photo is framed the same way. Real
        product photos vary a lot in crop/framing (a full frontal shot vs.
        a close-up on the hinge/bridge area, different padding, etc.) — a
        fixed fraction like "(0.22, 0.50) is always the left lens" only
        happens to be right for one specific framing and warps badly
        (stretching whatever content actually sits at that fraction) on
        anything cropped differently. This instead measures the actual
        non-transparent content: left/right anchors are the centroid of
        each half of the silhouette (split at its horizontal midpoint),
        and the top anchor is the topmost non-transparent pixel within the
        central band — which for a glasses cutout is the bridge regardless
        of how tightly or loosely the photo is cropped.
        """
        alpha = accessory_rgba[:, :, 3]
        ys, xs = np.nonzero(alpha > 10)
        if len(xs) == 0:
            raise ValueError("Accessory image has no visible (non-transparent) content.")

        x_min, x_max = float(xs.min()), float(xs.max())
        y_min = float(ys.min())
        mid_x = (x_min + x_max) / 2

        left_side = xs < mid_x
        right_side = ~left_side
        left_point = (
            (float(xs[left_side].mean()), float(ys[left_side].mean()))
            if left_side.any() else (x_min, (y_min + float(ys.max())) / 2)
        )
        right_point = (
            (float(xs[right_side].mean()), float(ys[right_side].mean()))
            if right_side.any() else (x_max, (y_min + float(ys.max())) / 2)
        )

        span = x_max - x_min
        central_band = (xs > mid_x - span * 0.15) & (xs < mid_x + span * 0.15)
        top_point = (mid_x, float(ys[central_band].min())) if central_band.any() else (mid_x, y_min)

        return left_point, right_point, top_point

    def validate(self, person_path: str) -> tuple[bool, str | None]:
        """Fast pre-flight check — no compositing, just "can we find a face here."""
        person_bgr = cv2.imread(person_path)
        if person_bgr is None:
            return False, "This photo couldn't be read — please choose a different one."
        if self.detect_face_points(person_bgr) is None:
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

        points = self.detect_face_points(person_bgr)
        if points is None:
            raise LandmarkNotDetectedError("No face detected in this photo.")
        image_left_eye, image_right_eye, nose_bridge = points

        accessory = Image.open(accessory_path).convert("RGBA")
        accessory_np = np.array(accessory)  # H x W x 4 (RGBA)

        accessory_left, accessory_right, accessory_top = self._derive_accessory_anchor_points(accessory_np)
        src_pts = np.float32([accessory_left, accessory_right, accessory_top])
        dst_pts = np.float32([image_left_eye, image_right_eye, nose_bridge])

        # Full affine transform derived directly from these 3
        # correspondences — captures rotation, scale, AND shear together,
        # so a turned/tilted head (or an accessory photo that isn't
        # perfectly frontal) warps the glasses to actually fit the face
        # geometry instead of pasting a rigid, uniformly-scaled cutout.
        M = cv2.getAffineTransform(src_pts, dst_pts)

        person_h, person_w = person_bgr.shape[:2]
        warped = cv2.warpAffine(
            accessory_np, M, (person_w, person_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
        )

        person_rgba = Image.open(person_path).convert("RGBA")
        warped_pil = Image.fromarray(warped, mode="RGBA")
        person_rgba.alpha_composite(warped_pil)

        person_rgba.convert("RGB").save(output_path, "JPEG", quality=95)
        logger.info(
            "Glasses overlay: affine-warped accessory anchors=%s/%s/%s -> face eyes=%s/%s, bridge=%s -> %s",
            accessory_left, accessory_right, accessory_top,
            image_left_eye, image_right_eye, nose_bridge, output_path,
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
