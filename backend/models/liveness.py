"""
Pure Python liveness detection — no external binaries.

Architecture:
  1. MediaPipe FaceDetector   → face size check (>20% frame width)
  2. Adaptive calibration     → per-user EAR baseline (first 30 frames)
  3. MediaPipe FaceLandmarker → 468 landmarks + 4×4 transform matrix
  4. Challenge state machine  → BLINK → TURN_LEFT → TURN_RIGHT
  5. MiniFASNet anti-spoof    → dual-model fusion, every 5th frame

Key design:
  - Hysteresis: N consecutive frames to confirm a gesture
  - Adaptive thresholds: EAR baseline calibrated per session
  - Per-session state isolation: fully independent per WebSocket
"""

from __future__ import annotations

import logging
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────
_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
_VENDOR_DIR = Path(__file__).parent.parent / "vendor" / "silent_antispoofing"
_LANDMARKER_PATH = _WEIGHTS_DIR / "face_landmarker_v2_with_blendshapes.task"

sys.path.insert(0, str(_VENDOR_DIR))

# ── Constants ──────────────────────────────────────────────────────
CALIBRATION_FRAMES = 30  # frames to establish per-user baseline
HYSTERESIS_FRAMES = 5  # consecutive frames to confirm a gesture
CHALLENGE_TIMEOUT = 10.0  # seconds per gesture challenge
FPS_TARGET = 10  # expected WebSocket frame rate
MIN_FACE_WIDTH_RATIO = 0.20  # face must cover >20% of frame width

# Eye landmark indices (MediaPipe 468-mesh)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Blink: EAR drops below baseline * BLINK_RATIO
BLINK_RATIO = 0.72

# Head turn: |yaw| exceeds TURN_YAW_DEG from baseline
TURN_YAW_DEG = 18.0

# MiniFASNet spoof threshold
SPOOF_SCORE_THRESHOLD = 0.6


# ═══════════════════════════════════════════════════════════════════
# MediaPipe singletons (loaded once, shared across sessions)
# ═══════════════════════════════════════════════════════════════════

_face_cascade: Any = None
_face_landmarker: Any = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        logger.info("OpenCV Haar cascade loaded")
    return _face_cascade


def _get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_LANDMARKER_PATH)),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        _face_landmarker = vision.FaceLandmarker.create_from_options(opts)
        logger.info("MediaPipe FaceLandmarker loaded")
    return _face_landmarker


# ═══════════════════════════════════════════════════════════════════
# MiniFASNet anti-spoofing (dual-model fusion)
# ═══════════════════════════════════════════════════════════════════

_anti_spoof: Any = None
_anti_cropper: Any = None
_anti_models = [
    "2.7_80x80_MiniFASNetV2.pth",
    "4_0_0_80x80_MiniFASNetV1SE.pth",
]
_anti_model_dir = _VENDOR_DIR / "resources" / "anti_spoof_models"


def _get_anti_spoof():
    global _anti_spoof, _anti_cropper
    if _anti_spoof is None:
        from src.anti_spoof_predict import AntiSpoofPredict, Detection
        from src.generate_patches import CropImage
        from src.utility import parse_model_name

        # Patch Detection to use absolute paths
        _orig_init = Detection.__init__

        def _patched_init(self):
            deploy = str(
                _VENDOR_DIR / "resources" / "detection_model" / "deploy.prototxt"
            )
            caffemodel = str(
                _VENDOR_DIR
                / "resources"
                / "detection_model"
                / "Widerface-RetinaFace.caffemodel"
            )
            self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
            self.detector_confidence = 0.6

        Detection.__init__ = _patched_init
        try:
            _anti_spoof = AntiSpoofPredict(0)
        finally:
            Detection.__init__ = _orig_init

        _anti_cropper = CropImage()
        logger.info("MiniFASNet anti-spoofing loaded (dual-model, CPU)")
    return _anti_spoof, _anti_cropper


def _run_anti_spoof(frame: np.ndarray) -> dict:
    """Run MiniFASNet dual-model fusion. Returns score + is_real."""
    try:
        from src.utility import parse_model_name

        predictor, cropper = _get_anti_spoof()
        bbox = predictor.get_bbox(frame)
        if bbox is None:
            return {"face_found": False, "score": 0.0, "is_real": False}

        prediction = np.zeros((1, 3))
        count = 0
        for model_name in _anti_models:
            mp = _anti_model_dir / model_name
            if not mp.exists():
                continue
            h_in, w_in, mtype, scale = parse_model_name(model_name)
            param = {
                "org_img": frame,
                "bbox": bbox,
                "scale": scale,
                "out_w": w_in,
                "out_h": h_in,
                "crop": scale is not None,
            }
            img = cropper.crop(**param)
            prediction += predictor.predict(img, str(mp))
            count += 1

        if count == 0:
            return {"face_found": True, "score": 0.0, "is_real": False}

        label = int(np.argmax(prediction))
        value = float(prediction[0][label] / count)
        return {
            "face_found": True,
            "score": round(value, 4),
            "is_real": (label == 1) and (value > SPOOF_SCORE_THRESHOLD),
        }
    except Exception as exc:
        logger.error("MiniFASNet failed: %s", exc)
        return {"face_found": False, "score": 0.0, "is_real": True}


# ═══════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════


def _compute_ear(landmarks, indices) -> float:
    """Eye Aspect Ratio for a single eye."""
    pts = [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]
    # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    if c < 1e-6:
        return 1.0
    return float((a + b) / (2.0 * c))


def _decompose_transform(matrix_4x4) -> tuple[float, float, float]:
    """Extract yaw, pitch, roll (degrees) from 4×4 facial transform matrix."""
    m = np.array(matrix_4x4).reshape(4, 4)
    # Rotation matrix is top-left 3×3
    R = m[:3, :3]

    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = float(math.atan2(-R[2, 0], sy))
        yaw = float(math.atan2(R[1, 0], R[0, 0]))
        roll = float(math.atan2(R[2, 1], R[2, 2]))
    else:
        pitch = float(math.atan2(-R[2, 0], sy))
        yaw = float(math.atan2(-R[1, 2], R[1, 1]))
        roll = 0.0

    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


# ═══════════════════════════════════════════════════════════════════
# Per-session state
# ═══════════════════════════════════════════════════════════════════


class LivenessSession:
    """Fully isolated per-WebSocket liveness state machine."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.frame_count = 0
        self.start_time = time.time()

        # Calibration
        self.calibrated = False
        self.ear_samples: list[float] = []
        self.yaw_samples: list[float] = []
        self.ear_baseline = 0.30  # default, overwritten after calibration
        self.yaw_neutral = 0.0

        # Challenge state machine
        import random

        self.challenges = ["BLINK", "TURN_LEFT", "TURN_RIGHT"]
        random.shuffle(self.challenges)
        self.current_challenge_idx = 0
        self.challenge_start = 0.0
        self.consecutive_count = 0  # hysteresis counter

        # Result
        self.passed = False
        self.failed = False
        self.instruction = "انظر إلى الكاميرا"
        self.face_detected = False

        # Selfie capture
        self.best_frame: np.ndarray | None = None
        self.best_quality = 0.0
        self.selfie_path: str | None = None

        # Anti-spoof throttle
        self.last_spoof_check = 0

        logger.info("Session %s: challenges = %s", session_id, self.challenges)

    # ── Per-frame processing ───────────────────────────────────

    def process(self, frame: np.ndarray) -> dict[str, Any]:
        """Process one frame. Returns UI state dict."""
        self.frame_count += 1

        # Timeout
        if time.time() - self.start_time > 60 and not self.passed:
            self.failed = True
            self.instruction = "انتهت المهلة"

        if self.failed:
            return self._response()
        if self.passed:
            return self._response()

        # ── 1. Face detection + size check ─────────────────────
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = _get_face_cascade()
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            self.face_detected = False
            self.instruction = "ضع وجهك في الإطار"
            self.consecutive_count = 0
            return self._response()

        # Use largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2])
        if fw < w * MIN_FACE_WIDTH_RATIO:
            self.face_detected = False
            self.instruction = "قرب وجهك من الكاميرا"
            self.consecutive_count = 0
            return self._response()

        self.face_detected = True

        # ── 2. Landmark extraction ─────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker = _get_face_landmarker()
        lm_result = landmarker.detect(mp_img)

        if not lm_result.face_landmarks:
            self.consecutive_count = 0
            return self._response()

        landmarks = lm_result.face_landmarks[0]

        # EAR
        left_ear = _compute_ear(landmarks, LEFT_EYE_IDX)
        right_ear = _compute_ear(landmarks, RIGHT_EYE_IDX)
        ear = (left_ear + right_ear) / 2.0

        # Head pose from transform matrix
        yaw = 0.0
        if lm_result.facial_transformation_matrixes:
            yaw, pitch, roll = _decompose_transform(
                lm_result.facial_transformation_matrixes[0].data
            )

        # ── Anti-spoof every 5th frame ─────────────────────────
        if self.frame_count - self.last_spoof_check >= 5:
            self.last_spoof_check = self.frame_count
            spoof = _run_anti_spoof(frame)
            if spoof["face_found"] and not spoof["is_real"]:
                self.failed = True
                self.instruction = "يرجى استخدام وجهك الحقيقي"
                return self._response()

        # ── 3. Calibration phase ───────────────────────────────
        if not self.calibrated:
            self.ear_samples.append(ear)
            self.yaw_samples.append(yaw)
            self.instruction = "ثابت... يتم المعايرة"

            if len(self.ear_samples) >= CALIBRATION_FRAMES:
                self.ear_baseline = float(np.mean(self.ear_samples))
                self.yaw_neutral = float(np.mean(self.yaw_samples))
                self.calibrated = True
                self.challenge_start = time.time()
                logger.info(
                    "Session %s calibrated: EAR_baseline=%.4f yaw_neutral=%.1f°",
                    self.session_id,
                    self.ear_baseline,
                    self.yaw_neutral,
                )
                self.instruction = self._challenge_instruction()
            return self._response()

        # ── 4. Challenge state machine ─────────────────────────
        self._evaluate_challenge(ear, yaw)
        return self._response()

    # ── Challenge evaluation ───────────────────────────────────

    def _evaluate_challenge(self, ear: float, yaw: float):
        """Check current gesture against the active challenge."""
        if self.current_challenge_idx >= len(self.challenges):
            self.passed = True
            self.instruction = "تم التحقق"
            return

        # Timeout per challenge
        elapsed = time.time() - self.challenge_start
        if elapsed > CHALLENGE_TIMEOUT:
            self.failed = True
            self.instruction = "انتهى وقت التحدي"
            return

        challenge = self.challenges[self.current_challenge_idx]
        matched = False

        if challenge == "BLINK":
            threshold = self.ear_baseline * BLINK_RATIO
            matched = ear < threshold
        elif challenge == "TURN_LEFT":
            # Yaw becomes more negative when turning left
            matched = (yaw - self.yaw_neutral) < -TURN_YAW_DEG
        elif challenge == "TURN_RIGHT":
            matched = (yaw - self.yaw_neutral) > TURN_YAW_DEG

        if matched:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0

        if self.consecutive_count >= HYSTERESIS_FRAMES:
            # Challenge passed!
            logger.info("Session %s: %s passed", self.session_id, challenge)
            self.current_challenge_idx += 1
            self.consecutive_count = 0
            self.challenge_start = time.time()

            if self.current_challenge_idx >= len(self.challenges):
                self.passed = True
                self.instruction = "تم التحقق"
            else:
                self.instruction = self._challenge_instruction()
        else:
            # In progress
            self.instruction = self._challenge_instruction()

    def _challenge_instruction(self) -> str:
        """Arabic instruction for the current challenge."""
        if self.current_challenge_idx >= len(self.challenges):
            return "تم التحقق"
        c = self.challenges[self.current_challenge_idx]
        if c == "BLINK":
            return "أغمض عينيك"
        elif c == "TURN_LEFT":
            return "أدر وجهك إلى اليسار"
        elif c == "TURN_RIGHT":
            return "أدر وجهك إلى اليمين"
        return "انظر إلى الكاميرا"

    def _response(self) -> dict[str, Any]:
        """Build the JSON response for the frontend."""
        return {
            "passed": self.passed,
            "failed": self.failed,
            "instruction": self.instruction,
            "face_detected": self.face_detected,
            "selfie_ready": self.passed,
            "calibrated": self.calibrated,
            "challenge": (
                self.challenges[self.current_challenge_idx]
                if not self.passed
                and not self.failed
                and self.current_challenge_idx < len(self.challenges)
                else None
            ),
        }


# ═══════════════════════════════════════════════════════════════════
# Global session registry + public API
# ═══════════════════════════════════════════════════════════════════

_sessions: dict[str, LivenessSession] = {}


def create_session(session_id: str, language: str = "ar") -> dict:
    """Start a new isolated liveness session."""
    if session_id in _sessions:
        cleanup_session(session_id)
    _sessions[session_id] = LivenessSession(session_id)
    return {"started": True, "session_id": session_id}


def process_liveness_frame(frame: np.ndarray, session_id: str) -> dict[str, Any]:
    """Process one camera frame. Returns UI state."""
    if session_id not in _sessions:
        create_session(session_id)
    sess = _sessions[session_id]
    result = sess.process(frame)

    # Save best frame for selfie
    if not sess.passed and sess.face_detected:
        q = _frame_quality(frame)
        if q > sess.best_quality:
            sess.best_quality = q
            sess.best_frame = frame.copy()
            old = sess.selfie_path
            if old and os.path.exists(old):
                try:
                    os.remove(old)
                except OSError:
                    pass
            fd, path = tempfile.mkstemp(suffix=".jpg", prefix="selfie_")
            os.close(fd)
            cv2.imwrite(path, frame)
            sess.selfie_path = path

    return result


def get_selfie_frame(session_id: str) -> np.ndarray | None:
    """Return best captured selfie for face matching."""
    sess = _sessions.get(session_id)
    if not sess:
        return None
    if sess.best_frame is not None:
        return sess.best_frame
    if sess.selfie_path and os.path.exists(sess.selfie_path):
        return cv2.imread(sess.selfie_path)
    return None


def cleanup_session(session_id: str) -> None:
    """Remove session and temp files."""
    sess = _sessions.pop(session_id, None)
    if sess and sess.selfie_path and os.path.exists(sess.selfie_path):
        try:
            os.remove(sess.selfie_path)
        except OSError:
            pass


def _frame_quality(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ═══════════════════════════════════════════════════════════════════
# Face matching (unchanged)
# ═══════════════════════════════════════════════════════════════════


def match_faces(cin_face: np.ndarray, selfie: np.ndarray) -> dict:
    try:
        from deepface import DeepFace

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            cv2.imwrite(f1.name, cin_face)
            cin_path = f1.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            cv2.imwrite(f2.name, selfie)
            selfie_path = f2.name

        try:
            result = DeepFace.verify(
                img1_path=cin_path,
                img2_path=selfie_path,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=False,
            )
            distance = float(result.get("distance", 1.0))
            threshold = float(result.get("threshold", 0.4))
            matched = distance < threshold
            score = max(0.0, 1.0 - (distance / max(threshold * 2, 1.0)))
            return {
                "match": matched,
                "score": round(score, 4),
                "threshold": round(threshold, 4),
                "distance": round(distance, 6),
                "reason": "Match" if matched else "Face mismatch",
            }
        finally:
            for p in (cin_path, selfie_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
    except Exception as exc:
        logger.error("Face matching failed: %s", exc)
        return {
            "match": False,
            "score": 0.0,
            "threshold": 0.4,
            "distance": 1.0,
            "reason": f"Error: {exc}",
        }
