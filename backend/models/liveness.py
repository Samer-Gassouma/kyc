"""
Pure ONNX liveness detection — all models run server-side via onnxruntime.

Models (from Faceplugin-ltd, included in public/models/):
  fr_detect.onnx   → face detection (240×320 → bbox)
  fr_landmark.onnx → 68 face landmarks (64×64 grayscale)
  fr_liveness.onnx → passive liveness score (128×128)
  fr_eye.onnx      → eye open/closed (24×24 grayscale per eye)
  fr_pose.onnx     → head pose yaw/pitch/roll (224×224)

Challenge flow: calibrate → BLINK → TURN_LEFT → TURN_RIGHT → pass
All thresholds permissive — real-world webcam friendly.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).parent.parent.parent / "id-capture" / "public" / "models"

# ── Constants (permissive) ─────────────────────────────────────────
CALIBRATION_FRAMES = 15
HYSTERESIS_FRAMES = 2
CHALLENGE_TIMEOUT = 15.0
BLINK_EYE_RATIO = 0.6  # eye open prob below this = blinking
TURN_YAW_DEG = 12.0  # yaw deviation to trigger turn
MIN_FACE_SCORE = 0.5  # face detection confidence
LIVENESS_SCORE_MIN = 0.3  # real face must score above this on class 1

# ── ONNX sessions (lazy singletons) ────────────────────────────────

_sessions_cache: dict[str, ort.InferenceSession] = {}


def _get_session(name: str) -> ort.InferenceSession:
    if name not in _sessions_cache:
        path = _MODEL_DIR / f"{name}.onnx"
        _sessions_cache[name] = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        logger.info("ONNX %s loaded (%s)", name, path.name)
    return _sessions_cache[name]


# ═══════════════════════════════════════════════════════════════════
# ONNX inference helpers
# ═══════════════════════════════════════════════════════════════════


def _detect_face(frame: np.ndarray) -> tuple | None:
    """Detect largest face. Returns (x1, y1, x2, y2, score) or None."""
    sess = _get_session("fr_detect")
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (320, 240))
    blob = img.astype(np.float32).transpose(2, 0, 1)[None]  # 1×3×240×320
    scores, boxes = sess.run(None, {"input": blob})

    best_idx = int(np.argmax(scores[0, :, 1]))
    best_score = float(scores[0, best_idx, 1])
    if best_score < MIN_FACE_SCORE:
        return None

    bx1, by1, bx2, by2 = boxes[0, best_idx]
    # Scale back to original size
    sx, sy = w / 320.0, h / 240.0
    return (int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy), best_score)


def _get_landmarks(frame: np.ndarray, bbox: tuple) -> list | None:
    """Extract 68 landmarks. Returns list of (x, y) tuples."""
    sess = _get_session("fr_landmark")
    x1, y1, x2, y2 = bbox[:4]
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    blob = resized.astype(np.float32)[None, None] / 255.0  # 1×1×64×64

    result = sess.run(None, {"input": blob})[0][0]  # [136]
    landmarks = []
    fw, fh = x2 - x1, y2 - y1
    for i in range(0, 136, 2):
        lx = x1 + result[i] * fw
        ly = y1 + result[i + 1] * fh
        landmarks.append((float(lx), float(ly)))
    return landmarks


def _check_liveness(frame: np.ndarray, bbox: tuple) -> float:
    """Passive liveness score. Returns probability of 'real' class (0-1)."""
    sess = _get_session("fr_liveness")
    x1, y1, x2, y2 = bbox[:4]
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return 0.0

    resized = cv2.resize(face, (128, 128))
    blob = resized.astype(np.float32).transpose(2, 0, 1)[None]  # 1×3×128×128
    result = sess.run(None, {"input": blob})[0][0]  # [3]
    # Softmax: class 1 = real
    exp = np.exp(result - np.max(result))
    probs = exp / exp.sum()
    return float(probs[1])


def _check_eyes(frame: np.ndarray, landmarks: list) -> tuple[float, float]:
    """Eye open probability for left and right eye. (0-1 each)."""
    sess = _get_session("fr_eye")

    def _eye_prob(eye_pts):
        xs = [p[0] for p in eye_pts]
        ys = [p[1] for p in eye_pts]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return 0.5
        eye = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (24, 24))
        blob = resized.astype(np.float32)[None, None] / 255.0  # 1×1×24×24
        result = sess.run(None, {"input": blob})[0][0]  # [2]
        exp = np.exp(result - np.max(result))
        return float(exp[1] / exp.sum())  # prob of "open"

    # Eye landmark indices (68-point model)
    left_eye_idx = list(range(36, 42))
    right_eye_idx = list(range(42, 48))

    if len(landmarks) < 48:
        return 1.0, 1.0

    left_open = _eye_prob([landmarks[i] for i in left_eye_idx])
    right_open = _eye_prob([landmarks[i] for i in right_eye_idx])
    return left_open, right_open


def _get_pose(frame: np.ndarray, bbox: tuple) -> tuple[float, float, float]:
    """Head pose. Returns (yaw, pitch, roll) in degrees."""
    sess = _get_session("fr_pose")
    x1, y1, x2, y2 = bbox[:4]
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return 0.0, 0.0, 0.0

    # ImageNet normalize
    resized = cv2.resize(face, (224, 224)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    blob = ((resized - mean) / std).transpose(2, 0, 1)[None]

    result = sess.run(None, {"input": blob})[0][0]  # [66]
    # The first 3 values are yaw/pitch/roll in radians
    yaw = float(result[0]) * 180.0 / math.pi
    pitch = float(result[1]) * 180.0 / math.pi
    roll = float(result[2]) * 180.0 / math.pi
    return yaw, pitch, roll


# ═══════════════════════════════════════════════════════════════════
# Per-session state machine
# ═══════════════════════════════════════════════════════════════════


class LivenessSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.frame_count = 0
        self.start_time = time.time()

        # Calibration
        self.calibrated = False
        self.eye_samples: list[float] = []
        self.yaw_samples: list[float] = []
        self.eye_open_baseline = 0.95
        self.yaw_neutral = 0.0

        # Challenge state
        import random

        self.challenges = ["BLINK", "TURN_LEFT", "TURN_RIGHT"]
        random.shuffle(self.challenges)
        self.current_challenge_idx = 0
        self.challenge_start = 0.0
        self.consecutive_count = 0

        # Result
        self.passed = False
        self.failed = False
        self.instruction = "انظر إلى الكاميرا"
        self.face_detected = False

        # Face tracking for frontend
        self.face_bbox: tuple | None = None
        self.face_yaw = 0.0
        self.landmarks_2d: list | None = None

        # Selfie
        self.best_frame: np.ndarray | None = None
        self.best_quality = 0.0
        self.selfie_path: str | None = None

        logger.info("Session %s: challenges=%s", session_id, self.challenges)

    def process(self, frame: np.ndarray) -> dict[str, Any]:
        self.frame_count += 1

        if self.passed:
            return self._response()
        if self.failed:
            return self._response()
        if time.time() - self.start_time > 90:
            self.failed = True
            self.instruction = "انتهت المهلة"
            return self._response()

        # ── 1. Face detection ──────────────────────────────────
        det = _detect_face(frame)
        if det is None:
            self.face_detected = False
            self.instruction = "ضع وجهك في الإطار"
            self.consecutive_count = 0
            return self._response()

        x1, y1, x2, y2, score = det
        self.face_detected = True
        h, w = frame.shape[:2]
        self.face_bbox = (x1, y1, x2 - x1, y2 - y1)

        # ── 2. Landmarks ──────────────────────────────────────
        lm = _get_landmarks(frame, (x1, y1, x2, y2))
        if lm and len(lm) >= 48:
            self.landmarks_2d = [{"x": round(p[0], 1), "y": round(p[1], 1)} for p in lm]
        else:
            self.landmarks_2d = None

        # ── 3. Liveness check ─────────────────────────────────
        liveness = _check_liveness(frame, (x1, y1, x2, y2))
        if liveness < LIVENESS_SCORE_MIN:
            self.failed = True
            self.instruction = "يرجى استخدام وجهك الحقيقي"
            return self._response()

        # ── 4. Eye closeness ──────────────────────────────────
        left_open, right_open = 1.0, 1.0
        if lm and len(lm) >= 48:
            left_open, right_open = _check_eyes(frame, lm)
        eye_open = min(left_open, right_open)

        # ── 5. Head pose ──────────────────────────────────────
        yaw, pitch, roll = _get_pose(frame, (x1, y1, x2, y2))
        self.face_yaw = yaw

        # ── Calibration ───────────────────────────────────────
        if not self.calibrated:
            self.eye_samples.append(eye_open)
            self.yaw_samples.append(yaw)
            self.instruction = "ثابت... يتم المعايرة"
            if len(self.eye_samples) >= CALIBRATION_FRAMES:
                self.eye_open_baseline = float(np.mean(self.eye_samples))
                self.yaw_neutral = float(np.mean(self.yaw_samples))
                self.calibrated = True
                self.challenge_start = time.time()
                logger.info(
                    "Session %s calibrated: eye=%.3f yaw=%.1f",
                    self.session_id,
                    self.eye_open_baseline,
                    self.yaw_neutral,
                )
                self.instruction = self._challenge_instruction()
            return self._response()

        # ── Challenge evaluation ──────────────────────────────
        self._evaluate(eye_open, yaw)
        return self._response()

    def _evaluate(self, eye_open: float, yaw: float):
        if self.current_challenge_idx >= len(self.challenges):
            self.passed = True
            self.instruction = "تم التحقق ✔"
            return

        elapsed = time.time() - self.challenge_start
        if elapsed > CHALLENGE_TIMEOUT:
            self.failed = True
            self.instruction = "انتهى وقت التحدي"
            return

        ch = self.challenges[self.current_challenge_idx]
        matched = False

        if ch == "BLINK":
            matched = eye_open < self.eye_open_baseline * BLINK_EYE_RATIO
        elif ch == "TURN_LEFT":
            matched = (yaw - self.yaw_neutral) < -TURN_YAW_DEG
        elif ch == "TURN_RIGHT":
            matched = (yaw - self.yaw_neutral) > TURN_YAW_DEG

        if matched:
            self.consecutive_count += 1
            remaining = HYSTERESIS_FRAMES - self.consecutive_count
            if remaining <= 0:
                logger.info("Session %s: %s passed", self.session_id, ch)
                self.current_challenge_idx += 1
                self.consecutive_count = 0
                self.challenge_start = time.time()
                if self.current_challenge_idx >= len(self.challenges):
                    self.passed = True
                    self.instruction = "تم التحقق ✔"
                else:
                    self.instruction = self._challenge_instruction() + " ✔"
            else:
                self.instruction = self._challenge_instruction() + f" ({remaining})"
        else:
            self.consecutive_count = 0
            self.instruction = self._challenge_instruction()

    def _challenge_instruction(self) -> str:
        n = self.current_challenge_idx + 1
        total = len(self.challenges)
        prefix = f"({n}/{total}) "
        c = self.challenges[self.current_challenge_idx]
        if c == "BLINK":
            return prefix + "أغمض عينيك"
        if c == "TURN_LEFT":
            return prefix + "أدر وجهك إلى اليسار"
        if c == "TURN_RIGHT":
            return prefix + "أدر وجهك إلى اليمين"
        return prefix

    def _response(self) -> dict:
        active = (
            not self.passed
            and not self.failed
            and self.current_challenge_idx < len(self.challenges)
        )
        return {
            "passed": self.passed,
            "failed": self.failed,
            "instruction": self.instruction,
            "face_detected": self.face_detected,
            "selfie_ready": self.passed,
            "calibrated": self.calibrated,
            "challenge": self.challenges[self.current_challenge_idx]
            if active
            else None,
            "challenge_idx": self.current_challenge_idx if active else -1,
            "total_challenges": len(self.challenges),
            "consecutive_progress": self.consecutive_count,
            "consecutive_needed": HYSTERESIS_FRAMES,
            "face_bbox": list(self.face_bbox) if self.face_bbox else None,
            "face_landmarks": self.landmarks_2d,
            "face_yaw": round(self.face_yaw, 1),
        }


# ═══════════════════════════════════════════════════════════════════
# Global registry + public API
# ═══════════════════════════════════════════════════════════════════

_sessions: dict[str, LivenessSession] = {}


def create_session(session_id: str, language: str = "ar") -> dict:
    if session_id in _sessions:
        cleanup_session(session_id)
    _sessions[session_id] = LivenessSession(session_id)
    return {"started": True, "session_id": session_id}


def process_liveness_frame(frame: np.ndarray, session_id: str) -> dict:
    if session_id not in _sessions:
        create_session(session_id)
    sess = _sessions[session_id]
    result = sess.process(frame)

    # Save selfie
    if not sess.passed and sess.face_detected:
        q = float(
            cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        )
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
    sess = _sessions.get(session_id)
    if not sess:
        return None
    if sess.best_frame is not None:
        return sess.best_frame
    if sess.selfie_path and os.path.exists(sess.selfie_path):
        return cv2.imread(sess.selfie_path)
    return None


def cleanup_session(session_id: str):
    sess = _sessions.pop(session_id, None)
    if sess and sess.selfie_path and os.path.exists(sess.selfie_path):
        try:
            os.remove(sess.selfie_path)
        except OSError:
            pass


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
            r = DeepFace.verify(
                img1_path=cin_path,
                img2_path=selfie_path,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=False,
            )
            d = float(r.get("distance", 1.0))
            t = float(r.get("threshold", 0.4))
            m = d < t
            s = max(0.0, 1.0 - (d / max(t * 2, 1.0)))
            return {
                "match": m,
                "score": round(s, 4),
                "threshold": round(t, 4),
                "distance": round(d, 6),
                "reason": "Match" if m else "Face mismatch",
            }
        finally:
            for p in (cin_path, selfie_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
    except Exception as e:
        logger.error("Face match error: %s", e)
        return {
            "match": False,
            "score": 0.0,
            "threshold": 0.4,
            "distance": 1.0,
            "reason": str(e),
        }
