"""Face detection, liveness check, and face matching using ONNX models + DeepFace."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent.parent / "id-capture" / "public" / "models"
SELFIE_DIR = Path(__file__).parent.parent / "selfies"

MIN_FACE_SCORE = 0.5
LIVENESS_THRESHOLD = 0.5
NEEDED_REAL_FRAMES = 12
TIMEOUT_SEC = 60

_sessions: dict[str, ort.InferenceSession] = {}


def _get_session(name: str) -> ort.InferenceSession:
    if name not in _sessions:
        path = _MODEL_DIR / f"{name}.onnx"
        _sessions[name] = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
    return _sessions[name]


# ── Face detection ──────────────────────────────────────────────────


def _detect_face_onnx(frame: np.ndarray) -> tuple | None:
    """Detect face using fr_detect.onnx. Returns (x1, y1, x2, y2, score) or None."""
    try:
        sess = _get_session("fr_detect")
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Try common input formats
        inp = sess.get_inputs()[0]
        in_name = inp.name
        in_shape = inp.shape

        if len(in_shape) == 4:
            tgt_h, tgt_w = in_shape[2], in_shape[3]
        else:
            tgt_h, tgt_w = 640, 640

        resized = cv2.resize(rgb, (tgt_w, tgt_h))
        blob = resized.astype(np.float32).transpose(2, 0, 1)[None]

        outputs = sess.run(None, {in_name: blob})

        # Try to parse outputs — common formats
        boxes = None
        scores = None
        for out in outputs:
            arr = np.array(out)
            if arr.ndim == 3 and arr.shape[2] == 4 and arr.shape[0] == 1:
                boxes = arr[0]  # (N, 4)
            elif arr.ndim == 2 and arr.shape[1] == 4 and boxes is None:
                boxes = arr  # (N, 4)
            elif arr.ndim == 2 and arr.shape[1] == 1:
                scores = arr.flatten()
            elif arr.ndim == 1 and len(arr) > 0 and 0 < float(arr[0]) <= 1:
                scores = arr

        if boxes is None or len(boxes) == 0:
            return None

        if scores is None:
            scores = np.ones(len(boxes)) * 0.8

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < MIN_FACE_SCORE:
            return None

        bx = boxes[best_idx]
        if len(bx) >= 4:
            x1 = int(bx[0] / tgt_w * w)
            y1 = int(bx[1] / tgt_h * h)
            x2 = int(bx[2] / tgt_w * w)
            y2 = int(bx[3] / tgt_h * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            fw, fh = x2 - x1, y2 - y1
            if fw < 40 or fh < 40 or fw > w * 0.9 or fh > h * 0.9:
                return None
            return (x1, y1, x2, y2, best_score)
        return None
    except Exception as exc:
        logger.debug("ONNX face detection failed: %s", exc)
        return None


def _detect_face_haar(frame: np.ndarray) -> tuple | None:
    """OpenCV Haar cascade fallback."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, fw, fh = faces[0]
    return (x, y, x + fw, y + fh, 0.7)


def detect_face(frame: np.ndarray) -> tuple | None:
    """Detect face: ONNX first, Haar fallback."""
    result = _detect_face_onnx(frame)
    if result is not None:
        return result
    return _detect_face_haar(frame)


# ── Liveness check ──────────────────────────────────────────────────


def check_liveness(frame: np.ndarray, bbox: tuple) -> float:
    """Run liveness classification on face crop. Returns real probability 0..1."""
    x1, y1, x2, y2, _ = bbox
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return 0.0
    try:
        sess = _get_session("fr_liveness")
        rgb = cv2.cvtColor(cv2.resize(face, (128, 128)), cv2.COLOR_BGR2RGB)
        blob = ((rgb.astype(np.float32) - 127.0) / 128.0).transpose(2, 0, 1)[None]
        result = sess.run(None, {"input": blob})[0][0]
        exp = np.exp(result - np.max(result))
        return float(exp[1] / exp.sum())
    except Exception as exc:
        logger.warning("Liveness check failed: %s", exc)
        return 0.0


# ── Session ─────────────────────────────────────────────────────────


class FaceScanSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.passed = False
        self.failed = False
        self.face_detected = False
        self.face_bbox: list | None = None
        self.liveness_score = 0.0
        self.real_frames = 0
        self.no_face_count = 0
        self.best_frame: np.ndarray | None = None
        self.best_quality = 0.0
        self.selfie_path: str | None = None

    def process(self, frame: np.ndarray) -> dict:
        if self.passed or self.failed:
            return self._response()
        if time.time() - self.start_time > TIMEOUT_SEC:
            self.failed = True
            return self._response()

        det = detect_face(frame)

        if det is None:
            self.face_detected = False
            self.no_face_count += 1
            self.real_frames = max(0, self.real_frames - 1)
            self.face_bbox = None
            return self._response()

        self.no_face_count = 0
        x1, y1, x2, y2, score = det
        self.face_detected = True
        self.face_bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

        liveness = check_liveness(frame, det)
        self.liveness_score = liveness

        # Track best frame by sharpness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quality = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if quality > self.best_quality:
            self.best_quality = quality
            self.best_frame = frame.copy()

        if liveness >= LIVENESS_THRESHOLD:
            self.real_frames += 1
            if self.real_frames >= NEEDED_REAL_FRAMES:
                self.passed = True
                self._save_selfie()
        else:
            self.real_frames = max(0, self.real_frames - 1)

        return self._response()

    def _save_selfie(self):
        SELFIE_DIR.mkdir(exist_ok=True)
        path = SELFIE_DIR / f"{self.session_id}.jpg"
        if self.best_frame is not None:
            cv2.imwrite(str(path), self.best_frame)
            self.selfie_path = str(path)

    def _response(self) -> dict:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "face_detected": self.face_detected,
            "liveness_score": round(self.liveness_score, 3),
            "progress": min(100, int(self.real_frames / NEEDED_REAL_FRAMES * 100)),
            "face_bbox": self.face_bbox,
        }


_sessions_store: dict[str, FaceScanSession] = {}


def create_session(session_id: str) -> FaceScanSession:
    sess = FaceScanSession(session_id)
    _sessions_store[session_id] = sess
    return sess


def get_session(session_id: str) -> FaceScanSession | None:
    return _sessions_store.get(session_id)


def cleanup_session(session_id: str):
    _sessions_store.pop(session_id, None)


# ── Face match ──────────────────────────────────────────────────────


def match_faces(cin_face: np.ndarray, selfie: np.ndarray) -> dict:
    """Compare CIN face photo to liveness selfie using DeepFace Facenet."""
    try:
        from deepface import DeepFace
    except ImportError:
        logger.warning("DeepFace not installed — face match skipped")
        return {
            "match": True,
            "score": 1.0,
            "threshold": 0.4,
            "reason": "Face match skipped (DeepFace not installed)",
        }

    import tempfile

    cin_path = None
    selfie_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, cin_face)
            cin_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, selfie)
            selfie_path = f.name

        result = DeepFace.verify(
            img1_path=cin_path,
            img2_path=selfie_path,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False,
        )
        distance = float(result.get("distance", 1.0))
        threshold = float(result.get("threshold", 0.4))
        match = distance < threshold
        score = max(0.0, 1.0 - distance / max(threshold * 2, 0.01))

        return {
            "match": match,
            "score": round(score, 4),
            "threshold": round(threshold, 4),
            "distance": round(distance, 6),
        }
    except Exception as exc:
        logger.error("Face match error: %s", exc)
        return {"match": False, "score": 0.0, "threshold": 0.4, "reason": str(exc)}
    finally:
        for p in (cin_path, selfie_path):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass
