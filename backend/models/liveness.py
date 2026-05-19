"""
Active + passive liveness detection.

Stage 1 (passive): ONNX DeepPixBiS anti-spoofing (~20ms/frame)
  — Catches photo/video replay before gesture challenge starts.

Stage 2 (active): liveness-detector gesture challenges
  — Randomized: blink, head turn left/right, smile.
  — Runs server-side, frames sent via WebSocket.
"""

from __future__ import annotations

import logging
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
_VENDOR_DIR = Path(__file__).parent.parent / "vendor" / "face-liveness"
_SPOOF_MODEL_PATH = (
    _VENDOR_DIR / "data" / "checkpoints" / "OULU_Protocol_2_model_0_0.onnx"
)

# ── Thresholds ─────────────────────────────────────────────────────
SPOOF_THRESHOLD = 0.03  # below this = spoof (photo/video replay)
MAX_TIMEOUT_SECONDS = 45  # total time for gesture challenge

# ── ONNX anti-spoofing (lazy singleton) ────────────────────────────
_spoof_session: ort.InferenceSession | None = None

# ImageNet normalization (same as torchvision)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_spoof_model() -> ort.InferenceSession:
    global _spoof_session
    if _spoof_session is None:
        if not _SPOOF_MODEL_PATH.exists():
            _SPOOF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request

            logger.info("Downloading ONNX anti-spoofing model...")
            urllib.request.urlretrieve(
                "https://github.com/ffletcherr/face-recognition-liveness/releases/download/v0.1/OULU_Protocol_2_model_0_0.onnx",
                str(_SPOOF_MODEL_PATH),
            )
        _spoof_session = ort.InferenceSession(
            str(_SPOOF_MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        logger.info("ONNX anti-spoofing model loaded")
    return _spoof_session


def check_spoof(face_crop: np.ndarray) -> float:
    """Run DeepPixBiS anti-spoofing on a face crop.

    Returns liveness score 0-1. Higher = more real.
    Score < 0.03 strongly indicates a photo/video replay.
    """
    try:
        sess = _get_spoof_model()

        # Preprocess: BGR → RGB → resize 224×224 → normalize → NCHW
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = np.transpose(blob, (2, 0, 1))[None]  # 1×3×224×224

        output_pixel, output_binary = sess.run(
            ["output_pixel", "output_binary"],
            {"input": blob.astype(np.float32)},
        )
        score = float((np.mean(output_pixel) + np.mean(output_binary)) / 2.0)
        return score
    except Exception as exc:
        logger.error("Spoof check failed: %s", exc)
        return 1.0  # fail open


# ── Session management ─────────────────────────────────────────────
_sessions: dict[str, dict[str, Any]] = {}


def create_session(session_id: str, language: str = "ar") -> dict:
    """Start a new active liveness session with randomized gesture challenges."""
    from liveness_detector.server_launcher import GestureServerClient

    if session_id in _sessions:
        cleanup_session(session_id)

    # Randomized challenge order
    import random

    gestures = ["blink", "turnLeft", "turnRight", "smile"]
    random.shuffle(gestures)
    gestures = gestures[:3]  # pick 3

    client = GestureServerClient(
        language=language,
        socket_path=f"/tmp/liveness_{session_id}",
        num_gestures=3,
        gestures_list=gestures,
    )

    state: dict[str, Any] = {
        "client": client,
        "passed": False,
        "failed": False,
        "instruction": "انظر إلى الكاميرا",
        "best_frame": None,
        "best_quality": 0.0,
        "frame_count": 0,
        "spoof_score": 1.0,
        "start_time": time.time(),
        "selfie_path": None,
    }

    def on_instruction(message: str):
        state["instruction"] = message
        logger.info("Liveness [%s]: %s", session_id, message)

    def on_result(alive: bool):
        state["passed"] = alive
        state["failed"] = not alive
        logger.info("Liveness result [%s]: %s", session_id, alive)

    client.set_string_callback(on_instruction)
    client.set_report_alive_callback(on_result)
    client.start_server()

    _sessions[session_id] = state
    logger.info("Liveness session %s created — gestures: %s", session_id, gestures)

    return {"started": True, "session_id": session_id}


def process_liveness_frame(
    frame: np.ndarray,
    session_id: str,
) -> dict[str, Any]:
    """Process one camera frame through passive + active liveness pipeline.

    Called per WebSocket frame. Returns UI state for frontend overlay.
    """
    if session_id not in _sessions:
        create_session(session_id)

    state = _sessions[session_id]
    state["frame_count"] += 1

    # ── Timeout check ──────────────────────────────────────────
    elapsed = time.time() - state["start_time"]
    if elapsed > MAX_TIMEOUT_SECONDS and not state["passed"]:
        state["failed"] = True

    if state["failed"]:
        return {
            "passed": False,
            "failed": True,
            "instruction": "انتهت المهلة — الرجاء المحاولة مرة أخرى",
            "face_detected": False,
            "spoof_score": round(state["spoof_score"], 4),
        }

    if state["passed"]:
        return {
            "passed": True,
            "failed": False,
            "instruction": "تم التحقق",
            "selfie_ready": True,
            "spoof_score": round(state["spoof_score"], 4),
        }

    # ── Stage 1: passive anti-spoofing (ONNX, ~20ms) ────────────
    spoof_score = check_spoof(frame)
    state["spoof_score"] = spoof_score

    if spoof_score < SPOOF_THRESHOLD:
        return {
            "passed": False,
            "failed": True,
            "instruction": "يرجى استخدام وجهك الحقيقي",
            "face_detected": True,
            "spoof_detected": True,
            "spoof_score": round(spoof_score, 4),
        }

    # ── Stage 2: active gesture challenge ───────────────────────
    client = state["client"]
    client.process_frame(frame)

    # Track best frame for selfie
    if not state["passed"]:
        quality = _frame_quality(frame)
        if quality > state["best_quality"]:
            state["best_quality"] = quality
            state["best_frame"] = frame.copy()
            # Persist to temp file (survives uvicorn reload)
            old = state.get("selfie_path")
            if old and os.path.exists(old):
                try:
                    os.remove(old)
                except OSError:
                    pass
            fd, path = tempfile.mkstemp(suffix=".jpg", prefix="selfie_")
            os.close(fd)
            cv2.imwrite(path, frame)
            state["selfie_path"] = path

    return {
        "passed": state["passed"],
        "failed": state["failed"],
        "instruction": state["instruction"],
        "face_detected": True,
        "spoof_score": round(spoof_score, 4),
    }


def get_selfie_frame(session_id: str) -> np.ndarray | None:
    """Return best captured selfie frame for face matching."""
    state = _sessions.get(session_id)
    if not state:
        return None
    frame = state.get("best_frame")
    if frame is not None:
        return frame
    path = state.get("selfie_path")
    if path and os.path.exists(path):
        frame = cv2.imread(path)
        if frame is not None:
            state["best_frame"] = frame
            return frame
    return None


def cleanup_session(session_id: str) -> None:
    """Stop gesture server, remove temp files, clear state."""
    state = _sessions.pop(session_id, None)
    if state is None:
        return
    try:
        state["client"].stop_server()
    except Exception as exc:
        logger.warning("Error stopping liveness server: %s", exc)
    path = state.get("selfie_path")
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _frame_quality(frame: np.ndarray) -> float:
    """Laplacian variance — higher = sharper face."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Face matching (unchanged) ──────────────────────────────────────
def match_faces(cin_face: np.ndarray, selfie: np.ndarray) -> dict:
    """Compare CIN face crop against the captured selfie using DeepFace."""
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
            "reason": f"Face matching error: {exc}",
        }
