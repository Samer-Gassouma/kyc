"""
Production-grade passive liveness detection.
Stage 1: Silent anti-spoofing (MiniFASNet dual-model fusion)
Stage 2: 3D depth map validation (MediaPipe FaceLandmarker)
Stage 3: Best frame capture for face matching
No user interaction required — fully passive/silent.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

# ── Vendored silent anti-spoofing ─────────────────────────────
VENDOR_PATH = Path(__file__).parent.parent / "vendor" / "silent_antispoofing"
sys.path.insert(0, str(VENDOR_PATH))

from src.anti_spoof_predict import AntiSpoofPredict, Detection
from src.generate_patches import CropImage
from src.utility import parse_model_name
import mediapipe as mp

logger = logging.getLogger(__name__)

# ── Model paths ────────────────────────────────────────────────
MODEL_DIR = VENDOR_PATH / "resources" / "anti_spoof_models"

ANTI_SPOOF_MODELS = [
    "2.7_80x80_MiniFASNetV2.pth",
    "4_0_0_80x80_MiniFASNetV1SE.pth",
]

# ── Thresholds ─────────────────────────────────────────────────
SPOOF_SCORE_THRESHOLD = 0.6  # above = real face
DEPTH_VARIANCE_MIN = 0.002   # min landmark depth variance for 3D confirmation
MIN_CONSECUTIVE_REAL = 5     # frames that must pass before capture
FACE_QUALITY_MIN = 40.0      # min Laplacian variance for selfie frame
MAX_TIMEOUT_SECONDS = 30     # max time for liveness check

# ── Lazy singletons ────────────────────────────────────────────
_predictor: Any | None = None
_cropper: CropImage | None = None
_landmarker: Any | None = None


def _get_predictor():
    global _predictor, _cropper
    if _predictor is None:
        logger.info("Loading MiniFASNet anti-spoofing models...")
        # Patch Detection.__init__ to use correct absolute paths
        _orig_detection_init = Detection.__init__
        _orig_predict_init = AntiSpoofPredict.__init__

        def _patched_detection_init(self):
            deploy = str(VENDOR_PATH / "resources" / "detection_model" / "deploy.prototxt")
            caffemodel = str(VENDOR_PATH / "resources" / "detection_model" / "Widerface-RetinaFace.caffemodel")
            self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
            self.detector_confidence = 0.6

        def _patched_predict_init(self, device_id):
            Detection.__init__(self)
            self.device = torch.device("cpu")

        Detection.__init__ = _patched_detection_init
        AntiSpoofPredict.__init__ = _patched_predict_init
        try:
            _predictor = AntiSpoofPredict(0)
        finally:
            Detection.__init__ = _orig_detection_init
            AntiSpoofPredict.__init__ = _orig_predict_init

        _cropper = CropImage()
        logger.info("Anti-spoofing models loaded (CPU)")
    return _predictor, _cropper


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks.python import vision, BaseOptions
        model_path = Path(__file__).parent.parent / "weights" / "face_landmarker_v2_with_blendshapes.task"
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
        logger.info("MediaPipe FaceLandmarker loaded")
    return _landmarker


# ── Stage 1: Silent anti-spoofing ──────────────────────────────
def run_anti_spoofing(frame: np.ndarray) -> dict:
    """
    Run dual MiniFASNet models on a single frame.
    Returns liveness score (0-1) and real/fake decision.
    """
    predictor, cropper = _get_predictor()
    h, w = frame.shape[:2]

    # Detect face bbox using the built-in RetinaFace detector
    image_bbox = predictor.get_bbox(frame)

    if image_bbox is None:
        return {
            "face_found": False,
            "score": 0.0,
            "is_real": False,
            "reason": "No face detected",
        }

    prediction = np.zeros((1, 3))
    model_count = 0

    for model_name in ANTI_SPOOF_MODELS:
        model_path = MODEL_DIR / model_name
        if not model_path.exists():
            logger.warning("Model not found: %s", model_path)
            continue

        h_input, w_input, model_type, scale = parse_model_name(model_name)

        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": scale is not None,
        }
        img_cropped = cropper.crop(**param)
        prediction += predictor.predict(img_cropped, str(model_path))
        model_count += 1

    if model_count == 0:
        return {
            "face_found": True,
            "score": 0.0,
            "is_real": False,
            "reason": "Model inference failed",
        }

    label = int(np.argmax(prediction))
    value = float(prediction[0][label] / model_count)
    is_real = (label == 1) and (value > SPOOF_SCORE_THRESHOLD)

    return {
        "face_found": True,
        "score": round(value, 4),
        "is_real": is_real,
        "label": label,
        "model_count": model_count,
        "bbox": image_bbox,
        "reason": "Real face" if is_real else "Spoofing detected",
    }


# ── Stage 2: 3D depth map from MediaPipe ──────────────────────
def run_depth_check(frame: np.ndarray) -> dict:
    """
    Use MediaPipe FaceLandmarker to build a 3D landmark mesh.
    A real face has high variance in Z-depth across landmarks.
    A flat photo has near-zero Z-depth variance.
    """
    try:
        landmarker = _get_landmarker()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            return {
                "face_3d": False,
                "depth_variance": 0.0,
                "is_3d": False,
                "reason": "No landmarks",
            }

        landmarks = result.face_landmarks[0]

        # Extract Z coordinates (depth) for all landmarks
        z_values = np.array([lm.z for lm in landmarks])
        depth_variance = float(np.var(z_values))

        # Compute head pose from transformation matrix
        yaw = 0.0
        pitch = 0.0
        if result.facial_transformation_matrixes:
            transform = np.array(result.facial_transformation_matrixes[0].data).reshape(4, 4)
            yaw = float(np.arctan2(transform[1, 0], transform[0, 0]) * 180 / np.pi)
            pitch = float(
                np.arctan2(
                    -transform[2, 0],
                    np.sqrt(transform[2, 1] ** 2 + transform[2, 2] ** 2),
                )
                * 180
                / np.pi
            )

        is_3d = depth_variance > DEPTH_VARIANCE_MIN

        return {
            "face_3d": True,
            "depth_variance": round(depth_variance, 6),
            "is_3d": is_3d,
            "landmark_count": len(landmarks),
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1),
            "reason": "3D face confirmed" if is_3d else "Flat face detected",
        }
    except Exception as e:
        logger.error("Depth check failed: %s", e)
        return {
            "face_3d": False,
            "depth_variance": 0.0,
            "is_3d": False,
            "reason": str(e),
        }


# ── Stage 3: Frame quality for selfie capture ─────────────────
def frame_quality(frame: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Session state ─────────────────────────────────────────────
_sessions: dict[str, dict] = {}


def reset_session(session_id: str):
    _sessions[session_id] = {
        "consecutive_real": 0,
        "frame_count": 0,
        "start_time": time.time(),
        "best_frame": None,
        "best_quality": 0.0,
        "selfie_path": None,
        "passed": False,
        "failed": False,
        "fail_reason": None,
    }


def clear_session(session_id: str) -> None:
    """Remove session data and temp selfie file."""
    session = _sessions.pop(session_id, None)
    if session:
        selfie_path = session.get("selfie_path")
        if selfie_path and os.path.exists(selfie_path):
            try:
                os.remove(selfie_path)
            except OSError:
                pass


def get_session_selfie(session_id: str) -> np.ndarray | None:
    """Return the captured selfie frame for a passed session."""
    sess = _sessions.get(session_id)
    if not sess:
        return None
    # In-memory frame may be lost on uvicorn reload; fall back to temp file
    frame = sess.get("best_frame")
    if frame is not None:
        return frame
    selfie_path = sess.get("selfie_path")
    if selfie_path and os.path.exists(selfie_path):
        frame = cv2.imread(selfie_path)
        if frame is not None:
            sess["best_frame"] = frame
            return frame
    return None


# ── Main per-frame entry point ─────────────────────────────────
def process_liveness_frame(
    frame: np.ndarray,
    session_id: str,
) -> dict[str, Any]:
    """
    Process one camera frame through the full passive liveness pipeline.
    Called per WebSocket frame. Returns state + UI instructions.
    """
    if session_id not in _sessions:
        reset_session(session_id)

    sess = _sessions[session_id]
    sess["frame_count"] += 1

    # Timeout
    elapsed = time.time() - sess["start_time"]
    if elapsed > MAX_TIMEOUT_SECONDS and not sess["passed"]:
        sess["failed"] = True
        sess["fail_reason"] = "Liveness check timed out"

    if sess["failed"]:
        return {
            "passed": False,
            "failed": True,
            "instruction": sess["fail_reason"],
            "consecutive_real": 0,
            "frame_count": sess["frame_count"],
        }

    if sess["passed"]:
        return {
            "passed": True,
            "failed": False,
            "instruction": "Liveness confirmed",
            "selfie_ready": True,
            "frame_count": sess["frame_count"],
        }

    # ── Stage 1: Anti-spoofing ─────────────────────────────────
    spoof_result = run_anti_spoofing(frame)

    if not spoof_result["face_found"]:
        sess["consecutive_real"] = 0
        return {
            "passed": False,
            "failed": False,
            "instruction": "Position your face in the frame",
            "face_detected": False,
            "consecutive_real": 0,
            "frame_count": sess["frame_count"],
        }

    if not spoof_result["is_real"]:
        sess["consecutive_real"] = 0
        return {
            "passed": False,
            "failed": False,
            "instruction": "Spoofing attempt detected — please use your real face",
            "face_detected": True,
            "spoof_score": spoof_result["score"],
            "consecutive_real": 0,
            "frame_count": sess["frame_count"],
        }

    # ── Stage 2: 3D depth check ────────────────────────────────
    depth_ok = True
    depth_result = {}
    if sess["frame_count"] % 3 == 0:
        depth_result = run_depth_check(frame)
        depth_ok = depth_result.get("is_3d", True)

    if not depth_ok:
        sess["consecutive_real"] = 0
        return {
            "passed": False,
            "failed": False,
            "instruction": "Hold your face steady and ensure good lighting",
            "face_detected": True,
            "spoof_score": spoof_result["score"],
            "depth_variance": depth_result.get("depth_variance", 0),
            "consecutive_real": 0,
            "frame_count": sess["frame_count"],
        }

    # ── Both stages passed — track consecutive frames ──────────
    sess["consecutive_real"] += 1

    # Track best quality frame for selfie
    quality = frame_quality(frame)
    if quality > sess["best_quality"]:
        sess["best_quality"] = quality
        sess["best_frame"] = frame.copy()
        # Persist to temp file so it survives uvicorn reloads
        selfie_path = sess.get("selfie_path")
        if selfie_path and os.path.exists(selfie_path):
            try:
                os.remove(selfie_path)
            except OSError:
                pass
        fd, selfie_path = tempfile.mkstemp(suffix=".jpg", prefix="selfie_")
        os.close(fd)
        cv2.imwrite(selfie_path, frame)
        sess["selfie_path"] = selfie_path

    progress = min(100, int(sess["consecutive_real"] / MIN_CONSECUTIVE_REAL * 100))

    if sess["consecutive_real"] >= MIN_CONSECUTIVE_REAL:
        sess["passed"] = True
        return {
            "passed": True,
            "failed": False,
            "instruction": "Liveness confirmed",
            "selfie_ready": True,
            "spoof_score": spoof_result["score"],
            "progress": 100,
            "frame_count": sess["frame_count"],
        }

    return {
        "passed": False,
        "failed": False,
        "instruction": "Hold still...",
        "face_detected": True,
        "spoof_score": round(spoof_result["score"], 3),
        "consecutive_real": sess["consecutive_real"],
        "progress": progress,
        "depth": depth_result,
        "frame_count": sess["frame_count"],
    }


# ── Face matching (unchanged from DeepFace) ─────────────────────
def match_faces(cin_face: np.ndarray, selfie: np.ndarray) -> dict:
    """Compare CIN face crop against the captured selfie using DeepFace."""
    try:
        from deepface import DeepFace
        import tempfile

        # Write both images to temp files for DeepFace
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
            # Convert distance to similarity score (0-1)
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
