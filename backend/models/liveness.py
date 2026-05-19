"""
Passive ONNX liveness — no gestures, just look at camera for 1.5s.

Face detection: OpenCV DNN RetinaFace
Liveness: Faceplugin fr_liveness.onnx (score > 0.3 = real)
"""

from __future__ import annotations

import logging, os, tempfile, time
from pathlib import Path
from typing import Any
import cv2, numpy as np, onnxruntime as ort

logger = logging.getLogger(__name__)
_MODEL_DIR = Path(__file__).parent.parent.parent / "id-capture" / "public" / "models"
MIN_FACE_SCORE = 0.5
LIVENESS_SCORE_MIN = 0.3
NEEDED_FRAMES = 15

_sessions_cache: dict[str, ort.InferenceSession] = {}

def _get_session(name: str) -> ort.InferenceSession:
    if name not in _sessions_cache:
        path = _MODEL_DIR / f"{name}.onnx"
        _sessions_cache[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        logger.info("ONNX %s loaded", name)
    return _sessions_cache[name]

_dnn_face_net: Any = None

def _detect_face(frame: np.ndarray) -> tuple | None:
    global _dnn_face_net
    h, w = frame.shape[:2]
    if _dnn_face_net is None:
        v = Path(__file__).parent.parent / "vendor" / "silent_antispoofing" / "resources" / "detection_model"
        _dnn_face_net = cv2.dnn.readNetFromCaffe(str(v / "deploy.prototxt"), str(v / "Widerface-RetinaFace.caffemodel"))
        logger.info("RetinaFace DNN loaded")
    blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 240), (104, 117, 123))
    _dnn_face_net.setInput(blob)
    dets = _dnn_face_net.forward()
    best_score, best_box = 0.0, None
    for i in range(dets.shape[2]):
        conf = dets[0, 0, i, 2]
        if conf > best_score:
            best_score = conf
            best_box = (int(dets[0,0,i,3]*w), int(dets[0,0,i,4]*h), int(dets[0,0,i,5]*w), int(dets[0,0,i,6]*h))
    if best_score < MIN_FACE_SCORE or best_box is None:
        return None
    return (*best_box, best_score)

def _check_liveness(frame: np.ndarray, bbox: tuple) -> float:
    sess = _get_session("fr_liveness")
    x1, y1, x2, y2 = bbox[:4]
    face = frame[y1:y2, x1:x2]
    if face.size == 0: return 0.0
    rgb = cv2.cvtColor(cv2.resize(face, (128, 128)), cv2.COLOR_BGR2RGB)
    blob = ((rgb.astype(np.float32) - 127.0) / 128.0).transpose(2, 0, 1)[None]
    result = sess.run(None, {"input": blob})[0][0]
    exp = np.exp(result - np.max(result))
    return float(exp[1] / exp.sum())

def _get_landmarks(frame: np.ndarray, bbox: tuple) -> list | None:
    sess = _get_session("fr_landmark")
    x1, y1, x2, y2 = bbox[:4]
    face = frame[y1:y2, x1:x2]
    if face.size == 0: return None
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    blob = ((cv2.resize(gray, (64, 64)).astype(np.float32) - 127.0) / 128.0)[None, None]
    result = sess.run(None, {"input": blob})[0][0]
    fw, fh = x2 - x1, y2 - y1
    return [(float(x1 + result[i] * fw), float(y1 + result[i + 1] * fh)) for i in range(0, 136, 2)]

class LivenessSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.passed = False
        self.failed = False
        self.instruction = "انظر إلى الكاميرا"
        self.face_detected = False
        self.face_bbox: tuple | None = None
        self.landmarks_2d: list | None = None
        self.liveness_score = 0.0
        self.real_frames = 0
        self.best_frame: np.ndarray | None = None
        self.best_quality = 0.0
        self.selfie_path: str | None = None
        logger.info("Liveness session %s started", session_id)

    def process(self, frame: np.ndarray) -> dict:
        if self.passed or self.failed:
            return self._response()
        if time.time() - self.start_time > 30:
            self.failed = True
            self.instruction = "انتهت المهلة"
            return self._response()

        det = _detect_face(frame)
        if det is None:
            self.face_detected = False
            self.real_frames = 0
            self.instruction = "ضع وجهك في الإطار"
            return self._response()

        x1, y1, x2, y2, score = det
        self.face_detected = True
        self.face_bbox = (x1, y1, x2 - x1, y2 - y1)

        lm = _get_landmarks(frame, (x1, y1, x2, y2))
        if lm and len(lm) >= 20:
            self.landmarks_2d = [{"x": round(p[0], 1), "y": round(p[1], 1)} for p in lm[:20]]

        liveness = _check_liveness(frame, (x1, y1, x2, y2))
        self.liveness_score = liveness

        if liveness >= LIVENESS_SCORE_MIN:
            self.real_frames += 1
            self.instruction = f"استمر... ({self.real_frames}/{NEEDED_FRAMES})"
            if self.real_frames >= NEEDED_FRAMES:
                self.passed = True
                self.instruction = "تم التحقق ✔"
        else:
            self.real_frames = max(0, self.real_frames - 1)
            self.instruction = "انظر مباشرة إلى الكاميرا"
        return self._response()

    def _response(self) -> dict:
        return {
            "passed": self.passed, "failed": self.failed,
            "instruction": self.instruction, "face_detected": self.face_detected,
            "selfie_ready": self.passed,
            "liveness_score": round(self.liveness_score, 3),
            "progress": min(100, int(self.real_frames / NEEDED_FRAMES * 100)),
            "face_bbox": list(self.face_bbox) if self.face_bbox else None,
            "face_landmarks": self.landmarks_2d,
        }

_sessions: dict[str, LivenessSession] = {}

def create_session(session_id: str, language: str = "ar") -> dict:
    if session_id in _sessions: cleanup_session(session_id)
    _sessions[session_id] = LivenessSession(session_id)
    return {"started": True, "session_id": session_id}

def process_liveness_frame(frame: np.ndarray, session_id: str) -> dict:
    if session_id not in _sessions: create_session(session_id)
    sess = _sessions[session_id]
    result = sess.process(frame)
    if not sess.passed and sess.face_detected:
        q = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if q > sess.best_quality:
            sess.best_quality = q
            sess.best_frame = frame.copy()
            old = sess.selfie_path
            if old and os.path.exists(old):
                try: os.remove(old)
                except OSError: pass
            fd, path = tempfile.mkstemp(suffix=".jpg", prefix="selfie_")
            os.close(fd); cv2.imwrite(path, frame); sess.selfie_path = path
    return result

def get_selfie_frame(session_id: str) -> np.ndarray | None:
    sess = _sessions.get(session_id)
    if not sess: return None
    if sess.best_frame is not None: return sess.best_frame
    if sess.selfie_path and os.path.exists(sess.selfie_path): return cv2.imread(sess.selfie_path)
    return None

def cleanup_session(session_id: str):
    sess = _sessions.pop(session_id, None)
    if sess and sess.selfie_path and os.path.exists(sess.selfie_path):
        try: os.remove(sess.selfie_path)
        except OSError: pass

def match_faces(cin_face: np.ndarray, selfie: np.ndarray) -> dict:
    try:
        from deepface import DeepFace
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            cv2.imwrite(f1.name, cin_face); cin_path = f1.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            cv2.imwrite(f2.name, selfie); selfie_path = f2.name
        try:
            r = DeepFace.verify(img1_path=cin_path, img2_path=selfie_path,
                                model_name="Facenet", detector_backend="opencv", enforce_detection=False)
            d = float(r.get("distance", 1.0)); t = float(r.get("threshold", 0.4))
            m = d < t; s = max(0.0, 1.0 - (d / max(t * 2, 1.0)))
            return {"match": m, "score": round(s, 4), "threshold": round(t, 4),
                    "distance": round(d, 6), "reason": "Match" if m else "Face mismatch"}
        finally:
            for p in (cin_path, selfie_path):
                try: os.remove(p)
                except OSError: pass
    except Exception as e:
        logger.error("Face match error: %s", e)
        return {"match": False, "score": 0.0, "threshold": 0.4, "distance": 1.0, "reason": str(e)}
