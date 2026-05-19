"""
Passive ONNX liveness — works in low light, shitty cameras.

Auto-brightness + CLAHE on every frame.
RetinaFace DNN → Haar cascade fallback for face detection.
Faceplugin fr_liveness.onnx for real/fake check.
"""

from __future__ import annotations

import logging, os, tempfile, time
from pathlib import Path
from typing import Any
import cv2, numpy as np, onnxruntime as ort

logger = logging.getLogger(__name__)
_MODEL_DIR = Path(__file__).parent.parent.parent / "id-capture" / "public" / "models"
MIN_FACE_SCORE = 0.7    # relaxed for low light
LIVENESS_SCORE_MIN = 0.5  # relaxed for low light
NEEDED_FRAMES = 12        # faster pass
TIMEOUT_SEC = 45           # longer timeout

_sessions_cache: dict[str, ort.InferenceSession] = {}

def _enhance(frame: np.ndarray) -> np.ndarray:
    """Auto-brightness + CLAHE — makes low-light frames usable."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Auto brightness: scale so mean ≈ 127
    mean_val = gray.mean()
    if mean_val < 60 or mean_val > 190:
        alpha = 127.0 / max(mean_val, 1.0)
        frame = cv2.convertScaleAbs(frame, alpha=min(alpha, 2.5), beta=0)
    # CLAHE on L channel of LAB for contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def _get_session(name: str) -> ort.InferenceSession:
    if name not in _sessions_cache:
        path = _MODEL_DIR / f"{name}.onnx"
        _sessions_cache[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _sessions_cache[name]

_dnn_face_net: Any = None

def _detect_face_dnn(frame: np.ndarray) -> tuple | None:
    global _dnn_face_net
    h, w = frame.shape[:2]
    if _dnn_face_net is None:
        v = Path(__file__).parent.parent / "vendor" / "silent_antispoofing" / "resources" / "detection_model"
        _dnn_face_net = cv2.dnn.readNetFromCaffe(str(v / "deploy.prototxt"), str(v / "Widerface-RetinaFace.caffemodel"))
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


def _detect_face(frame: np.ndarray) -> tuple | None:
    """DNN detection with size sanity check."""
    result = _detect_face_dnn(frame)
    if result is None:
        return None
    x1, y1, x2, y2, score = result
    h, w = frame.shape[:2]
    fw, fh = x2 - x1, y2 - y1
    # Face must be reasonable size: >60px and <85% of frame
    if fw < 60 or fh < 60 or fw > w * 0.85 or fh > h * 0.85:
        return None
    return result

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
        self.last_bbox: tuple | None = None  # for stability check
        self.landmarks_2d: list | None = None
        self.liveness_score = 0.0
        # Phase 1: face presence confirmation
        self.face_presence_count = 0
        self.face_confirmed = False
        self.face_confirm_needed = 12  # frames with stable face before starting
        # Phase 2: liveness verification
        self.real_frames = 0
        self.landmark_history: list = []  # last 3 frames for smoothing
        self.best_frame: np.ndarray | None = None
        self.best_quality = 0.0
        self.selfie_path: str | None = None
        self.no_face_count = 0

    def process(self, frame: np.ndarray) -> dict:
        if self.passed or self.failed:
            return self._response()
        if time.time() - self.start_time > TIMEOUT_SEC:
            self.failed = True
            self.instruction = "انتهت المهلة — حاول مجدداً"
            return self._response()

        enhanced = _enhance(frame)
        det = _detect_face(enhanced) or _detect_face(frame)

        # ── No face ─────────────────────────────────────────
        if det is None:
            self.face_detected = False
            self.no_face_count += 1
            self.face_presence_count = max(0, self.face_presence_count - 1)
            self.real_frames = max(0, self.real_frames - 1)
            if self.no_face_count > 40:
                self.instruction = "تأكد من وجود إضاءة كافية"
            else:
                self.instruction = "ضع وجهك في الإطار"
            return self._response()

        self.no_face_count = 0
        x1, y1, x2, y2, score = det
        fw, fh = x2 - x1, y2 - y1
        cx, cy = x1 + fw // 2, y1 + fh // 2

        # ── Phase 1: Face presence confirmation ─────────────
        if not self.face_confirmed:
            # Check bbox stability — center shouldn't jump >20% of face size
            if self.last_bbox is not None:
                lx, ly, lfw, lfh = self.last_bbox
                lcx, lcy = lx + lfw // 2, ly + lfh // 2
                jump = abs(cx - lcx) / max(fw, 1) + abs(cy - lcy) / max(fh, 1)
                if jump < 0.15:  # stable
                    self.face_presence_count += 1
                else:
                    self.face_presence_count = max(0, self.face_presence_count - 1)
            else:
                self.face_presence_count += 1

            self.last_bbox = (x1, y1, fw, fh)
            self.face_detected = True
            self.face_bbox = (int(x1), int(y1), int(fw), int(fh))

            if self.face_presence_count >= self.face_confirm_needed:
                self.face_confirmed = True
                self.instruction = "جاري التحقق..."
            else:
                pct = int(self.face_presence_count / self.face_confirm_needed * 100)
                self.instruction = f"ثابت... جاري التعرف ({pct}%)"
            return self._response()

        # ── Phase 2: Liveness verification ──────────────────
        self.face_detected = True
        self.face_bbox = (int(x1), int(y1), int(fw), int(fh))

        lm = _get_landmarks(enhanced, (x1, y1, x2, y2))
        if lm and len(lm) >= 20:
            self.landmarks_2d = [{"x": round(p[0], 1), "y": round(p[1], 1)} for p in lm]

        liveness = _check_liveness(enhanced, (x1, y1, x2, y2))
        self.liveness_score = liveness

        if liveness >= LIVENESS_SCORE_MIN:
            self.real_frames += 1
            remaining = NEEDED_FRAMES - self.real_frames
            if remaining <= 0:
                self.passed = True
                self.instruction = "تم التحقق ✔"
                try:
                    selfie_dir = Path(__file__).parent.parent / "selfies"
                    selfie_dir.mkdir(exist_ok=True)
                    spath = selfie_dir / f"{self.session_id}.jpg"
                    if self.best_frame is not None:
                        cv2.imwrite(str(spath), self.best_frame)
                        self.selfie_path = str(spath)
                except Exception as e:
                    logger.warning("Selfie save failed: %s", e)
            else:
                self.instruction = f"تم التحقق... {self.real_frames}/{NEEDED_FRAMES}"
        else:
            self.real_frames = max(0, self.real_frames - 1)
            self.instruction = "انظر مباشرة إلى الكاميرا"

        return self._response()

    def _response(self) -> dict:
        return {
            "passed": self.passed, "failed": self.failed,
            "instruction": self.instruction, "face_detected": self.face_detected,
            "selfie_ready": self.passed,
            "selfie_url": f"/api/liveness/selfie/{self.session_id}.jpg" if self.passed else None,
            "liveness_score": round(self.liveness_score, 3),
            "progress": min(100, int(self.real_frames / NEEDED_FRAMES * 100)),
            "face_bbox": [int(v) for v in self.face_bbox] if self.face_bbox else None,
            "face_landmarks": self.landmarks_2d,
            "face_landmarks_full": True,
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
