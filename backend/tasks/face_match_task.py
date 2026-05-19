"""Celery task: Face matching between ID photo and liveness frame."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from models.liveness import compare_faces

logger = logging.getLogger(__name__)


def process_face_match(
    id_face_bytes: bytes,
    live_face_bytes: bytes,
    session_id: str,
) -> dict:
    """Compare face from ID document to live capture.

    Called by Celery worker.
    """
    id_arr = np.frombuffer(id_face_bytes, dtype=np.uint8)
    id_face = cv2.imdecode(id_arr, cv2.IMREAD_COLOR)

    live_arr = np.frombuffer(live_face_bytes, dtype=np.uint8)
    live_face = cv2.imdecode(live_arr, cv2.IMREAD_COLOR)

    if id_face is None or live_face is None:
        return {"session_id": session_id, "match": False, "error": "Failed to decode images"}

    result = compare_faces(id_face, live_face)
    result["session_id"] = session_id

    # Update DB
    from core.db import SessionLocal, LivenessSession

    db = SessionLocal()
    try:
        session = db.query(LivenessSession).filter_by(id=session_id).first()
        if session:
            session.face_match_score = result["score"]
            session.liveness_passed = result["match"]
            session.status = "completed"
            db.commit()
    finally:
        db.close()

    return result
