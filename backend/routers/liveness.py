"""WebSocket /ws/liveness/{session_id} + finalize + reset endpoints."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from core.auth import get_current_user_or_api_key
from core.db import Capture, KYCResult, LivenessSession, SessionLocal
from core.storage import download_decrypted
from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from models.liveness import (
    cleanup_session,
    create_session,
    get_selfie_frame,
    match_faces,
    process_liveness_frame,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/liveness/{session_id}")
async def ws_liveness(websocket: WebSocket, session_id: str):
    """WebSocket for active liveness gesture challenges.

    Frames sent from browser at ~10fps. Server runs gesture detection
    and returns real-time instructions + final pass/fail.
    """
    await websocket.accept()
    logger.info("Liveness WebSocket connected session=%s", session_id)

    create_session(session_id)

    try:
        while True:
            data = await websocket.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json(
                    {
                        "passed": False,
                        "failed": False,
                        "instruction": "تعذر فك ترميز الإطار",
                        "face_detected": False,
                    }
                )
                continue

            result = process_liveness_frame(frame, session_id)
            await websocket.send_json(result)

            if result.get("passed") or result.get("failed"):
                break

    except WebSocketDisconnect:
        logger.info("Liveness WebSocket disconnected session=%s", session_id)
    except Exception as exc:
        logger.error("Liveness WebSocket error session=%s: %s", session_id, exc)
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass


@router.post("/api/kyc/finalize/{session_id}")
async def finalize_kyc(
    session_id: str,
    front_capture_id: str | None = Form(None),
    _user: dict = Depends(get_current_user_or_api_key),
) -> dict:
    """Run after liveness passed. Compare selfie vs CIN face crop."""
    db = SessionLocal()
    try:
        selfie = get_selfie_frame(session_id)
        if selfie is None:
            raise HTTPException(400, "Liveness not completed or selfie not captured")

        cin_face = None
        capture_id_for_kyc = None

        if front_capture_id:
            capture = db.query(Capture).filter_by(id=front_capture_id).first()
            if capture and capture.face_crop_s3_key:
                face_bytes = download_decrypted(capture.face_crop_s3_key)
                arr = np.frombuffer(face_bytes, dtype=np.uint8)
                cin_face = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                capture_id_for_kyc = front_capture_id

        if cin_face is None:
            raise HTTPException(400, "CIN face crop not found")

        match_result = match_faces(cin_face, selfie)

        ls = db.query(LivenessSession).filter_by(id=session_id).first()
        if ls:
            ls.liveness_passed = True
            ls.face_match_score = match_result["score"]
            ls.status = "completed" if match_result["match"] else "failed"
            db.commit()
            if not capture_id_for_kyc:
                capture_id_for_kyc = ls.capture_front_id

        kyc = KYCResult(
            capture_id=capture_id_for_kyc or session_id,
            side="front",
            liveness_passed=True,
            face_match_score=match_result["score"],
        )
        db.add(kyc)
        db.commit()

        cleanup_session(session_id)

        return {
            "session_id": session_id,
            "kyc_passed": match_result["match"],
            "face_match_score": match_result["score"],
            "face_match_threshold": match_result.get("threshold", 0.4),
            "reason": match_result["reason"],
        }
    finally:
        db.close()


@router.post("/api/liveness/reset/{session_id}")
async def reset_liveness(session_id: str) -> dict:
    """Reset liveness session — start fresh with new random challenges."""
    cleanup_session(session_id)
    create_session(session_id)
    return {"reset": True}
