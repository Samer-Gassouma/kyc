"""Face scan WebSocket + face match endpoints."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from core.auth import get_current_user_or_api_key
from core.db import Capture, ExtractionSession, LivenessSession, SessionLocal
from core.storage import download_decrypted
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from models.liveness import (
    FaceScanSession,
    cleanup_session,
    create_session,
    get_session,
    match_faces,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════
# WebSocket — face scan
# ═══════════════════════════════════════════════════════════════════════


@router.websocket("/ws/face-scan/{session_id}")
async def ws_face_scan(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info("Face scan WS connected: %s", session_id)

    sess = create_session(session_id)

    try:
        while True:
            data = await websocket.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json(
                    {"passed": False, "failed": False, "face_detected": False}
                )
                continue

            result = sess.process(frame)
            await websocket.send_json(result)

            if result.get("passed") or result.get("failed"):
                break
    except WebSocketDisconnect:
        logger.info("Face scan WS disconnected: %s", session_id)
    except Exception as exc:
        logger.error("Face scan WS error %s: %s", session_id, exc)
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# Selfie serving
# ═══════════════════════════════════════════════════════════════════════


@router.get("/api/liveness/selfie/{filename}")
async def get_selfie(filename: str):
    path = Path(__file__).parent.parent / "selfies" / filename
    if not path.exists():
        raise HTTPException(404, "Selfie not found")
    return FileResponse(str(path), media_type="image/jpeg")


# ═══════════════════════════════════════════════════════════════════════
# Face match — compare selfie to CIN photo
# ═══════════════════════════════════════════════════════════════════════


@router.post("/api/face-match/{session_id}")
async def face_match(
    session_id: str,
    _user: dict = Depends(get_current_user_or_api_key),
) -> dict:
    """Run face match: CIN photo vs liveness selfie. Call after face scan passes."""
    db = SessionLocal()
    try:
        ext = db.query(ExtractionSession).filter_by(id=session_id).first()
        if not ext:
            raise HTTPException(404, "Extraction session not found")

        # Get CIN face photo
        cin_face = None
        if ext.front_capture_id:
            cap = db.query(Capture).filter_by(id=ext.front_capture_id).first()
            if cap and cap.face_crop_s3_key:
                try:
                    fb = download_decrypted(cap.face_crop_s3_key)
                    cin_face = cv2.imdecode(
                        np.frombuffer(fb, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                except Exception:
                    pass
        if cin_face is None and ext.face_crop_base64:
            try:
                cin_face = cv2.imdecode(
                    np.frombuffer(
                        base64.b64decode(ext.face_crop_base64), dtype=np.uint8
                    ),
                    cv2.IMREAD_COLOR,
                )
            except Exception:
                pass

        if cin_face is None:
            raise HTTPException(400, "CIN face photo not available for this session")

        # Get selfie
        selfie_path = Path(__file__).parent.parent / "selfies" / f"{session_id}.jpg"
        if not selfie_path.exists():
            raise HTTPException(400, "Face scan selfie not found — run face scan first")

        selfie = cv2.imread(str(selfie_path))
        if selfie is None:
            raise HTTPException(400, "Failed to load selfie")

        result = match_faces(cin_face, selfie)

        # Update liveness session
        ls = db.query(LivenessSession).filter_by(id=session_id).first()
        if ls:
            ls.liveness_passed = True
            ls.face_match_score = result["score"]
            ls.status = "completed"
            db.commit()

        cleanup_session(session_id)
        return result
    finally:
        db.close()
