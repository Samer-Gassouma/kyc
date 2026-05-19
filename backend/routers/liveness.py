"""WebSocket + unified KYC endpoints: start, liveness, finalize, status, selfie."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from core.auth import get_current_user_or_api_key
from core.db import Capture, ExtractionSession, KYCResult, LivenessSession, SessionLocal
from core.storage import download_decrypted
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from models.liveness import (
    cleanup_session,
    create_session,
    get_selfie_frame,
    match_faces,
    process_liveness_frame,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ═══════════════════════════════════════════════════════════════════
# Step 1: Start KYC — upload CIN images + begin extraction
# ═══════════════════════════════════════════════════════════════════


@router.post("/api/kyc/start")
async def kyc_start(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    _user: dict = Depends(get_current_user_or_api_key),
) -> dict:
    """Upload front+back CIN images. Returns session_id.

    CIN extraction runs in background via Celery.
    Use /ws/liveness/{session_id} for the liveness check.
    Poll /api/kyc/status/{session_id} for combined results.
    """
    import asyncio

    front_bytes = await front.read()
    back_bytes = await back.read()

    db = SessionLocal()
    try:
        # Create extraction session
        ext = ExtractionSession(status="processing")
        db.add(ext)
        db.commit()
        db.refresh(ext)
        session_id = ext.id

        # Also create liveness session placeholder
        ls = LivenessSession(id=session_id)
        db.add(ls)
        db.commit()
    finally:
        db.close()

    # Fire background CIN extraction (same as extract/start)
    from routers.extract import _prepare_and_dispatch

    async def _extract_background():
        db2 = SessionLocal()
        try:
            sess = db2.query(ExtractionSession).filter_by(id=session_id).first()
            if not sess:
                return

            loop = asyncio.get_event_loop()
            front_id, back_id = await asyncio.gather(
                loop.run_in_executor(None, _prepare_and_dispatch, front_bytes, "front"),
                loop.run_in_executor(None, _prepare_and_dispatch, back_bytes, "back"),
            )

            sess.front_capture_id = front_id
            sess.back_capture_id = back_id
            db2.commit()

            # Update liveness session with capture IDs
            ls2 = db2.query(LivenessSession).filter_by(id=session_id).first()
            if ls2:
                ls2.capture_front_id = front_id
                ls2.capture_back_id = back_id
                db2.commit()

            logger.info(
                "KYC session %s: CIN dispatched front=%s back=%s",
                session_id,
                front_id,
                back_id,
            )

            # Wait for CIN extraction to complete
            from routers.extract import _wait_for_captures

            await _wait_for_captures(front_id, back_id, timeout_s=300)

            # Merge results
            from core.db import KYCResult as KYCRes

            front_result = db2.query(KYCRes).filter_by(capture_id=front_id).first()
            back_result = db2.query(KYCRes).filter_by(capture_id=back_id).first()

            merged = {}
            face_b64 = None
            if front_result and front_result.ocr_fields:
                try:
                    fp = json.loads(front_result.ocr_fields)
                    merged.update(fp.get("roi_fields", {}))
                    face_b64 = fp.get("face_crop_base64")
                except json.JSONDecodeError:
                    pass
            if back_result and back_result.ocr_fields:
                try:
                    bp = json.loads(back_result.ocr_fields)
                    merged.update(bp.get("roi_fields", {}))
                except json.JSONDecodeError:
                    pass

            sess.merged_fields = json.dumps(merged)
            sess.face_crop_base64 = face_b64
            sess.status = "completed"
            db2.commit()
            logger.info("KYC session %s: CIN extraction completed", session_id)
        except Exception as e:
            logger.error("KYC session %s background error: %s", session_id, e)
            try:
                sess2 = db2.query(ExtractionSession).filter_by(id=session_id).first()
                if sess2:
                    sess2.status = "failed"
                    sess2.error_reason = str(e)
                    db2.commit()
            except Exception:
                pass
        finally:
            db2.close()

    asyncio.create_task(_extract_background())

    return {
        "session_id": session_id,
        "status": "processing",
        "message": "CIN extraction started. Open liveness WebSocket to continue.",
    }


# ═══════════════════════════════════════════════════════════════════
# Step 2: Liveness WebSocket
# ═══════════════════════════════════════════════════════════════════


@router.websocket("/ws/liveness/{session_id}")
async def ws_liveness(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info("Liveness WS: %s", session_id)
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
        logger.info("Liveness WS disconnect: %s", session_id)
    except Exception as exc:
        logger.error("Liveness WS error %s: %s", session_id, exc)
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# Step 3: Finalize — run face match when liveness passes
# ═══════════════════════════════════════════════════════════════════


@router.post("/api/kyc/finalize/{session_id}")
async def kyc_finalize(
    session_id: str,
    _user: dict = Depends(get_current_user_or_api_key),
) -> dict:
    """Called after liveness passes. Runs face match against CIN photo.
    Returns only when CIN extraction is also complete."""
    db = SessionLocal()
    try:
        selfie = get_selfie_frame(session_id)
        if selfie is None:
            raise HTTPException(400, "Liveness not completed")

        # Check CIN extraction status
        ext = db.query(ExtractionSession).filter_by(id=session_id).first()
        if not ext:
            raise HTTPException(404, "KYC session not found")

        if ext.status == "failed":
            raise HTTPException(400, f"CIN extraction failed: {ext.error_reason}")

        if ext.status != "completed":
            return {
                "session_id": session_id,
                "kyc_passed": False,
                "status": "processing",
                "message": "CIN extraction still in progress. Poll /api/kyc/status for results.",
            }

        # CIN extraction done — get face crop
        cin_face = None
        if ext.front_capture_id:
            cap = db.query(Capture).filter_by(id=ext.front_capture_id).first()
            if cap and cap.face_crop_s3_key:
                try:
                    face_bytes = download_decrypted(cap.face_crop_s3_key)
                    arr = np.frombuffer(face_bytes, dtype=np.uint8)
                    cin_face = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                except Exception:
                    pass

        # Fallback: try base64 from merged fields
        if cin_face is None and ext.face_crop_base64:
            import base64

            try:
                cin_bytes = base64.b64decode(ext.face_crop_base64)
                arr = np.frombuffer(cin_bytes, dtype=np.uint8)
                cin_face = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                pass

        if cin_face is None:
            # No face to match — pass liveness-only if CIN extraction worked
            merged = {}
            if ext.merged_fields:
                try:
                    merged = json.loads(ext.merged_fields)
                except json.JSONDecodeError:
                    pass
            cleanup_session(session_id)
            return {
                "session_id": session_id,
                "kyc_passed": True,
                "face_match_score": None,
                "face_match_possible": False,
                "cin_fields": merged,
                "reason": "Liveness passed, CIN extracted, face match skipped (no photo found)",
            }

        match_result = match_faces(cin_face, selfie)

        # Update DB
        ls = db.query(LivenessSession).filter_by(id=session_id).first()
        if ls:
            ls.liveness_passed = True
            ls.face_match_score = match_result["score"]
            ls.status = "completed" if match_result["match"] else "failed"
            db.commit()

        merged = {}
        if ext.merged_fields:
            try:
                merged = json.loads(ext.merged_fields)
            except json.JSONDecodeError:
                pass

        cleanup_session(session_id)

        return {
            "session_id": session_id,
            "kyc_passed": match_result["match"],
            "face_match_score": match_result["score"],
            "face_match_threshold": match_result.get("threshold", 0.4),
            "cin_fields": merged,
            "reason": match_result["reason"],
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════
# Status — poll for combined KYC results
# ═══════════════════════════════════════════════════════════════════


@router.get("/api/kyc/status/{session_id}")
async def kyc_status(
    session_id: str,
    _user: dict = Depends(get_current_user_or_api_key),
) -> dict:
    """Get combined KYC status: CIN extraction + liveness state."""
    db = SessionLocal()
    try:
        ext = db.query(ExtractionSession).filter_by(id=session_id).first()
        ls = db.query(LivenessSession).filter_by(id=session_id).first()

        cin_status = ext.status if ext else "unknown"
        cin_fields = {}
        if ext and ext.merged_fields:
            try:
                cin_fields = json.loads(ext.merged_fields)
            except json.JSONDecodeError:
                pass

        liveness_passed = ls.liveness_passed if ls else False
        face_match_score = ls.face_match_score if ls else None

        kyc_passed = cin_status == "completed" and liveness_passed

        return {
            "session_id": session_id,
            "cin_status": cin_status,
            "cin_fields": cin_fields,
            "liveness_passed": liveness_passed,
            "face_match_score": face_match_score,
            "kyc_passed": kyc_passed,
            "cin_error": ext.error_reason if ext and ext.status == "failed" else None,
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════
# Selfie serving
# ═══════════════════════════════════════════════════════════════════


@router.get("/api/liveness/selfie/{filename}")
async def get_selfie(filename: str):
    path = Path(__file__).parent.parent / "selfies" / filename
    if not path.exists():
        raise HTTPException(404, "Selfie not found")
    return FileResponse(str(path), media_type="image/jpeg")


# ═══════════════════════════════════════════════════════════════════
# Reset
# ═══════════════════════════════════════════════════════════════════


@router.post("/api/liveness/reset/{session_id}")
async def reset_liveness(session_id: str) -> dict:
    cleanup_session(session_id)
    create_session(session_id)
    return {"reset": True}
