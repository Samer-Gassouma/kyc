"""Extraction session router — async front+back ID card processing."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

import cv2
import numpy as np
from core.auth import get_current_user_or_api_key
from core.config import settings
from core.db import Capture, ExtractionSession, KYCResult, SessionLocal
from core.storage import upload_encrypted
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from models.yolo_detector import detect_frame

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/extract", tags=["extract"])

# ── Redis Pub/Sub wait helper ──────────────────────────────────────


async def _wait_for_captures(
    front_id: str,
    back_id: str,
    timeout_s: float = 300,
) -> bool:
    """Wait for both Celery workers to finish via Redis Pub/Sub.

    Returns True when both captures are done, False on timeout.
    Falls back to polling if Redis is unavailable.
    """
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.redis_url)
        pubsub = r.pubsub()
        channels = [f"kyc:capture:{front_id}:done", f"kyc:capture:{back_id}:done"]
        await pubsub.subscribe(*channels)

        done: set[str] = set()

        async def _listen():
            async for message in pubsub.listen():
                if message["type"] == "message":
                    ch = message["channel"].decode()
                    if ch.startswith("kyc:capture:") and ch.endswith(":done"):
                        capture_id = ch.split(":")[2]
                        done.add(capture_id)
                        logger.info("Redis Pub/Sub: capture %s done", capture_id)

        listen_task = asyncio.create_task(_listen())

        try:
            async with asyncio.timeout(timeout_s):
                while front_id not in done or back_id not in done:
                    await asyncio.sleep(0.1)
        except asyncio.TimeoutError:
            logger.warning("Redis Pub/Sub timed out after %.0fs", timeout_s)
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass
            await pubsub.close()
            await r.close()
            return False

        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass
        await pubsub.close()
        await r.close()
        return True

    except Exception as exc:
        logger.warning("Redis Pub/Sub unavailable (%s), falling back to polling", exc)
        return await _poll_for_captures(front_id, back_id, timeout_s)


async def _poll_for_captures(
    front_id: str,
    back_id: str,
    timeout_s: float = 300,
) -> bool:
    """Poll DB until both captures complete. Used as fallback when Redis is down."""
    db = SessionLocal()
    try:
        interval = 1.0
        elapsed = 0.0
        while elapsed < timeout_s:
            await asyncio.sleep(interval)
            elapsed += interval

            front_cap = db.query(Capture).filter_by(id=front_id).first()
            back_cap = db.query(Capture).filter_by(id=back_id).first()

            front_done = front_cap and front_cap.status in ("completed", "failed")
            back_done = back_cap and back_cap.status in ("completed", "failed")

            if front_done and back_done:
                return True
        return False
    finally:
        db.close()


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _prepare_and_dispatch(
    contents: bytes,
    side: str,
) -> str:
    """Detect card, correct perspective, create capture record, then dispatch OCR to Celery worker.

    Returns capture_id immediately — the heavy OCR runs in a separate Celery process.
    """
    arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image data")

    # YOLO detect + perspective correct
    from models.geometry import analyze_frame, correct_perspective

    detection = detect_frame(image)
    bbox = detection.get("bbox", [])
    bbox_tuple = tuple(int(v) for v in bbox) if len(bbox) == 4 else None
    geo = analyze_frame(image, bbox=bbox_tuple)

    corrected = None
    if geo.detected and geo.corners is not None:
        corrected = correct_perspective(image, geo.corners)

    card_crop = corrected if corrected is not None else image

    # Persist capture
    db = SessionLocal()
    try:
        capture = Capture(
            side=side,
            validation_passed=True,
            confidence=round(detection.get("confidence", 0), 3) if detection else 0.0,
            card_type_detected="national_id",
        )
        db.add(capture)
        db.commit()
        db.refresh(capture)
        capture_id = capture.id

        # Upload original to S3 (best effort)
        try:
            s3_key = upload_encrypted(contents, prefix=f"captures/{capture_id}")
            capture.s3_key = s3_key
            capture.status = "processing"
            db.commit()
        except Exception as e:
            logger.warning("S3 upload failed (non-fatal): %s", e)
            capture.status = "processing"
            db.commit()

        # Encode corrected card for Celery dispatch
        corrected_hex = None
        if corrected is not None:
            ok_c, buf_c = cv2.imencode(
                ".jpg", corrected, [cv2.IMWRITE_JPEG_QUALITY, 92]
            )
            if ok_c:
                corrected_hex = buf_c.tobytes().hex()

        # Dispatch heavy OCR to Celery worker (runs in separate process, truly parallel)
        try:
            from celery_worker import ocr_task

            ocr_task.delay(
                contents.hex(),
                capture_id,
                side,
                corrected_card_hex=corrected_hex,
            )
            logger.info("Dispatched ocr_task for capture=%s side=%s", capture_id, side)
        except Exception as e:
            logger.warning("Celery dispatch failed — falling back to inline OCR: %s", e)
            try:
                from tasks.ocr_task import process_ocr

                corrected_bytes = (
                    bytes.fromhex(corrected_hex) if corrected_hex else None
                )
                process_ocr(
                    contents, capture_id, side, corrected_card_bytes=corrected_bytes
                )
                capture.status = "completed"
                db.commit()
            except Exception as e2:
                logger.error("Inline OCR fallback failed: %s", e2)
                capture.status = "failed"
                db.commit()
                raise

        return capture_id
    finally:
        db.close()


@router.post("/start")
async def extract_start(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    _user: dict = Depends(get_current_user_or_api_key),
) -> dict[str, Any]:
    """Start a new extraction session with front+back images.

    Returns immediately with a session_id. Poll /status/{session_id} for progress.
    """
    front_bytes = await front.read()
    back_bytes = await back.read()

    # Create session record
    db = SessionLocal()
    try:
        session = ExtractionSession(status="processing")
        db.add(session)
        db.commit()
        db.refresh(session)
        session_id = session.id
    finally:
        db.close()

    # Kick off background processing (fire-and-forget)
    async def _background() -> None:
        db2 = SessionLocal()
        try:
            sess = db2.query(ExtractionSession).filter_by(id=session_id).first()
            if not sess:
                return

            # 1. Dispatch both sides in parallel (light prep + Celery dispatch)
            front_id: str | None = None
            back_id: str | None = None
            try:
                front_id, back_id = await asyncio.gather(
                    asyncio.get_event_loop().run_in_executor(
                        None, _prepare_and_dispatch, front_bytes, "front"
                    ),
                    asyncio.get_event_loop().run_in_executor(
                        None, _prepare_and_dispatch, back_bytes, "back"
                    ),
                )
            except Exception as e:
                logger.error(
                    "Extraction dispatch failed for session %s: %s", session_id, e
                )
                sess.status = "failed"
                sess.error_reason = str(e)
                db2.commit()
                return

            sess.front_capture_id = front_id
            sess.back_capture_id = back_id
            db2.commit()
            logger.info(
                "Session %s dispatched front=%s back=%s", session_id, front_id, back_id
            )

            # 2. Wait for both Celery workers via Redis Pub/Sub (max 5 min)
            done = await _wait_for_captures(front_id, back_id, timeout_s=300)
            if not done:
                logger.warning("Session %s processing timed out", session_id)
                sess.status = "failed"
                sess.error_reason = "Processing timed out"
                db2.commit()
                return

            # 3. Merge results
            front_result = db2.query(KYCResult).filter_by(capture_id=front_id).first()
            back_result = db2.query(KYCResult).filter_by(capture_id=back_id).first()

            merged: dict[str, Any] = {}
            face_b64: str | None = None

            if front_result and front_result.ocr_fields:
                try:
                    front_payload = json.loads(front_result.ocr_fields)
                    merged.update(front_payload.get("roi_fields", {}))
                    face_b64 = front_payload.get("face_crop_base64")
                except json.JSONDecodeError:
                    pass

            if back_result and back_result.ocr_fields:
                try:
                    back_payload = json.loads(back_result.ocr_fields)
                    merged.update(back_payload.get("roi_fields", {}))
                except json.JSONDecodeError:
                    pass

            sess.merged_fields = json.dumps(merged)
            sess.status = "completed"
            db2.commit()
            logger.info("Extraction session %s completed", session_id)
        finally:
            db2.close()

    asyncio.create_task(_background())

    return {"session_id": session_id, "status": "processing"}


@router.get("/status/{session_id}")
async def extract_status(
    session_id: str,
    _user: dict = Depends(get_current_user_or_api_key),
) -> dict[str, Any]:
    """Get extraction session status and results when complete."""
    db = SessionLocal()
    try:
        sess = db.query(ExtractionSession).filter_by(id=session_id).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")

        result: dict[str, Any] = {
            "session_id": sess.id,
            "status": sess.status,
            "created_at": sess.created_at.isoformat() if sess.created_at else None,
            "updated_at": sess.updated_at.isoformat() if sess.updated_at else None,
        }

        # Self-healing: if background task died (e.g. uvicorn reload), finalize on poll
        if sess.status == "processing":
            front_cap = (
                db.query(Capture).filter_by(id=sess.front_capture_id).first()
                if sess.front_capture_id
                else None
            )
            back_cap = (
                db.query(Capture).filter_by(id=sess.back_capture_id).first()
                if sess.back_capture_id
                else None
            )
            front_done = front_cap is not None and front_cap.status in (
                "completed",
                "failed",
            )
            back_done = back_cap is not None and back_cap.status in (
                "completed",
                "failed",
            )

            if front_done and back_done:
                try:
                    front_result = (
                        db.query(KYCResult)
                        .filter_by(capture_id=sess.front_capture_id)
                        .first()
                    )
                    back_result = (
                        db.query(KYCResult)
                        .filter_by(capture_id=sess.back_capture_id)
                        .first()
                    )

                    merged: dict[str, Any] = {}
                    face_b64: str | None = None

                    if front_result and front_result.ocr_fields:
                        try:
                            front_payload = json.loads(front_result.ocr_fields)
                            merged.update(front_payload.get("roi_fields", {}))
                            face_b64 = front_payload.get("face_crop_base64")
                        except json.JSONDecodeError:
                            pass

                    if back_result and back_result.ocr_fields:
                        try:
                            back_payload = json.loads(back_result.ocr_fields)
                            merged.update(back_payload.get("roi_fields", {}))
                        except json.JSONDecodeError:
                            pass

                    sess.merged_fields = json.dumps(merged)
                    sess.status = "completed"
                    db.commit()
                    logger.info("Session %s finalized on status poll", session_id)
                except Exception as exc:
                    logger.error(
                        "Session %s finalize-on-poll failed: %s", session_id, exc
                    )
                    sess.status = "failed"
                    sess.error_reason = f"Finalize failed: {exc}"
                    db.commit()

        if sess.status == "completed":
            merged = {}
            if sess.merged_fields:
                try:
                    merged = json.loads(sess.merged_fields)
                except json.JSONDecodeError:
                    pass
            result["data"] = merged
            result["face_crop_url"] = (
                f"/api/capture/{sess.front_capture_id}/face-crop"
                if sess.front_capture_id
                else None
            )

        if sess.status == "failed":
            result["error"] = sess.error_reason

        return result
    finally:
        db.close()
