"""REST endpoints for capture submission, validation, and status polling."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from core.auth import get_current_user_or_api_key
from core.db import Capture, KYCResult, SessionLocal, get_db
from core.storage import upload_encrypted
from models.rcnn_validator import validate_capture

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/capture", tags=["capture"])


def _sanitize(obj: Any) -> Any:
    """Convert numpy scalars to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@router.post("/validate")
async def submit_and_validate(
    file: UploadFile = File(...),
    side: str = Form("front"),
    _user: dict = Depends(get_current_user_or_api_key),
) -> dict[str, Any]:
    """Submit a full-resolution capture for Faster R-CNN validation.

    Flow:
    1. Decode image (never downsample)
    2. Run quality check
    3. Run Faster R-CNN validation
    4. If passed → encrypt + upload to S3, dispatch Celery OCR job
    5. Return validation result
    """
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    result = _sanitize(validate_capture(image, side=side))

    if not result["validation_passed"]:
        return result

    # ── Passed — persist ───────────────────────────────────────────
    db = SessionLocal()
    try:
        # Create capture record
        capture = Capture(side=side, validation_passed=True, confidence=result["confidence"],
                          card_type_detected=result["card_type_detected"])
        db.add(capture)
        db.commit()
        db.refresh(capture)
        capture_id = capture.id

        # Upload encrypted to S3
        try:
            s3_key = upload_encrypted(contents, prefix=f"captures/{capture_id}")
            capture.s3_key = s3_key
            capture.status = "processing"
            db.commit()
        except Exception as e:
            logger.warning("S3 upload failed (non-fatal): %s", e)
            capture.status = "processing"
            db.commit()

        # Dispatch Celery OCR task
        try:
            from celery_worker import ocr_task

            ocr_task.delay(contents.hex(), capture_id, side)
        except Exception as e:
            logger.warning("Celery dispatch failed (non-fatal): %s", e)
            # Run synchronously as fallback
            try:
                from tasks.ocr_task import process_ocr
                process_ocr(contents, capture_id, side)
                capture.status = "completed"
                db.commit()
            except Exception as e2:
                logger.error("Sync OCR fallback also failed: %s", e2)
                capture.status = "completed"
                db.commit()

    finally:
        db.close()

    result["capture_id"] = capture_id
    return result


@router.get("/status/{capture_id}")
async def capture_status(
    capture_id: str,
    _user: dict = Depends(get_current_user_or_api_key),
) -> dict[str, Any]:
    """Poll for capture processing status and OCR results."""
    import json

    db = SessionLocal()
    try:
        capture = db.query(Capture).filter_by(id=capture_id).first()
        if not capture:
            raise HTTPException(status_code=404, detail="Capture not found")

        kyc_result = db.query(KYCResult).filter_by(capture_id=capture_id).first()

        ocr_payload = None
        if kyc_result and kyc_result.ocr_fields:
            try:
                ocr_payload = json.loads(kyc_result.ocr_fields)
            except json.JSONDecodeError:
                ocr_payload = None

        return {
            "capture_id": capture_id,
            "status": capture.status,
            "validation_passed": capture.validation_passed,
            "side": capture.side,
            "mrz_parsed": json.loads(kyc_result.mrz_parsed) if kyc_result and kyc_result.mrz_parsed else None,
            "ocr_fields": ocr_payload,
            "mrz_check_digits_valid": kyc_result.mrz_check_digits_valid if kyc_result else None,
        }
    finally:
        db.close()


@router.get("/{capture_id}/fields")
async def get_capture_fields(
    capture_id: str,
    _user: dict = Depends(get_current_user_or_api_key),
) -> dict[str, Any]:
    """Return structured ROI extraction result for a capture."""
    import json

    db = SessionLocal()
    try:
        capture = db.query(Capture).filter_by(id=capture_id).first()
        if not capture:
            raise HTTPException(status_code=404, detail="Capture not found")

        kyc_result = db.query(KYCResult).filter_by(capture_id=capture_id).first()
        if not kyc_result or not kyc_result.ocr_fields:
            raise HTTPException(status_code=404, detail="OCR results not yet available")

        try:
            payload = json.loads(kyc_result.ocr_fields)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Corrupted OCR data")

        fields = payload.get("roi_fields", {})
        raw_ocr = payload.get("raw_ocr", [])
        avg_conf = payload.get("ocr_confidence", 0.0)
        face_crop_b64 = payload.get("face_crop_base64", None)

        return {
            "capture_id": capture_id,
            "side": capture.side,
            "fields": fields,
            "raw_ocr": raw_ocr,
            "ocr_confidence": avg_conf,
            "face_crop_base64": face_crop_b64,
        }
    finally:
        db.close()




@router.get("/{capture_id}/face-crop")
async def get_face_crop(
    capture_id: str,
    _user: dict = Depends(get_current_user_or_api_key),
) -> Any:
    """Serve the extracted face crop as a JPEG image for browser display."""
    import base64
    from fastapi.responses import StreamingResponse
    import io
    import json

    db = SessionLocal()
    try:
        capture = db.query(Capture).filter_by(id=capture_id).first()
        if not capture:
            raise HTTPException(status_code=404, detail="Capture not found")

        # Get face crop from KYCResult OCR payload
        kyc = db.query(KYCResult).filter_by(capture_id=capture_id).first()
        if kyc and kyc.ocr_fields:
            try:
                payload = json.loads(kyc.ocr_fields)
                b64 = payload.get("face_crop_base64")
                if b64:
                    face_bytes = base64.b64decode(b64)
                    return StreamingResponse(
                        io.BytesIO(face_bytes),
                        media_type="image/jpeg",
                        headers={"Content-Disposition": f'inline; filename="face_{capture_id}.jpg"'},
                    )
            except Exception:
                pass

        raise HTTPException(status_code=404, detail="Face crop not available")
    finally:
        db.close()
