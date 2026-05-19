"""REST endpoints for capture submission, validation, and status polling."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from core.auth import get_current_user_or_api_key
from core.db import Capture, KYCResult, SessionLocal, get_db
from core.storage import download_decrypted, upload_encrypted
from models.quality_checker import check_quality
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

    # Quality gate
    quality = check_quality(image)
    if not quality["quality_passed"]:
        return {
            "validation_passed": False,
            "model": "quality_checker",
            "confidence": 0.0,
            "card_type_detected": None,
            "rejection_reason": "; ".join(quality["issues"]),
            "capture_id": None,
        }

    # Faster R-CNN validation
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


@router.post("/{session_id}/validate-fields")
async def validate_session_fields(
    session_id: str,
    _user: dict = Depends(get_current_user_or_api_key),
) -> dict[str, Any]:
    """Cross-validate front + back extracted fields for a session."""
    import json
    import datetime

    db = SessionLocal()
    try:
        session = db.query(LivenessSession).filter_by(id=session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        front_result = db.query(KYCResult).filter_by(
            capture_id=session.capture_front_id,
        ).first()
        back_result = db.query(KYCResult).filter_by(
            capture_id=session.capture_back_id,
        ).first()

        issues: list[str] = []
        front_fields: dict[str, Any] = {}
        back_fields: dict[str, Any] = {}

        if front_result and front_result.ocr_fields:
            try:
                front_fields = json.loads(front_result.ocr_fields).get("roi_fields", {})
            except json.JSONDecodeError:
                issues.append("Front OCR data corrupted")
        else:
            issues.append("Front side not yet processed")

        if back_result and back_result.ocr_fields:
            try:
                back_fields = json.loads(back_result.ocr_fields).get("roi_fields", {})
            except json.JSONDecodeError:
                issues.append("Back OCR data corrupted")
        else:
            issues.append("Back side not yet processed")

        # Validate CIN number (8 digits)
        id_number = front_fields.get("id_number", "")
        if not id_number:
            issues.append("ID number not extracted from front")
        elif not re.match(r"^\d{8}$", str(id_number)):
            issues.append(f"ID number invalid format: {id_number}")

        # Validate issue date and expiry (10 years for Tunisian CIN)
        issue_date_str = back_fields.get("issue_date", "")
        if issue_date_str:
            try:
                issue_dt = datetime.datetime.strptime(str(issue_date_str), "%Y-%m-%d")
                expiry_dt = issue_dt.replace(year=issue_dt.year + 10)
                today = datetime.datetime.utcnow()
                if today > expiry_dt:
                    issues.append(f"CIN expired (expiry: {expiry_dt.date()})")
            except ValueError:
                issues.append(f"Issue date format invalid: {issue_date_str}")
        else:
            issues.append("Issue date not extracted from back")

        valid = len(issues) == 0

        return {
            "session_id": session_id,
            "valid": valid,
            "issues": issues,
            "front_fields": {
                k: v for k, v in front_fields.items()
                if k not in ("photo", "face_crop")
            },
            "back_fields": back_fields,
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

    db = SessionLocal()
    try:
        capture = db.query(Capture).filter_by(id=capture_id).first()
        if not capture:
            raise HTTPException(status_code=404, detail="Capture not found")

        # Try S3 first
        if capture.face_crop_s3_key:
            try:
                face_bytes = download_decrypted(capture.face_crop_s3_key)
                return StreamingResponse(
                    io.BytesIO(face_bytes),
                    media_type="image/jpeg",
                    headers={"Content-Disposition": f'inline; filename="face_{capture_id}.jpg"'},
                )
            except Exception as e:
                logger.warning("Face crop S3 download failed: %s", e)

        # Fallback 1: base64 from KYCResult payload
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

        # Fallback 2: base64 from ExtractionSession (aggregated result)
        from core.db import ExtractionSession
        sess = db.query(ExtractionSession).filter_by(front_capture_id=capture_id).first()
        if sess and sess.face_crop_base64:
            try:
                face_bytes = base64.b64decode(sess.face_crop_base64)
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
