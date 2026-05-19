"""Gallery upload endpoint — auto-detect, crop, rectify, validate ID card."""

from __future__ import annotations

import base64
import logging
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from core.auth import get_current_user_or_api_key
from core.db import Capture, SessionLocal
from core.storage import upload_encrypted
from models.geometry import (
    analyze_frame,
    correct_perspective,
    draw_geometry_overlay,
)
from models.quality_checker import check_quality
from models.yolo_detector import detect_frame

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/gallery", tags=["gallery"])


# ── Sanitize for JSON ────────────────────────────────────────────
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


def detect_card_side(corrected_card: np.ndarray) -> str:
    """
    Detect if card is front or back based on visual features.
    Front: has a face photo (skin tones in left region)
    Back: has a fingerprint (dark textured circle in right region)
    """
    h, w = corrected_card.shape[:2]

    left_region = corrected_card[int(h * 0.2):int(h * 0.9), 0:int(w * 0.3)]
    hsv = cv2.cvtColor(left_region, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

    right_region = corrected_card[0:int(h * 0.6), int(w * 0.62):w]
    gray = cv2.cvtColor(right_region, cv2.COLOR_BGR2GRAY)
    dark_ratio = np.sum(gray < 80) / gray.size

    # Strong face photo → front (true fronts: 0.22-0.49)
    if skin_ratio > 0.20:
        return "front"

    # Strong fingerprint → back (true backs: 0.32 or ~0.06)
    if dark_ratio > 0.03:
        return "back"

    # Ambiguous: use ratio comparison
    if skin_ratio > dark_ratio * 3.0:
        return "front"
    if dark_ratio > skin_ratio * 2.0:
        return "back"

    return "unknown"


# ── Main endpoint ───────────────────────────────────────────────
@router.post("/process")
async def gallery_process(
    file: UploadFile = File(...),
    side: str = Form("front"),
    _user: dict = Depends(get_current_user_or_api_key),  # type: ignore[arg-type]
) -> dict[str, Any]:
    """Auto-detect, crop, rectify and validate an ID card from a gallery image.

    Returns:
        {
            "detected": bool,
            "crop_base64": str | null,  # cropped card image as JPEG base64
            "validation_passed": bool,
            "rejection_reason": str | null,
            "capture_id": str | null,
            "detection_confidence": float,
            "quality_details": dict | null,
        }
"""
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # ── Step 1: Detect the card ──────────────────────────────────
    detection = detect_frame(image)
    detection = _sanitize(detection)

    if not detection.get("detected"):
        return {
            "detected": False,
            "crop_base64": None,
            "corrected_base64": None,
            "validation_passed": False,
            "rejection_reason": "No ID card detected in image",
            "capture_id": None,
            "detection_confidence": 0.0,
            "quality_details": None,
            "geometry": None,
        }

    # ── Step 2: Geometry analysis — detect corners, compute measurements ──
    bbox = detection.get("bbox", [])
    bbox_tuple = tuple(int(v) for v in bbox) if len(bbox) == 4 else None

    geo = analyze_frame(image, bbox=bbox_tuple)
    logger.debug("Geometry: detected=%s angle=%.1f issues=%s ready=%s",
                 geo.detected, geo.angle, geo.issues, geo.ready_to_capture)

    # Draw overlay with actual mask outline + corners + brackets + issue text
    annotated = draw_geometry_overlay(image, geo, mask=geo.mask)

    # Encode annotated image
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    crop_b64 = base64.b64encode(buf).decode("utf-8") if ok else None

    # ── Step 3: If geometry is good, apply perspective correction ──
    corrected = None
    if geo.detected and geo.corners is not None:
        corrected = correct_perspective(image, geo.corners)

    # Quality-check the corrected card (if available) or original region
    card_crop = corrected if corrected is not None else image
    quality = _sanitize(check_quality(card_crop))

    # Relaxed thresholds for gallery: user intentionally chose these photos
    blur_ok = quality.get("blur_score", 0) > 8.0   # was 80 → 15 → 8
    glare_ok = quality.get("glare_ratio", 0) < 0.10   # was 0.05 → 0.08 → 0.10
    brightness_ok = 35.0 < quality.get("mean_brightness", 0) < 235.0  # was 60-200 → 40-220 → 35-235

    rejection_reason = None
    if not blur_ok:
        rejection_reason = f"Image is too blurry (score={quality['blur_score']:.0f})"
    elif not glare_ok:
        rejection_reason = f"Too much glare detected ({quality['glare_ratio']:.1%})"
    elif not brightness_ok:
        if quality["mean_brightness"] <= 40:
            rejection_reason = f"Image too dark (brightness={quality['mean_brightness']:.0f})"
        else:
            rejection_reason = f"Image too bright / overexposed (brightness={quality['mean_brightness']:.0f})"

    if rejection_reason:
        frame_h, frame_w = image.shape[:2]
        return {
            "detected": True,
            "crop_base64": crop_b64,
            "corrected_base64": None,
            "validation_passed": False,
            "rejection_reason": rejection_reason,
            "capture_id": None,
            "detection_confidence": round(detection.get("confidence", 0), 3),
            "quality_details": quality,
            "geometry": _sanitize(geo.to_dict(frame_w, frame_h)) if geo.detected else None,
        }

    # ── Step 4: Persist (same as capture router) ──────────────────
    db = SessionLocal()
    try:
        capture = Capture(
            side=side,
            validation_passed=True,
            confidence=round(detection.get("confidence", 0), 3),
            card_type_detected="national_id",
        )
        db.add(capture)
        db.commit()
        db.refresh(capture)
        capture_id = capture.id

        # Upload original image (not crop) to S3 for record keeping
        try:
            s3_key = upload_encrypted(contents, prefix=f"captures/{capture_id}")
            capture.s3_key = s3_key
            capture.status = "processing"
            db.commit()
        except Exception as e:
            logger.warning("S3 upload failed (non-fatal): %s", e)
            capture.status = "processing"
            db.commit()

        # Encode corrected card for OCR task if available
        corrected_bytes = None
        if corrected is not None:
            ok_c, buf_c = cv2.imencode(".jpg", corrected, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok_c:
                corrected_bytes = buf_c.tobytes()

        # Dispatch OCR + ROI extraction
        try:
            from celery_worker import ocr_task
            ocr_task.delay(
                contents.hex(),
                capture_id,
                side,
                corrected_card_hex=corrected_bytes.hex() if corrected_bytes else None,
            )
        except Exception as e:
            logger.warning("Celery dispatch failed (non-fatal): %s", e)
            try:
                from tasks.ocr_task import process_ocr
                process_ocr(
                    contents,
                    capture_id,
                    side,
                    corrected_card_bytes=corrected_bytes,
                )
                capture.status = "completed"
                db.commit()
            except Exception as e2:
                logger.error("Sync OCR fallback also failed: %s", e2)
                capture.status = "completed"
                db.commit()

    finally:
        db.close()

    # Encode corrected card if geometry succeeded
    corrected_b64 = None
    if corrected is not None:
        ok2, buf2 = cv2.imencode(".jpg", corrected, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok2:
            corrected_b64 = base64.b64encode(buf2).decode("utf-8")

    frame_h, frame_w = image.shape[:2]
    return {
        "detected": True,
        "crop_base64": crop_b64,
        "corrected_base64": corrected_b64,
        "validation_passed": True,
        "rejection_reason": None,
        "capture_id": capture_id,
        "detection_confidence": round(detection.get("confidence", 0), 3),
        "quality_details": quality,
        "geometry": _sanitize(geo.to_dict(frame_w, frame_h)) if geo.detected else None,
    }


# ── Debug: ROI overlay visualization ─────────────────────────────
@router.get("/debug/roi/{capture_id}/{side}")
async def debug_roi(
    capture_id: str,
    side: str,
    current_user: dict = Depends(get_current_user_or_api_key),
):
    """Return the corrected card with ROI boxes drawn for visual verification."""
    from models.roi_extractor import draw_roi_debug
    from core.db import Capture

    db = SessionLocal()
    try:
        capture = db.query(Capture).filter(Capture.capture_id == capture_id).first()
        if not capture:
            raise HTTPException(status_code=404, detail="Capture not found")

        # Load corrected image from DB or S3 if available
        # Fallback: use the raw capture image and correct it on-the-fly
        image = cv2.imread(capture.file_path) if capture.file_path else None
        if image is None:
            raise HTTPException(status_code=404, detail="Image not available")

        # Run geometry to get corrected card
        from models.geometry import analyze_frame, correct_perspective
        from models.yolo_detector import detect_frame

        det = detect_frame(image)
        bbox = det.get("bbox", [])
        bbox_tuple = tuple(int(v) for v in bbox) if len(bbox) == 4 else None
        geo = analyze_frame(image, bbox=bbox_tuple)

        if not geo.detected or geo.corners is None:
            raise HTTPException(status_code=422, detail="Could not detect card geometry")

        corrected = correct_perspective(image, geo.corners)
        debug_img = draw_roi_debug(corrected, side)

        ok, buf = cv2.imencode(".jpg", debug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode debug image")

        return {
            "capture_id": capture_id,
            "side": side,
            "debug_image_base64": base64.b64encode(buf).decode("utf-8"),
        }
    finally:
        db.close()
