"""Celery task: OCR + MRZ parsing (EasyOCR + passporteye)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import cv2
import numpy as np

from core.config import settings
from core.db import SessionLocal, KYCResult, Capture

logger = logging.getLogger(__name__)


_reader = None


def _get_reader():
    """Cache EasyOCR reader globally (heavy init)."""
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["ar", "en"], gpu=True, verbose=False)
    return _reader


def _run_easyocr(image: np.ndarray) -> list[dict[str, Any]]:
    """Run EasyOCR and return structured results."""
    try:
        reader = _get_reader()
        results = reader.readtext(image)
        return [
            {"text": text, "confidence": float(conf), "bbox": [list(map(int, p)) for p in bbox]}
            for bbox, text, conf in results
        ]
    except Exception as e:
        logger.error("EasyOCR failed: %s", e)
        return []


def _run_mrz_parse(image: np.ndarray) -> dict[str, Any] | None:
    """Attempt MRZ detection and parsing.

    Tries passporteye first, falls back to regex over OCR text.
    """
    try:
        from passporteye import read_mrz
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, image)
            tmp_path = f.name

        mrz = read_mrz(tmp_path)
        os.unlink(tmp_path)

        if mrz is not None:
            mrz_data = mrz.to_dict()
            return {
                "surname": mrz_data.get("surname", ""),
                "given_names": mrz_data.get("names", ""),
                "nationality": mrz_data.get("nationality", ""),
                "dob": mrz_data.get("date_of_birth", ""),
                "expiry": mrz_data.get("expiration_date", ""),
                "id_number": mrz_data.get("number", ""),
                "sex": mrz_data.get("sex", ""),
                "valid_score": mrz_data.get("valid_score", 0),
                "check_digits_valid": mrz_data.get("valid_score", 0) >= 80,
            }
    except ImportError:
        logger.info("passporteye not available, using regex MRZ fallback")
    except Exception as e:
        logger.error("passporteye MRZ failed: %s", e)

    try:
        ocr_results = _run_easyocr(image)
        all_text = "\n".join(r["text"] for r in ocr_results)
        mrz_lines = re.findall(r"[A-Z0-9<]{30,44}", all_text)
        if len(mrz_lines) >= 2:
            line1, line2 = mrz_lines[0], mrz_lines[1]
            return {
                "surname": line1[5:].split("<<")[0].replace("<", " ").strip(),
                "given_names": line1[5:].split("<<")[1].replace("<", " ").strip() if "<<" in line1[5:] else "",
                "nationality": line2[15:18].replace("<", ""),
                "dob": line2[13:19] if len(line2) > 19 else "",
                "id_number": line2[0:9].replace("<", ""),
                "sex": line2[20] if len(line2) > 20 else "",
                "valid_score": 50,
                "check_digits_valid": False,
            }
    except Exception as e:
        logger.error("MRZ regex fallback failed: %s", e)

    return None


def _validate_tunisian_cin(cin: str) -> bool:
    """Validate Tunisian CIN format: 8 digits."""
    return bool(re.match(r"^\d{8}$", cin.strip()))


def _save_roi_result(
    capture_id: str,
    side: str,
    fields: dict[str, Any],
    raw_ocr: list[dict[str, Any]],
    face_crop: np.ndarray | None,
) -> None:
    """Persist ROI extraction result to DB and store face crop if present."""
    db = SessionLocal()
    try:
        # Store face crop to S3 if available (front side only)
        face_key: str | None = None
        if side == "front":
            if face_crop is None:
                logger.warning("[face_crop] capture=%s side=front: face_crop is None", capture_id)
            else:
                logger.info("[face_crop] capture=%s side=front: shape=%s dtype=%s", capture_id, face_crop.shape, face_crop.dtype)
                try:
                    from core.storage import upload_encrypted
                    ok, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    logger.info("[face_crop] imencode ok=%s buf_len=%s", ok, len(buf) if ok else 0)
                    if ok:
                        face_key = upload_encrypted(
                            buf.tobytes(),
                            prefix=f"captures/{capture_id}/face_crop",
                        )
                        logger.info("[face_crop] uploaded key=%s", face_key)
                except Exception as exc:
                    logger.warning("[face_crop] upload failed: %s", exc)

        # Update capture record with face crop key and mark completed
        capture = db.query(Capture).filter_by(id=capture_id).first()
        if capture:
            if face_key:
                capture.face_crop_s3_key = face_key
            capture.status = "completed"

        # Compute average OCR confidence
        confidences = [r["confidence"] for r in raw_ocr if r.get("confidence", 0) > 0]
        avg_conf = round(float(np.mean(confidences)), 3) if confidences else 0.0

        # Encode face crop as base64 for inline storage (fallback when S3 down)
        face_crop_b64: str | None = None
        if side == "front" and face_crop is not None:
            try:
                import base64
                ok, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    face_crop_b64 = base64.b64encode(buf).decode("utf-8")
                    logger.info("[face_crop] base64 encoded: %s bytes", len(face_crop_b64))
                else:
                    logger.warning("[face_crop] cv2.imencode returned False — crop shape=%s dtype=%s", face_crop.shape, face_crop.dtype)
            except Exception as exc:
                logger.warning("[face_crop] base64 encode failed: %s", exc)

        # Upsert KYCResult
        existing = db.query(KYCResult).filter_by(capture_id=capture_id).first()
        payload = {
            "roi_fields": fields,
            "raw_ocr": raw_ocr,
            "ocr_confidence": avg_conf,
            "face_crop_base64": face_crop_b64,
        }
        if existing:
            existing.ocr_fields = json.dumps(payload)
            existing.mrz_check_digits_valid = fields.get("id_number_valid", False)
        else:
            db.add(KYCResult(
                capture_id=capture_id,
                side=side,
                ocr_fields=json.dumps(payload),
                mrz_check_digits_valid=fields.get("id_number_valid", False),
            ))
        db.commit()
    finally:
        db.close()


def process_roi_extraction(
    corrected_card: np.ndarray,
    capture_id: str,
    side: str,
) -> dict[str, Any]:
    """Run ROI-based field extraction on the SAM-corrected flat card."""
    from models.roi_extractor import extract_card_fields

    reader = _get_reader()
    result = extract_card_fields(corrected_card, side=side, reader=reader)

    fields = result["fields"]
    raw_ocr = result["raw_ocr"]
    face_crop = result.get("face_crop")

    _save_roi_result(capture_id, side, fields, raw_ocr, face_crop)

    return {
        "capture_id": capture_id,
        "side": side,
        "fields": fields,
        "raw_ocr": raw_ocr,
        "face_crop": face_crop is not None,
    }


def process_ocr(
    image_bytes: bytes,
    capture_id: str,
    side: str,
    corrected_card_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Main OCR processing function — called by Celery worker.

    If corrected_card_bytes is provided, runs ROI-based extraction.
    Otherwise falls back to generic EasyOCR + MRZ on the raw image.
    """
    # ── ROI path: corrected flat card available ─────────────────────
    if corrected_card_bytes:
        arr = np.frombuffer(corrected_card_bytes, dtype=np.uint8)
        corrected_card = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if corrected_card is not None:
            logger.info("Running ROI extraction for capture %s (%s)", capture_id, side)
            return process_roi_extraction(corrected_card, capture_id, side)
        logger.warning("Failed to decode corrected card for %s, falling back", capture_id)

    # ── Fallback: generic OCR on raw uploaded image ────────────────
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        logger.error("Failed to decode image for capture %s", capture_id)
        return {"error": "Failed to decode image"}

    ocr_results = _run_easyocr(image)
    mrz_parsed = _run_mrz_parse(image)

    all_text = " ".join(r["text"] for r in ocr_results)
    ocr_fields = {
        "raw_text": all_text,
        "items": ocr_results,
    }

    cin_match = re.search(r"\b(\d{8})\b", all_text)
    if cin_match:
        cin_val = cin_match.group(1)
        ocr_fields["cin_number"] = cin_val
        ocr_fields["cin_valid"] = _validate_tunisian_cin(cin_val)

    check_valid = False
    if mrz_parsed:
        check_valid = mrz_parsed.get("check_digits_valid", False)

    db = SessionLocal()
    try:
        result = KYCResult(
            capture_id=capture_id,
            side=side,
            mrz_raw=json.dumps(mrz_parsed) if mrz_parsed else None,
            mrz_parsed=json.dumps(mrz_parsed) if mrz_parsed else None,
            ocr_fields=json.dumps(ocr_fields),
            mrz_check_digits_valid=check_valid,
        )
        db.add(result)
        db.commit()
        db.refresh(result)
    finally:
        db.close()

    return {
        "capture_id": capture_id,
        "side": side,
        "mrz_parsed": mrz_parsed,
        "ocr_fields": ocr_fields,
        "mrz_check_digits_valid": check_valid,
    }
