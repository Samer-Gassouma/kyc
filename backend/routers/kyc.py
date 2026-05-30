"""KYC verification flow — session-based multi-step identity verification."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

import asyncpg
import cv2
import httpx
import numpy as np
from core.config import settings
from core.pg_db import get_pg_db, get_raw_db
from core.storage import upload_encrypted, download_decrypted
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from models.face import get_face_encoder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kyc", tags=["kyc"])

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


async def get_session_or_404(session_id: uuid.UUID, db: asyncpg.Connection) -> dict:
    row = await db.fetchrow("SELECT * FROM kyc_sessions WHERE id = $1", session_id)
    if not row:
        raise HTTPException(404, "Session not found")
    return dict(row)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ──────────────────────────────────────────────────────────────────────
# POST /session — create new KYC session
# ──────────────────────────────────────────────────────────────────────


@router.post("/session")
async def create_session(
    db: asyncpg.Connection = Depends(get_raw_db),
) -> dict[str, Any]:
    session_id = await db.fetchval(
        "INSERT INTO kyc_sessions DEFAULT VALUES RETURNING id"
    )
    return {"session_id": str(session_id)}


# ──────────────────────────────────────────────────────────────────────
# PATCH /session/{id}/liveness — store liveness result
# ──────────────────────────────────────────────────────────────────────


@router.patch("/session/{session_id}/liveness")
async def store_liveness(
    session_id: uuid.UUID,
    liveness_passed: bool = Form(...),
    liveness_score: float = Form(0.0),
    db: asyncpg.Connection = Depends(get_raw_db),
) -> dict[str, Any]:
    await get_session_or_404(session_id, db)
    await db.execute(
        """
        UPDATE kyc_sessions
        SET liveness_passed = $1, liveness_score = $2, updated_at = NOW()
        WHERE id = $3
        """,
        liveness_passed,
        liveness_score,
        session_id,
    )
    return {"ok": True}


# ──────────────────────────────────────────────────────────────────────
# POST /session/{id}/phone/send — generate & send OTP via SMS
# ──────────────────────────────────────────────────────────────────────


@router.post("/session/{session_id}/phone/send")
async def send_phone_otp(
    session_id: uuid.UUID,
    phone_number: str = Form(...),
    db: asyncpg.Connection = Depends(get_raw_db),
) -> dict[str, Any]:
    session = await get_session_or_404(session_id, db)
    if not session["liveness_passed"]:
        raise HTTPException(403, "Liveness check required first")

    otp = str(secrets.randbelow(900000) + 100000)
    otp_hash = hashlib.sha256(otp.encode()).hexdigest()
    expires = _utcnow() + timedelta(minutes=10)

    await db.execute(
        """
        UPDATE kyc_sessions
        SET phone_number = $1, phone_otp_hash = $2,
            phone_otp_expires_at = $3, phone_otp_attempts = 0, updated_at = NOW()
        WHERE id = $4
        """,
        phone_number,
        otp_hash,
        expires,
        session_id,
    )

    if settings.debug:
        logger.info("[DEV] Phone OTP for %s: %s", phone_number, otp)
    else:
        logger.info("Phone OTP sent to %s", phone_number)
    # TODO: integrate SMS provider (Twilio, Vonage, etc.)

    return {"sent": True, "expires_in": 600}


# ──────────────────────────────────────────────────────────────────────
# POST /session/{id}/phone/verify — verify OTP
# ──────────────────────────────────────────────────────────────────────


@router.post("/session/{session_id}/phone/verify")
async def verify_phone_otp(
    session_id: uuid.UUID,
    otp: str = Form(...),
    db: asyncpg.Connection = Depends(get_raw_db),
) -> dict[str, Any]:
    session = await get_session_or_404(session_id, db)

    if session["phone_otp_attempts"] >= 5:
        raise HTTPException(429, "Too many attempts")
    if _utcnow() > session["phone_otp_expires_at"]:
        raise HTTPException(400, "OTP expired")

    await db.execute(
        "UPDATE kyc_sessions SET phone_otp_attempts = phone_otp_attempts + 1 WHERE id = $1",
        session_id,
    )

    submitted_hash = hashlib.sha256(otp.encode()).hexdigest()
    if submitted_hash != session["phone_otp_hash"]:
        raise HTTPException(400, "Invalid OTP")

    await db.execute(
        "UPDATE kyc_sessions SET phone_verified = TRUE, updated_at = NOW() WHERE id = $1",
        session_id,
    )

    return {"verified": True}


# ──────────────────────────────────────────────────────────────────────
# PATCH /session/{id}/document-type — store country + document type
# ──────────────────────────────────────────────────────────────────────


@router.patch("/session/{session_id}/document-type")
async def set_document_type(
    session_id: uuid.UUID,
    country: str = Form("TN"),
    document_type: str = Form("id_card"),
    db: asyncpg.Connection = Depends(get_raw_db),
) -> dict[str, Any]:
    session = await get_session_or_404(session_id, db)
    if not session["phone_verified"]:
        raise HTTPException(403, "Phone verification required first")

    await db.execute(
        """
        UPDATE kyc_sessions
        SET document_country = $1, document_type = $2, updated_at = NOW()
        WHERE id = $3
        """,
        country,
        document_type,
        session_id,
    )
    return {"ok": True}


# ──────────────────────────────────────────────────────────────────────
# POST /session/{id}/document/front — upload front image + run OCR
# ──────────────────────────────────────────────────────────────────────


@router.post("/session/{session_id}/document/front")
async def upload_document_front(
    session_id: uuid.UUID,
    image: UploadFile = File(...),
    db: asyncpg.Connection = Depends(get_raw_db),
) -> dict[str, Any]:
    session = await get_session_or_404(session_id, db)
    if not session["phone_verified"]:
        raise HTTPException(403, "Phone verification required first")

    img_bytes = await image.read()
    s3_key = f"kyc/{session_id}/front.jpg"

    # Store image to S3
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, upload_encrypted, img_bytes, f"kyc/{session_id}"
        )
    except Exception as e:
        logger.warning("S3 upload failed (non-fatal): %s", e)

    # Run OCR pipeline
    extracted = await _run_ocr(img_bytes, session["document_type"])

    await db.execute(
        """
        UPDATE kyc_sessions
        SET document_front_s3_key = $1, document_data = $2, updated_at = NOW()
        WHERE id = $3
        """,
        s3_key,
        json.dumps(extracted, default=str),
        session_id,
    )

    return {"stored": True, "extracted": extracted}


# ──────────────────────────────────────────────────────────────────────
# POST /session/{id}/document/back — upload back image + run OCR
# ──────────────────────────────────────────────────────────────────────


@router.post("/session/{session_id}/document/back")
async def upload_document_back(
    session_id: uuid.UUID,
    image: UploadFile = File(...),
    db: asyncpg.Connection = Depends(get_raw_db),
) -> dict[str, Any]:
    session = await get_session_or_404(session_id, db)
    if not session["phone_verified"]:
        raise HTTPException(403, "Phone verification required first")
    if not session["document_front_s3_key"]:
        raise HTTPException(403, "Document front required first")

    img_bytes = await image.read()
    s3_key = f"kyc/{session_id}/back.jpg"

    try:
        await asyncio.get_event_loop().run_in_executor(
            None, upload_encrypted, img_bytes, f"kyc/{session_id}"
        )
    except Exception as e:
        logger.warning("S3 upload failed (non-fatal): %s", e)

    # Run OCR pipeline on back
    extracted = await _run_ocr(img_bytes, session["document_type"])

    # Merge back data into document_data
    existing_data = session["document_data"] or {}
    if isinstance(existing_data, str):
        try:
            existing_data = json.loads(existing_data)
        except json.JSONDecodeError:
            existing_data = {}
    existing_data.update(extracted)

    await db.execute(
        """
        UPDATE kyc_sessions
        SET document_back_s3_key = $1, document_data = $2, updated_at = NOW()
        WHERE id = $3
        """,
        s3_key,
        json.dumps(existing_data, default=str),
        session_id,
    )

    return {"stored": True, "extracted": extracted}


async def _run_ocr(img_bytes: bytes, document_type: str) -> dict[str, Any]:
    """Run the existing OCR pipeline on document image bytes."""
    try:
        from models.yolo_detector import detect_frame
        from models.roi_extractor import extract_roi_fields

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Invalid image data"}

        # Detect card + perspective correct
        detection = detect_frame(image)
        bbox_raw = detection.get("bbox", [])
        corrected = image

        if len(bbox_raw) == 4:
            try:
                from models.geometry import analyze_frame, correct_perspective

                bbox_tuple = tuple(int(v) for v in bbox_raw)
                geo = analyze_frame(image, bbox=bbox_tuple)
                if geo.detected and geo.corners is not None:
                    corrected = correct_perspective(image, geo.corners)
            except Exception as e:
                logger.warning("Perspective correction failed: %s", e)

        # Extract ROI fields
        loop = asyncio.get_event_loop()
        fields = await loop.run_in_executor(
            None, extract_roi_fields, corrected, document_type
        )
        return fields if isinstance(fields, dict) else {"raw": str(fields)}
    except Exception as e:
        logger.error("OCR pipeline failed: %s", e)
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────────────
# POST /session/{id}/face — enroll face + cross-check against document
# ──────────────────────────────────────────────────────────────────────


@router.post("/session/{session_id}/face")
async def enroll_face(
    session_id: uuid.UUID,
    live_image: UploadFile = File(...),
    liveness_score: float = Form(0.0),
    landmarks_3d: str | None = Form(None),
    quality_score: float | None = Form(None),
    db: asyncpg.Connection = Depends(get_raw_db),
    pg_db: AsyncSession = Depends(get_pg_db),
) -> dict[str, Any]:
    session = await get_session_or_404(session_id, db)
    if not session["phone_verified"]:
        raise HTTPException(403, "Previous steps incomplete")
    if not session["document_front_s3_key"]:
        raise HTTPException(403, "Document upload required first")

    live_bytes = await live_image.read()

    # ── Enroll face embedding ────────────────────────────────────────
    from core.pg_db import User, FaceProfile

    loop = asyncio.get_event_loop()
    encoder = get_face_encoder()

    live_arr = cv2.imdecode(np.frombuffer(live_bytes, np.uint8), cv2.IMREAD_COLOR)
    if live_arr is None:
        raise HTTPException(400, "Invalid live image")

    live_result = await loop.run_in_executor(None, encoder.encode, live_arr)
    if live_result is None:
        raise HTTPException(400, "No face detected in live image")

    live_emb, _aligned = live_result

    # Parse landmarks
    landmarks_json = None
    if landmarks_3d:
        try:
            landmarks_json = json.loads(landmarks_3d)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid landmarks_3d JSON")

    # Create user + profile
    new_user = User()
    pg_db.add(new_user)
    await pg_db.flush()
    uid = new_user.id

    profile = FaceProfile(
        user_id=uid,
        embedding=live_emb.tolist(),
        landmarks_3d=landmarks_json,
        liveness_score=liveness_score,
        quality_score=quality_score if quality_score and quality_score > 0 else None,
        verified=True,
    )
    pg_db.add(profile)
    await pg_db.commit()

    # ── Cross-check: live face vs document photo ─────────────────────
    match = False
    similarity = 0.0

    try:
        # Load document image from S3
        doc_bytes = await loop.run_in_executor(
            None, download_decrypted, session["document_front_s3_key"]
        )
        doc_arr = cv2.imdecode(np.frombuffer(doc_bytes, np.uint8), cv2.IMREAD_COLOR)
        if doc_arr is not None:
            doc_result = await loop.run_in_executor(None, encoder.encode, doc_arr)
            if doc_result is not None:
                doc_emb, _ = doc_result
                similarity = float(np.dot(live_emb, doc_emb))
                match = similarity >= settings.face_match_threshold
    except Exception as e:
        logger.warning("Face-to-document cross-check failed: %s", e)

    await db.execute(
        """
        UPDATE kyc_sessions
        SET face_user_id = $1, face_document_match = $2,
            face_document_similarity = $3, updated_at = NOW()
        WHERE id = $4
        """,
        uid,
        match,
        similarity,
        session_id,
    )

    return {
        "enrolled": True,
        "face_match": match,
        "similarity": round(similarity, 4),
        "user_id": str(uid),
    }


# ──────────────────────────────────────────────────────────────────────
# POST /session/{id}/email/send — generate & send OTP via email
# ──────────────────────────────────────────────────────────────────────


@router.post("/session/{session_id}/email/send")
async def send_email_otp(
    session_id: uuid.UUID,
    email: str = Form(...),
    db: asyncpg.Connection = Depends(get_raw_db),
) -> dict[str, Any]:
    session = await get_session_or_404(session_id, db)
    if not session["face_document_match"]:
        raise HTTPException(403, "Face verification required first")

    otp = str(secrets.randbelow(900000) + 100000)
    otp_hash = hashlib.sha256(otp.encode()).hexdigest()
    expires = _utcnow() + timedelta(minutes=10)

    await db.execute(
        """
        UPDATE kyc_sessions
        SET email = $1, email_otp_hash = $2,
            email_otp_expires_at = $3, email_otp_attempts = 0, updated_at = NOW()
        WHERE id = $4
        """,
        email,
        otp_hash,
        expires,
        session_id,
    )

    if settings.debug:
        logger.info("[DEV] Email OTP for %s: %s", email, otp)
    else:
        logger.info("Email OTP sent to %s", email)
    # TODO: integrate email provider (SendGrid, SES, SMTP, etc.)

    return {"sent": True, "expires_in": 600}


# ──────────────────────────────────────────────────────────────────────
# POST /session/{id}/email/verify — verify email OTP
# ──────────────────────────────────────────────────────────────────────


@router.post("/session/{session_id}/email/verify")
async def verify_email_otp(
    session_id: uuid.UUID,
    otp: str = Form(...),
    db: asyncpg.Connection = Depends(get_raw_db),
) -> dict[str, Any]:
    session = await get_session_or_404(session_id, db)

    if session["email_otp_attempts"] >= 5:
        raise HTTPException(429, "Too many attempts")
    if _utcnow() > session["email_otp_expires_at"]:
        raise HTTPException(400, "OTP expired")

    await db.execute(
        "UPDATE kyc_sessions SET email_otp_attempts = email_otp_attempts + 1 WHERE id = $1",
        session_id,
    )

    submitted_hash = hashlib.sha256(otp.encode()).hexdigest()
    if submitted_hash != session["email_otp_hash"]:
        raise HTTPException(400, "Invalid OTP")

    await db.execute(
        "UPDATE kyc_sessions SET email_verified = TRUE, updated_at = NOW() WHERE id = $1",
        session_id,
    )

    return {"verified": True}


# ──────────────────────────────────────────────────────────────────────
# POST /session/{id}/submit — final checks → approve/reject + webhook
# ──────────────────────────────────────────────────────────────────────


@router.post("/session/{session_id}/submit")
async def submit_session(
    session_id: uuid.UUID,
    db: asyncpg.Connection = Depends(get_raw_db),
) -> dict[str, Any]:
    session = await get_session_or_404(session_id, db)

    checks = {
        "liveness": session["liveness_passed"],
        "phone": session["phone_verified"],
        "document": session["document_front_s3_key"] is not None,
        "face": session["face_document_match"],
        "email": session["email_verified"],
    }

    failed_checks = [k for k, v in checks.items() if not v]
    rejection_reasons = []

    if failed_checks:
        rejection_reasons = [f"{k}_check_failed" for k in failed_checks]

    # Duplicate CIN check
    doc_data = session.get("document_data") or {}
    if isinstance(doc_data, str):
        try:
            doc_data = json.loads(doc_data)
        except json.JSONDecodeError:
            doc_data = {}

    cin = doc_data.get("cin") or doc_data.get("id_number") or doc_data.get("cin_number")
    if cin:
        duplicate = await db.fetchval(
            """
            SELECT COUNT(*) FROM kyc_sessions
            WHERE document_data->>'cin' = $1
            AND status = 'approved'
            AND id != $2
            """,
            str(cin),
            session_id,
        )
        if duplicate > 0:
            rejection_reasons.append("duplicate_identity")

    status = "approved" if not rejection_reasons else "rejected"

    await db.execute(
        """
        UPDATE kyc_sessions
        SET status = $1, rejection_reasons = $2,
            completed_at = NOW(), updated_at = NOW()
        WHERE id = $3
        """,
        status,
        json.dumps(rejection_reasons),
        session_id,
    )

    # Fire webhook (non-blocking)
    asyncio.create_task(_fire_webhook(session, status))

    return {"status": status, "rejection_reasons": rejection_reasons}


async def _fire_webhook(session: dict, status: str) -> None:
    """Send webhook notification about completed verification."""
    # For now, webhook URL is configurable via settings.
    # In production, this would be per-tenant from a tenants table.
    webhook_url = getattr(settings, "kyc_webhook_url", None)
    if not webhook_url:
        logger.info("No webhook URL configured, skipping")
        return

    payload = {
        "event": "verification.completed",
        "session_id": str(session["id"]),
        "status": status,
        "timestamp": _utcnow().isoformat(),
        "checks": {
            "liveness": session["liveness_passed"],
            "phone": session["phone_verified"],
            "document": session["document_front_s3_key"] is not None,
            "face_match": session["face_document_match"],
            "face_score": session["face_document_similarity"],
            "email": session["email_verified"],
        },
    }

    try:
        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json=payload, timeout=10)
        logger.info("Webhook fired for session %s → %s", session["id"], status)
    except Exception as e:
        logger.error("Webhook failed for session %s: %s", session["id"], e)


# ──────────────────────────────────────────────────────────────────────
# GET /session/{id}/status — poll for current status
# ──────────────────────────────────────────────────────────────────────


@router.get("/session/{session_id}/status")
async def get_session_status(
    session_id: uuid.UUID,
    db: asyncpg.Connection = Depends(get_raw_db),
) -> dict[str, Any]:
    session = await get_session_or_404(session_id, db)

    doc_data = session.get("document_data") or {}
    if isinstance(doc_data, str):
        try:
            doc_data = json.loads(doc_data)
        except json.JSONDecodeError:
            doc_data = {}

    return {
        "session_id": str(session["id"]),
        "status": session["status"],
        "step": _current_step(session),
        "checks": {
            "liveness": session["liveness_passed"],
            "phone": session["phone_verified"],
            "document": session["document_front_s3_key"] is not None,
            "face_match": session["face_document_match"],
            "email": session["email_verified"],
        },
        "rejection_reasons": (
            json.loads(session["rejection_reasons"])
            if isinstance(session["rejection_reasons"], str)
            else (session["rejection_reasons"] or [])
        ),
        "extracted": doc_data,
        "created_at": session["created_at"].isoformat() if session["created_at"] else None,
        "updated_at": session["updated_at"].isoformat() if session["updated_at"] else None,
    }


def _current_step(session: dict) -> str:
    """Derive the current step from session state for UI polling."""
    if session["status"] in ("approved", "rejected"):
        return session["status"]
    if not session["liveness_passed"]:
        return "liveness"
    if not session["phone_verified"]:
        return "phone"
    if not session["document_front_s3_key"]:
        return "document"
    if not session["face_document_match"]:
        return "face"
    if not session["email_verified"]:
        return "email"
    return "in_progress"
