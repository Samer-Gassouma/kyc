"""Face enrollment, verification, and profile endpoints."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import uuid
from typing import Any

import cv2
import numpy as np
from core.auth import get_current_user_or_api_key
from core.config import settings
from core.pg_db import FaceProfile, User, get_pg_db
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pgvector.sqlalchemy import Vector
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.face import get_face_encoder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/face", tags=["face"])


@router.post("/enroll")
async def enroll(
    image: UploadFile = File(...),
    user_id: str | None = Form(None),
    landmarks_3d: str | None = Form(None),
    liveness_score: float = Form(0.0),
    quality_score: float = Form(0.0),
    _user: dict = Depends(get_current_user_or_api_key),
    pg_db: AsyncSession = Depends(get_pg_db),
) -> dict[str, Any]:
    """Enroll a face: generate embedding and store in pgvector.

    Creates a new User if user_id is not provided.
    """
    contents = await image.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image data")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, get_face_encoder().encode, frame)
    if result is None:
        raise HTTPException(400, "No face detected in image")

    embedding, _aligned = result

    # Parse landmarks if provided
    landmarks_json = None
    if landmarks_3d:
        try:
            landmarks_json = json.loads(landmarks_3d)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid landmarks_3d JSON")

    # Resolve or create user
    if user_id:
        try:
            uid = uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(400, "Invalid user_id format")
        existing = await pg_db.execute(select(User).where(User.id == uid))
        if not existing.scalar_one_or_none():
            pg_db.add(User(id=uid))
            await pg_db.flush()
    else:
        new_user = User()
        pg_db.add(new_user)
        await pg_db.flush()
        uid = new_user.id

    profile = FaceProfile(
        user_id=uid,
        embedding=embedding.tolist(),
        landmarks_3d=landmarks_json,
        liveness_score=liveness_score,
        quality_score=quality_score if quality_score > 0 else None,
        verified=True,
    )
    pg_db.add(profile)
    await pg_db.commit()

    return {
        "user_id": str(uid),
        "verified": True,
        "embedding_dim": len(embedding),
    }


@router.post("/verify")
async def verify(
    image: UploadFile = File(...),
    user_id: str = Form(...),
    _user: dict = Depends(get_current_user_or_api_key),
    pg_db: AsyncSession = Depends(get_pg_db),
) -> dict[str, Any]:
    """Verify a face against stored embedding using cosine similarity."""
    contents = await image.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image data")

    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(400, "Invalid user_id format")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, get_face_encoder().encode, frame)
    if result is None:
        raise HTTPException(400, "No face detected in image")

    embedding, _aligned = result

    # Query pgvector for closest match
    query = (
        select(
            FaceProfile,
            FaceProfile.embedding.cosine_distance(embedding.tolist()).label("distance"),
        )
        .where(FaceProfile.user_id == uid)
        .order_by(FaceProfile.embedding.cosine_distance(embedding.tolist()))
        .limit(1)
    )
    row = (await pg_db.execute(query)).first()
    if row is None:
        raise HTTPException(404, "No face profile found for this user")

    distance = float(row.distance)
    confidence = 1.0 - distance  # cosine distance → similarity

    return {
        "matched": confidence >= settings.face_match_threshold,
        "confidence": round(confidence, 4),
        "threshold_used": settings.face_match_threshold,
        "user_id": str(uid),
    }


@router.get("/profile/{user_id}")
async def get_profile(
    user_id: str,
    _user: dict = Depends(get_current_user_or_api_key),
    pg_db: AsyncSession = Depends(get_pg_db),
) -> dict[str, Any]:
    """Get the latest face profile for a user."""
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(400, "Invalid user_id format")

    query = (
        select(FaceProfile)
        .where(FaceProfile.user_id == uid)
        .order_by(FaceProfile.created_at.desc())
        .limit(1)
    )
    result = await pg_db.execute(query)
    profile = result.scalar_one_or_none()
    if profile is None:
        raise HTTPException(404, "No face profile found for this user")

    return {
        "user_id": str(profile.user_id),
        "verified": profile.verified,
        "liveness_score": profile.liveness_score,
        "quality_score": profile.quality_score,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
    }
