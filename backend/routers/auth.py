"""Auth endpoints — token issuance for KYC sessions."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from core.auth import create_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


class TokenRequest(BaseModel):
    session_id: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/token", response_model=TokenResponse)
async def get_token(req: TokenRequest) -> TokenResponse:
    """Issue a JWT for a KYC session.

    In production this would validate the session_id against a
    pre-registered application. For dev, any non-empty session_id works.
    """
    token = create_token(subject=req.session_id, extra={"type": "kyc_session"})
    return TokenResponse(access_token=token)
