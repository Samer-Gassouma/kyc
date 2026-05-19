"""JWT authentication utilities."""

from __future__ import annotations

import datetime

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from core.config import settings

security = HTTPBearer(auto_error=False)


def create_token(subject: str, extra: dict | None = None) -> str:
    payload = {
        "sub": subject,
        "exp": datetime.datetime.utcnow()
        + datetime.timedelta(minutes=settings.jwt_expire_minutes),
        "iat": datetime.datetime.utcnow(),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))


async def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """Dependency – extracts and validates JWT from header or query param. Returns payload dict."""
    token = creds.credentials if creds else None
    if not token:
        token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    return decode_token(token)


# Lazy-load valid API keys from settings
_valid_api_keys: set[str] | None = None


def _get_api_keys() -> set[str]:
    global _valid_api_keys
    if _valid_api_keys is None:
        _valid_api_keys = {
            k.strip() for k in settings.api_keys.split(",") if k.strip()
        }
    return _valid_api_keys


async def get_current_user_or_api_key(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """Dependency – tries API key first (X-API-Key header), then falls back to JWT.

    Returns payload dict with at least a 'sub' or 'service' key so downstream
    code can identify the caller.
    """
    # 1. API key via header
    api_key = request.headers.get("x-api-key")
    if api_key and api_key in _get_api_keys():
        return {"service": "api_key", "sub": "ewallet"}

    # 2. JWT via Authorization header or ?token= query param
    token = creds.credentials if creds else None
    if not token:
        token = request.query_params.get("token")
    if token:
        return decode_token(token)

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token or API key")
