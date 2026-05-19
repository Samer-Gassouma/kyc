"""FastAPI application — mounts all routers, configures CORS, inits DB."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.db import init_db
from routers import auth, capture, extract, gallery, liveness, stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KYC Verification API",
    version="1.0.0",
    description="Real-time KYC ID capture and verification system",
)

# ── CORS ───────────────────────────────────────────────────────────
_origins = [o.strip() for o in settings.allowed_origins.split(",")]
# Dev mode: wildcard disables credentials to avoid browser rejection
_use_wildcard = "*" in _origins
origins = ["*"] if _use_wildcard else _origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=not _use_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(stream.router)
app.include_router(capture.router)
app.include_router(extract.router)
app.include_router(gallery.router)
app.include_router(liveness.router)


@app.on_event("startup")
async def startup():
    init_db()
    logger.info("Database initialized")
    logger.info("CORS origins: %s", origins)


@app.get("/health")
async def health():
    return {"status": "ok"}
