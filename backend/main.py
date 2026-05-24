"""FastAPI application — mounts all routers, configures CORS, inits DB."""

from __future__ import annotations

import logging

from core.config import settings
from core.db import init_db
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
app.include_router(liveness.router)
app.include_router(stream.router)
app.include_router(capture.router)
app.include_router(extract.router)
app.include_router(gallery.router)


@app.on_event("startup")
async def startup():
    init_db()
    logger.info("Database initialized")
    logger.info("CORS origins: %s", origins)

    # ── Preload models to avoid first-request latency ──────────────
    import asyncio

    loop = asyncio.get_event_loop()

    async def _warm_models():
        """Warm up all ML models so the first user doesn't pay the init cost."""
        logger.info("Preloading models...")

        # YOLOv8 detector
        try:
            await loop.run_in_executor(
                None,
                lambda: __import__(
                    "models.yolo_detector", fromlist=["_load_model"]
                )._load_model(),
            )
            logger.info("  ✓ YOLOv8 loaded")
        except Exception as e:
            logger.warning("  ✗ YOLOv8 preload failed: %s", e)

        # SAM predictor (if available)
        try:
            await loop.run_in_executor(
                None,
                lambda: __import__(
                    "models.geometry", fromlist=["_get_sam_predictor"]
                )._get_sam_predictor(),
            )
            logger.info("  ✓ SAM loaded")
        except Exception as e:
            logger.info("  - SAM skipped: %s", e)

        # EasyOCR reader
        try:
            await loop.run_in_executor(
                None,
                lambda: __import__(
                    "tasks.ocr_task", fromlist=["_get_reader"]
                )._get_reader(),
            )
            logger.info("  ✓ EasyOCR loaded")
        except Exception as e:
            logger.warning("  ✗ EasyOCR preload failed: %s", e)

        logger.info("All models ready.")

    asyncio.create_task(_warm_models())


@app.get("/health")
async def health():
    return {"status": "ok"}
