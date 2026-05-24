"""Celery worker configuration and task registration."""

from __future__ import annotations

from celery import Celery

from core.config import settings

app = Celery("kyc_worker", broker=settings.redis_url, backend=settings.redis_url)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@app.task(name="tasks.ocr")
def ocr_task(
    image_bytes_hex: str,
    capture_id: str,
    side: str,
    corrected_card_hex: str | None = None,
) -> dict:
    """Run OCR + MRZ on a captured image."""
    from tasks.ocr_task import process_ocr

    image_bytes = bytes.fromhex(image_bytes_hex)
    corrected_card = bytes.fromhex(corrected_card_hex) if corrected_card_hex else None
    return process_ocr(image_bytes, capture_id, side, corrected_card_bytes=corrected_card)


