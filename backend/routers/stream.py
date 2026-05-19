"""WebSocket /ws/stream — real-time ID document detection + geometry overlay."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from models.geometry import analyze_frame
from models.yolo_detector import detect_frame

logger = logging.getLogger(__name__)
router = APIRouter()


def _sanitize(obj):
    """Convert numpy scalars to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@router.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """Accept binary JPEG frames, run YOLO detection, return JSON results.

    Protocol:
    - Client sends raw JPEG bytes as binary messages at ≤10fps
    - Server responds with JSON detection result per frame
    - No frames are stored during streaming — detection only
    """
    await websocket.accept()
    logger.info("WebSocket stream connected: %s", websocket.client)

    # Per-connection cache: SAM runs every 5th frame (~2fps at 10fps input)
    _frame_counter = 0
    _cached_geo: Any | None = None
    _cached_mask: np.ndarray | None = None

    try:
        while True:
            data = await websocket.receive_bytes()

            # Decode JPEG → numpy
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Failed to decode frame"})
                continue

            # Run YOLO detection (fast, ~30ms)
            yolo_result = _sanitize(detect_frame(frame))

            # ── Geometry analysis ──────────────────────────────────
            bbox = yolo_result.get("bbox", [])
            bbox_tuple = tuple(int(v) for v in bbox) if len(bbox) == 4 else None

            # Run SAM only every 5th frame; use cached mask in between
            _frame_counter += 1
            run_sam = (_frame_counter % 5 == 1)  # frames 1, 6, 11...

            if run_sam or _cached_geo is None or _cached_mask is None:
                geo = analyze_frame(frame, bbox=bbox_tuple, use_sam=True)
                _cached_geo = geo
                _cached_mask = geo.mask if geo.mask is not None else None
                logger.debug("Stream SAM run (frame %d) source=%s", _frame_counter, geo.source)
            else:
                # Recompute geometry from cached mask + new bbox if bbox moved
                geo = analyze_frame(frame, bbox=bbox_tuple, use_sam=False)
                # Carry forward SAM mask for overlay
                if geo.mask is None and _cached_mask is not None:
                    geo.mask = _cached_mask
                _cached_geo = geo
                logger.debug("Stream cache hit (frame %d) source=%s", _frame_counter, geo.source)

            # Merge geometry into detection result
            if geo.detected:
                frame_h, frame_w = frame.shape[:2]
                yolo_result["geometry"] = _sanitize(geo.to_dict(frame_w, frame_h))
            else:
                yolo_result["geometry"] = None

            await websocket.send_json(yolo_result)

    except WebSocketDisconnect:
        logger.info("WebSocket stream disconnected: %s", websocket.client)
    except Exception as e:
        logger.error("WebSocket stream error: %s", e)
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
