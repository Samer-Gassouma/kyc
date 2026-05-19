"""YOLOv8 + contour-based real-time document detection wrapper.

Strategy: Use YOLOv8n (COCO) to detect rectangular objects, then run
contour refinement to precisely locate the ID card boundary.  If YOLO
finds nothing relevant, fall back to pure contour analysis.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import cv2
import numpy as np

from core.config import settings

logger = logging.getLogger(__name__)

_model = None

# Tunable via environment variable (or .env) — SAM handles quality gating,
# so YOLO only needs to find the rough card region.
CONF_THRESHOLD = settings.yolo_conf_threshold

# ID card standard aspect ratio (ISO/IEC 7810 ID-1)
ID_ASPECT_RATIO = 1.586
ASPECT_TOLERANCE = 0.35

# COCO classes that may correspond to a card / flat object in-hand
CARD_LIKE_CLASSES = {73: "book", 67: "cell phone", 63: "laptop", 56: "chair"}


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from ultralytics import YOLO

        _model = YOLO(settings.yolo_weights_path)
        logger.info("YOLOv8 loaded from %s", settings.yolo_weights_path)
    except Exception as exc:
        logger.warning("YOLO load failed (%s) — pure contour mode", exc)
        _model = "fallback"
    return _model


# ── Contour-based card finder ──────────────────────────────────────
def _find_card_contour(frame: np.ndarray, roi: tuple[int, int, int, int] | None = None):
    """Find the best quadrilateral contour (ID card) in frame or ROI.

    Returns (approx_contour, minAreaRect) or (None, None).
    """
    if roi:
        rx, ry, rw, rh = roi
        region = frame[ry : ry + rh, rx : rx + rw]
    else:
        region = frame
        rx, ry = 0, 0

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold + Canny for robustness
    edged = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rh_full, rw_full = region.shape[:2]
    min_area = rh_full * rw_full * 0.05

    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if 4 <= len(approx) <= 6 and area > best_area:
            best = approx
            best_area = area

    if best is None:
        return None, None

    # Offset contour back to full-frame coords
    best_shifted = best.copy()
    best_shifted[:, :, 0] += rx
    best_shifted[:, :, 1] += ry

    rect = cv2.minAreaRect(best_shifted)
    return best_shifted, rect


def detect_frame(frame: np.ndarray) -> dict[str, Any]:
    """Run hybrid detection: YOLO + contour refinement."""
    model = _load_model()
    h, w = frame.shape[:2]

    yolo_box = None
    yolo_conf = 0.0

    # ── YOLO pass ──────────────────────────────────────────────────
    if model != "fallback":
        try:
            results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                detections: list[dict[str, Any]] = []
                for i in range(len(boxes)):
                    cls_id = int(boxes[i].cls[0])
                    conf = float(boxes[i].conf[0])
                    bx1, by1, bx2, by2 = boxes[i].xyxy[0].tolist()
                    box_area = (bx2 - bx1) * (by2 - by1)
                    frame_area = h * w
                    # Prefer card-like COCO classes
                    bonus = 0.3 if cls_id in CARD_LIKE_CLASSES else 0.0
                    # Prefer boxes that cover 5-80% of frame (card-sized)
                    ratio = box_area / frame_area
                    if 0.05 < ratio < 0.80:
                        bonus += 0.1
                    score = conf + bonus
                    detections.append({
                        "idx": i,
                        "score": score,
                        "conf": conf,
                        "box": [bx1, by1, bx2, by2],
                    })

                # NMS: keep only the highest-confidence detection per frame
                if detections:
                    detections.sort(key=lambda x: x["score"], reverse=True)
                    best = detections[0]
                    yolo_box = best["box"]
                    yolo_conf = best["conf"]
        except Exception as exc:
            logger.debug("YOLO inference error: %s", exc)

    # ── Contour refinement ─────────────────────────────────────────
    roi = None
    if yolo_box:
        # Expand YOLO box by 15% for contour search
        bx1, by1, bx2, by2 = yolo_box
        pad_x = (bx2 - bx1) * 0.15
        pad_y = (by2 - by1) * 0.15
        rx = max(0, int(bx1 - pad_x))
        ry = max(0, int(by1 - pad_y))
        rw = min(w, int(bx2 + pad_x)) - rx
        rh_roi = min(h, int(by2 + pad_y)) - ry
        roi = (rx, ry, rw, rh_roi)

    contour, rect = _find_card_contour(frame, roi)

    # If contour found, use it; otherwise fall back to YOLO box directly
    if contour is not None and rect is not None:
        (cx, cy), (rw_r, rh_r), angle = rect
        if rw_r < rh_r:
            rw_r, rh_r = rh_r, rw_r
            angle = (angle + 90) % 180
        if angle > 90:
            angle -= 180

        x, y, bw, bh = cv2.boundingRect(contour)
        x1c, y1c = max(0, x), max(0, y)
        x2c, y2c = min(w, x + bw), min(h, y + bh)
        crop = frame[y1c:y2c, x1c:x2c]
        conf_out = max(yolo_conf, 0.70)

        return _build_result(
            detected=True,
            confidence=conf_out,
            bbox=[x1c, y1c, x2c, y2c],
            angle=float(angle) if abs(angle) < 45 else 0.0,
            cx=float(cx),
            cy=float(cy),
            frame_h=h,
            frame_w=w,
            crop=crop,
        )

    elif yolo_box:
        bx1, by1, bx2, by2 = [int(v) for v in yolo_box]
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        crop = frame[by1:by2, bx1:bx2]
        return _build_result(
            detected=True,
            confidence=yolo_conf,
            bbox=[bx1, by1, bx2, by2],
            angle=0.0,
            cx=cx,
            cy=cy,
            frame_h=h,
            frame_w=w,
            crop=crop,
        )

    # ── Pure contour fallback (no YOLO hit) ────────────────────────
    contour, rect = _find_card_contour(frame, None)
    if contour is not None and rect is not None:
        (cx, cy), (rw_r, rh_r), angle = rect
        if rw_r < rh_r:
            rw_r, rh_r = rh_r, rw_r
            angle = (angle + 90) % 180
        if angle > 90:
            angle -= 180
        x, y, bw, bh = cv2.boundingRect(contour)
        x1c, y1c = max(0, x), max(0, y)
        x2c, y2c = min(w, x + bw), min(h, y + bh)
        crop = frame[y1c:y2c, x1c:x2c]
        return _build_result(
            detected=True,
            confidence=0.55,
            bbox=[x1c, y1c, x2c, y2c],
            angle=float(angle) if abs(angle) < 45 else 0.0,
            cx=float(cx),
            cy=float(cy),
            frame_h=h,
            frame_w=w,
            crop=crop,
        )

    return _empty_result()


def _build_result(
    *,
    detected: bool,
    confidence: float,
    bbox: list[int],
    angle: float,
    cx: float,
    cy: float,
    frame_h: int,
    frame_w: int,
    crop: np.ndarray,
) -> dict[str, Any]:
    """Build the structured detection result with quality checks."""
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]

    # ── Quality checks ─────────────────────────────────────────────
    # Centering: bbox center within 18% of frame center (real-world tolerance)
    fc_x, fc_y = frame_w / 2, frame_h / 2
    centered = abs(cx - fc_x) < frame_w * 0.18 and abs(cy - fc_y) < frame_h * 0.18

    # Straightness: angle < 15°, aspect ratio within tolerance
    straight = abs(angle) < 15.0
    aspect = bw / max(bh, 1)
    if aspect < 1:
        aspect = 1 / aspect
    aspect_ok = abs(aspect - ID_ASPECT_RATIO) < ASPECT_TOLERANCE

    # Visibility: bbox area ≥ 20% of frame (card just needs to be visible)
    frame_area = frame_w * frame_h
    bbox_area = bw * bh
    fully_visible = bbox_area >= frame_area * 0.20
    # Not clipped by frame edge
    margin = 3
    not_clipped = bbox[0] > margin and bbox[1] > margin and bbox[2] < frame_w - margin and bbox[3] < frame_h - margin

    # Blur: Laplacian variance on crop (relaxed for mobile cameras)
    sharp = True
    if crop is not None and crop.size > 0:
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        lap_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
        sharp = lap_var > 35.0

    # Glare / Exposure: mean pixel value (relaxed range)
    lighting_ok = True
    if crop is not None and crop.size > 0:
        mean_val = np.mean(crop)
        lighting_ok = 25.0 < mean_val < 235.0

    issues: list[str] = []
    if not centered:
        issues.append("Center the card in the frame")
    if not straight:
        issues.append("Straighten the card")
    if not aspect_ok:
        issues.append("Card aspect ratio incorrect")
    if not fully_visible:
        issues.append("Move closer — card too small")
    if not not_clipped:
        issues.append("Entire card must be visible")
    if not sharp:
        issues.append("Image is blurry — hold steady")
    if not lighting_ok:
        issues.append("Adjust lighting — too dark or bright")

    # ready_to_capture: card must be detected, not clipped, and most quality checks pass
    quality_passes = sum([centered, straight, aspect_ok, fully_visible, not_clipped, sharp, lighting_ok])
    ready = detected and confidence >= 0.30 and not_clipped and quality_passes >= 5

    return {
        "detected": bool(detected),
        "model": "yolov9",
        "confidence": round(float(confidence), 3),
        "bbox": [int(v) for v in bbox],
        "rotated_bbox": {"angle": round(float(angle), 2), "cx": round(float(cx), 1), "cy": round(float(cy), 1)},
        "quality": {
            "centered": bool(centered),
            "straight": bool(straight),
            "fully_visible": bool(fully_visible and not_clipped),
            "sharp": bool(sharp),
            "lighting_ok": bool(lighting_ok),
        },
        "issues": issues,
        "ready_to_capture": bool(ready),
    }


def _empty_result() -> dict[str, Any]:
    return {
        "detected": False,
        "model": "yolov9",
        "confidence": 0.0,
        "bbox": [],
        "rotated_bbox": {"angle": 0, "cx": 0, "cy": 0},
        "quality": {
            "centered": False,
            "straight": False,
            "fully_visible": False,
            "sharp": False,
            "lighting_ok": False,
        },
        "issues": ["No ID card detected"],
        "ready_to_capture": False,
    }
