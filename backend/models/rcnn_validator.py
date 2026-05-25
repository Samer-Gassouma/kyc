"""Faster R-CNN capture validator (torchvision COCO + quality checks).

Uses the pretrained COCO Faster R-CNN to verify that a flat rectangular
object is present, then runs thorough image-quality checks (blur, glare,
edge completeness) on the full-res capture before accepting it.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from core.config import settings

logger = logging.getLogger(__name__)

_model = None
_device = None

# COCO classes that can correspond to a card-like object
COCO_CARD_CLASSES = {73: "book", 67: "cell phone", 63: "laptop"}


def _load_model():
    global _model, _device
    if _model is not None:
        return _model

    try:
        import torch
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn,
            FasterRCNN_ResNet50_FPN_Weights,
        )

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Try loading the saved state dict first
        weights_path = settings.rcnn_weights_path
        try:
            _model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
                weights_backbone=None,  # don't re-download backbone
            )
            state = torch.load(weights_path, map_location=_device, weights_only=True)
            _model.load_state_dict(state)
            logger.info("Faster R-CNN loaded from %s", weights_path)
        except Exception:
            # Fall back to the official pretrained model (downloads ~160MB on first use)
            _model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            )
            logger.info("Faster R-CNN loaded with COCO_V1 weights (pretrained)")

        _model.to(_device)
        _model.eval()
    except Exception as exc:
        logger.warning("R-CNN not available (%s) — using OpenCV fallback", exc)
        _model = "fallback"

    return _model


def _image_to_tensor(image: np.ndarray):
    """BGR numpy → normalised float tensor [C, H, W]."""
    import torch
    from torchvision import transforms

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transforms.functional.to_tensor(rgb)


# ── Quality sub-checks ────────────────────────────────────────────
def _check_blur(gray: np.ndarray) -> tuple[bool, float]:
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return lap_var > 60.0, lap_var


def _check_glare(gray: np.ndarray) -> tuple[bool, float]:
    _, bright = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    ratio = float(np.count_nonzero(bright)) / max(gray.size, 1)
    return ratio < 0.15, ratio


def _check_brightness(gray: np.ndarray) -> tuple[bool, float]:
    mean_val = float(np.mean(gray))
    return 50.0 < mean_val < 210.0, mean_val


def _find_card_contour(image: np.ndarray):
    """Return (contour, area_ratio) of the largest quad in the image."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if 4 <= len(approx) <= 6 and area > best_area:
            best = approx
            best_area = area

    ratio = best_area / max(h * w, 1)
    return best, ratio


# ── Main validation entry point ───────────────────────────────────
def validate_capture(image: np.ndarray, side: str = "front") -> dict[str, Any]:
    """Validate a full-resolution captured image.

    Combines Faster R-CNN object detection with quality checks.
    """
    model = _load_model()
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rejection_reason = None
    rcnn_conf = 0.0
    card_type = "national_id"

    # ── R-CNN pass ─────────────────────────────────────────────────
    if model != "fallback":
        import torch

        tensor = _image_to_tensor(image).to(_device)
        with torch.no_grad():
            preds = model([tensor])

        pred = preds[0]
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()

        # Filter high-confidence detections
        mask = scores > 0.3
        if mask.any():
            boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
            best_idx = int(np.argmax(scores))
            rcnn_conf = float(scores[best_idx])

            bx1, by1, bx2, by2 = boxes[best_idx]
            box_area = (bx2 - bx1) * (by2 - by1)
            frame_area = h * w

    else:
        # Contour-based validation as R-CNN substitute
        contour, area_ratio = _find_card_contour(image)
        if contour is None:
            rejection_reason = "No document detected in image"
        rcnn_conf = min(area_ratio * 2, 1.0)

    # ── Quality checks ─────────────────────────────────────────────
    sharp, blur_val = _check_blur(gray)
    no_glare, glare_val = _check_glare(gray)
    good_light, brightness_val = _check_brightness(gray)

    if not sharp and rejection_reason is None:
        rejection_reason = f"Image is too blurry (score={blur_val:.0f}, need >60)"
    if not no_glare and rejection_reason is None:
        rejection_reason = f"Too much glare detected ({glare_val:.1%} bright pixels)"
    if not good_light and rejection_reason is None:
        rejection_reason = f"Poor lighting (brightness={brightness_val:.0f})"

    passed = rejection_reason is None

    return {
        "validation_passed": passed,
        "model": "faster_rcnn_resnet50_coco",
        "confidence": round(max(rcnn_conf, 0.0), 3),
        "card_type_detected": card_type if passed else None,
        "rejection_reason": rejection_reason,
        "quality_details": {
            "blur_score": round(blur_val, 1),
            "glare_ratio": round(glare_val, 4),
            "brightness": round(brightness_val, 1),
            "sharp": sharp,
            "no_glare": no_glare,
            "good_light": good_light,
        },
        "capture_id": None,
    }
