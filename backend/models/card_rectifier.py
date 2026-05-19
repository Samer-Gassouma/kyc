"""Production-grade card rectification for KYC.

Conservative, validated pipeline:
  1. Multi-method rotation estimation with consensus voting
  2. Before/after validation — reject harmful rotations
  3. Precise bounding-box crop with padding
  4. Never rotates 90° or over-crops
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ID_ASPECT_RATIO = 85.60 / 53.98  # ≈ 1.586
TARGET_LONG = 1024
TARGET_SHORT = int(TARGET_LONG / ID_ASPECT_RATIO)


# ── Utilities ───────────────────────────────────────────────────
def _lap_var(img: np.ndarray) -> float:
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _horizontal_edge_score(gray: np.ndarray) -> float:
    """Higher = more horizontal structure (text lines, card edges)."""
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # Ratio of horizontal vs vertical gradients
    h_energy = np.sum(np.abs(sobel_y))
    v_energy = np.sum(np.abs(sobel_x)) + 1e-6
    return float(h_energy / v_energy)


def _bbox_from_contour(cnt: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, x + w, y + h


def _pad_crop(img: np.ndarray, pad: float = 0.02) -> np.ndarray:
    """Add small border pad so rotation doesn't clip corners."""
    h, w = img.shape[:2]
    px = max(2, int(w * pad))
    py = max(2, int(h * pad))
    return cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_REPLICATE)


# ── Rotation Estimators ─────────────────────────────────────────
def estimate_by_hough(gray: np.ndarray) -> float:
    """Return dominant angle [-45,45] from Hough lines. 0 = horizontal."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    if lines is None:
        return 0.0
    angles = []
    for l in lines[:40]:
        theta = float(l[0][1])
        deg = np.degrees(theta) - 90
        while deg > 45:
            deg -= 90
        while deg < -45:
            deg += 90
        if abs(deg) < 40:
            angles.append(deg)
    return float(np.median(angles)) if angles else 0.0


def estimate_by_projection(gray: np.ndarray) -> float:
    """Find angle that maximizes horizontal projection variance."""
    best_angle, best_score = 0.0, -1.0
    for angle in np.linspace(-8, 8, 33):
        M = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), angle, 1.0)
        rot = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_NEAREST)
        profile = np.sum(rot, axis=1)
        score = float(np.var(profile))
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def estimate_by_fft(gray: np.ndarray) -> float:
    """FFT-based rotation estimation — very accurate for small angles."""
    # Resize for speed
    h, w = gray.shape
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    else:
        small = gray

    # Polar transform of power spectrum
    f = np.fft.fft2(small)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    # The main energy line in spectrum indicates rotation
    # Use log-polar transform to find angle
    cy, cx = np.array(magnitude.shape) // 2
    radius = min(cy, cx)

    # Sample in polar coordinates
    angles = np.linspace(-45, 45, 181)
    scores = []
    for a in angles:
        rad = np.radians(a)
        # Sample along a ray from center
        ys = cy + np.arange(0, radius) * np.sin(rad)
        xs = cx + np.arange(0, radius) * np.cos(rad)
        ys = np.clip(ys, 0, magnitude.shape[0] - 1).astype(int)
        xs = np.clip(xs, 0, magnitude.shape[1] - 1).astype(int)
        scores.append(float(np.sum(magnitude[ys, xs])))

    best_idx = int(np.argmax(scores))
    return float(angles[best_idx])


def consensus_angle(gray: np.ndarray) -> tuple[float, float]:
    """Run multiple estimators, return (median_angle, confidence).

    confidence = 1 - std/mean_abs  (higher = estimators agree)
    """
    angles = []
    weights = []

    a1 = estimate_by_fft(gray)
    if abs(a1) < 45:
        angles.append(a1)
        weights.append(2.0)  # FFT is most accurate

    a2 = estimate_by_projection(gray)
    if abs(a2) < 45:
        angles.append(a2)
        weights.append(1.0)

    a3 = estimate_by_hough(gray)
    if abs(a3) < 45:
        angles.append(a3)
        weights.append(1.0)

    if not angles:
        return 0.0, 0.0

    angles_arr = np.array(angles)
    # Weighted median
    sorted_idx = np.argsort(angles_arr)
    sorted_angles = angles_arr[sorted_idx]
    sorted_weights = np.array(weights)[sorted_idx]
    cumsum = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    median = float(sorted_angles[median_idx])

    # Confidence = agreement among estimators
    std = float(np.std(angles_arr))
    mean_abs = float(np.mean(np.abs(angles_arr))) + 1e-6
    confidence = max(0.0, 1.0 - std / mean_abs) if mean_abs > 1 else 1.0

    return median, confidence


# ── Safe rotation ───────────────────────────────────────────────
def safe_rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate with bounding-box expansion — never clips corners."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ── Card boundary detection ─────────────────────────────────────
def find_card_bounds(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find card bounding box using multi-strategy contour detection.

    Returns (x1, y1, x2, y2) or None.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Strategy 1: Adaptive threshold + morphological closing
    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 15, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(adap, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0.0
    frame_area = h * w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.03:
            continue

        x1, y1, x2, y2 = _bbox_from_contour(cnt)
        bw, bh = x2 - x1, y2 - y1
        if bw < 50 or bh < 50:
            continue

        aspect = max(bw, bh) / min(bw, bh)
        aspect_err = abs(aspect - ID_ASPECT_RATIO) / ID_ASPECT_RATIO
        if aspect_err > 0.6:
            continue

        # Score: large + correct aspect + near center
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        center_dist = ((cx - w / 2) / w) ** 2 + ((cy - h / 2) / h) ** 2
        score = (area / frame_area) * (1 - aspect_err) * (1 - center_dist)

        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)

    if best and best_score > 0.05:
        return best

    # Strategy 2: Canny + largest contour
    edges = cv2.Canny(gray, 30, 120)
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours2, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0.0
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.03:
            continue
        x1, y1, x2, y2 = _bbox_from_contour(cnt)
        bw, bh = x2 - x1, y2 - y1
        aspect = max(bw, bh) / min(bw, bh)
        aspect_err = abs(aspect - ID_ASPECT_RATIO) / ID_ASPECT_RATIO
        if aspect_err > 0.6:
            continue
        score = (area / frame_area) * (1 - aspect_err)
        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)

    return best


# ── Postprocess ─────────────────────────────────────────────────
def postprocess(image: np.ndarray) -> np.ndarray:
    """Resize to standard ID dimensions without changing orientation."""
    h, w = image.shape[:2]
    aspect = w / h

    if abs(aspect - ID_ASPECT_RATIO) < 0.2:
        resized = cv2.resize(image, (TARGET_LONG, TARGET_SHORT), interpolation=cv2.INTER_LANCZOS4)
    elif aspect > ID_ASPECT_RATIO:
        new_w = int(h * ID_ASPECT_RATIO)
        start_x = (w - new_w) // 2
        crop = image[:, start_x:start_x + new_w]
        resized = cv2.resize(crop, (TARGET_LONG, TARGET_SHORT), interpolation=cv2.INTER_LANCZOS4)
    else:
        new_h = int(w / ID_ASPECT_RATIO)
        start_y = (h - new_h) // 2
        crop = image[start_y:start_y + new_h, :]
        resized = cv2.resize(crop, (TARGET_LONG, TARGET_SHORT), interpolation=cv2.INTER_LANCZOS4)

    # Very mild sharpening
    blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
    return cv2.addWeighted(resized, 1.15, blurred, -0.15, 0)


# ── Main Entry ────────────────────────────────────────────────────
def rectify_card(image: np.ndarray) -> dict[str, Any]:
    """Production-grade rectification: detect tilt → validate → rotate → crop → standardize.

    Returns {"success", "rectified", "angle", "method", "metrics"}.
    """
    original = image.copy()
    h, w = original.shape[:2]
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # ── 1. Detect tilt angle (consensus of 3 methods) ─────────────
    angle, confidence = consensus_angle(gray)
    logger.debug("Consensus angle=%.2f° confidence=%.2f", angle, confidence)

    # ── 2. Validation — only rotate if beneficial ─────────────────
    do_rotate = False
    if abs(angle) > 0.5 and confidence >= 0.5:
        # Test: compare horizontal structure before/after
        before_score = _horizontal_edge_score(gray)

        # Quick test rotation (small, no border expansion needed for test)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        test_rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR)
        after_score = _horizontal_edge_score(test_rot)

        # Also compare sharpness
        before_sharp = _lap_var(gray)
        after_sharp = _lap_var(test_rot)

        # Rotation is beneficial if horizontal alignment improves
        # or stays same AND sharpness doesn't drop significantly
        alignment_improved = after_score > before_score * 0.95
        sharpness_ok = after_sharp > before_sharp * 0.85

        if alignment_improved and sharpness_ok:
            do_rotate = True
            logger.debug("Rotation validated: align_before=%.1f after=%.1f sharp_before=%.0f after=%.0f",
                         before_score, after_score, before_sharp, after_sharp)
        else:
            logger.debug("Rotation REJECTED: align_before=%.1f after=%.1f sharp_before=%.0f after=%.0f",
                         before_score, after_score, before_sharp, after_sharp)
            angle = 0.0

    # ── 3. Apply validated rotation ─────────────────────────────
    if do_rotate:
        # Add padding before rotation to prevent corner clipping
        padded = _pad_crop(original, pad=0.03)
        rotated = safe_rotate(padded, angle)
    else:
        rotated = original.copy()
        angle = 0.0

    # ── 4. Find precise card bounds ───────────────────────────────
    bounds = find_card_bounds(rotated)
    if bounds:
        x1, y1, x2, y2 = bounds
        # Clamp and add small padding (2%) to preserve card edges
        pw = max(0, int((x2 - x1) * 0.02))
        ph = max(0, int((y2 - y1) * 0.02))
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(rotated.shape[1], x2 + pw)
        y2 = min(rotated.shape[0], y2 + ph)
        cropped = rotated[y1:y2, x1:x2]
        logger.debug("Crop bounds: (%d,%d)-(%d,%d) size=%dx%d", x1, y1, x2, y2,
                     cropped.shape[1], cropped.shape[0])
    else:
        # Fallback: very conservative center crop (remove only obvious background)
        ch, cw = rotated.shape[:2]
        margin_x = int(cw * 0.01)
        margin_y = int(ch * 0.01)
        cropped = rotated[margin_y:ch - margin_y, margin_x:cw - margin_x]
        logger.debug("Fallback crop, no bounds found")

    # ── 5. Standardize ────────────────────────────────────────────
    final = postprocess(cropped)

    metrics = {
        "original_size": f"{w}x{h}",
        "cropped_size": f"{cropped.shape[1]}x{cropped.shape[0]}",
        "final_size": f"{final.shape[1]}x{final.shape[0]}",
        "confidence": round(confidence, 2),
    }

    return {
        "success": True,
        "rectified": final,
        "angle": round(angle, 2),
        "method": "deskew_validated" if do_rotate else "none",
        "metrics": metrics,
    }
