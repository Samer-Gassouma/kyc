"""Server-side image quality checks (Laplacian blur, glare, centering)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def check_quality(image: np.ndarray) -> dict[str, Any]:
    """Run quality checks on a full-resolution image.

    Returns dict with blur_score, glare_score, brightness, and pass/fail.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Blur: Laplacian variance (higher = sharper)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_ok = lap_var > 8.0  # very relaxed for real-world phone photos

    # Glare: detect bright hotspots
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    glare_ratio = float(np.sum(thresh > 0)) / (gray.shape[0] * gray.shape[1])
    glare_ok = glare_ratio < 0.10  # relaxed for normal lighting

    # Overall brightness
    mean_val = float(np.mean(gray))
    brightness_ok = 40.0 < mean_val < 230.0  # wider range for various lighting

    passed = blur_ok and glare_ok and brightness_ok

    issues: list[str] = []
    if not blur_ok:
        issues.append("Image is blurry")
    if not glare_ok:
        issues.append("Glare detected on document")
    if not brightness_ok:
        if mean_val <= 60:
            issues.append("Image too dark")
        else:
            issues.append("Image too bright / overexposed")

    return {
        "blur_score": round(lap_var, 2),
        "blur_ok": blur_ok,
        "glare_ratio": round(glare_ratio, 4),
        "glare_ok": glare_ok,
        "mean_brightness": round(mean_val, 2),
        "brightness_ok": brightness_ok,
        "quality_passed": passed,
        "issues": issues,
    }
