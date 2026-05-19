"""Card Geometry Detection & Auto-Correction Module.

Two-stage pipeline:
  1. SAM-based segmentation (pixel-perfect mask from YOLO bbox prompt)
  2. Mask → 4-corner quadrilateral + geometric measurements

Fallback chain:
  SAM → classical contours → YOLO bbox corners

Used by:
  - WebSocket stream router (real-time overlay, SAM every 5th frame)
  - Gallery router (upload processing, full SAM every time)
  - Capture router (final snapshot correction)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ID_ASPECT_RATIO = 85.60 / 53.98  # ≈ 1.586
TARGET_LONG = 856
TARGET_SHORT = 540
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
SAM_PATH = os.path.join(WEIGHTS_DIR, "sam_vit_b.pth")

# ── SAM Lazy Loader ───────────────────────────────────────────────
_sam_predictor: Any | None = None
_sam_available: bool | None = None


def _sam_is_available() -> bool:
    """Check once whether segment-anything is importable and weights exist."""
    global _sam_available
    if _sam_available is not None:
        return _sam_available
    try:
        from segment_anything import sam_model_registry, SamPredictor  # noqa: F401
        _sam_available = os.path.exists(SAM_PATH)
    except Exception:
        _sam_available = False
    if not _sam_available:
        logger.warning("SAM unavailable (install: pip install segment-anything; weights: %s)", SAM_PATH)
    return _sam_available


def _get_sam_predictor() -> Any:
    """Lazy-load SAM ViT-B predictor (CPU)."""
    global _sam_predictor
    if _sam_predictor is None and _sam_is_available():
        from segment_anything import sam_model_registry, SamPredictor
        logger.info("Loading SAM ViT-B from %s ...", SAM_PATH)
        sam = sam_model_registry["vit_b"](checkpoint=SAM_PATH)
        sam.to("cpu")
        _sam_predictor = SamPredictor(sam)
        logger.info("SAM loaded.")
    return _sam_predictor


# ── Data Structures ───────────────────────────────────────────────
@dataclass
class CardCorners:
    tl: tuple[float, float]
    tr: tuple[float, float]
    br: tuple[float, float]
    bl: tuple[float, float]

    def as_array(self) -> np.ndarray:
        return np.array([self.tl, self.tr, self.br, self.bl], dtype=np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "tl": [round(float(self.tl[0]), 1), round(float(self.tl[1]), 1)],
            "tr": [round(float(self.tr[0]), 1), round(float(self.tr[1]), 1)],
            "br": [round(float(self.br[0]), 1), round(float(self.br[1]), 1)],
            "bl": [round(float(self.bl[0]), 1), round(float(self.bl[1]), 1)],
        }


@dataclass
class GeometryResult:
    detected: bool
    corners: CardCorners | None
    angle: float
    offset_dx: float
    offset_dy: float
    skew_ratio: float
    coverage: float
    aspect_ratio: float
    issues: list[str]
    ready_to_capture: bool
    source: str = ""  # "sam", "contour", "bbox_fallback"
    mask: np.ndarray | None = None  # binary mask if SAM succeeded

    def to_dict(self, frame_w: int, frame_h: int) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "corners": self.corners.to_dict() if self.corners else None,
            "angle": round(self.angle, 1),
            "offset": {
                "dx": round(self.offset_dx, 1),
                "dy": round(self.offset_dy, 1),
            },
            "skew_ratio": round(self.skew_ratio, 3),
            "coverage": round(self.coverage, 3),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "issues": self.issues,
            "ready_to_capture": self.ready_to_capture,
            "source": self.source,
            "frame_size": [frame_w, frame_h],
        }


# ── Stage 1: SAM Segmentation ─────────────────────────────────────
def segment_card_sam(image_bgr: np.ndarray, bbox: list[int]) -> np.ndarray | None:
    """Run SAM with YOLO bbox as prompt. Returns binary mask or None."""
    predictor = _get_sam_predictor()
    if predictor is None:
        return None

    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        input_box = np.array(bbox)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(np.uint8) * 255
    except Exception as exc:
        logger.warning("SAM segmentation failed: %s", exc)
        return None


# ── Stage 2: Mask → Quadrilateral ───────────────────────────────
def mask_to_quad(mask: np.ndarray) -> np.ndarray | None:
    """Extract 4-corner quadrilateral from binary mask.

    Uses morphological cleanup, then approxPolyDP or minAreaRect fallback.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    card_contour = max(contours, key=cv2.contourArea)

    # Approach 1: approxPolyDP for exact polygon
    peri = cv2.arcLength(card_contour, True)
    approx = cv2.approxPolyDP(card_contour, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)

    # Approach 2: minAreaRect (always 4 corners, rotation-invariant)
    rect = cv2.minAreaRect(card_contour)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


# ── Classical CV Fallbacks ──────────────────────────────────────
def _preprocess(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """Grayscale → Blur → Canny → Dilate → Contours."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return gray, dilated, contours


def _detect_contour_quad(
    contours: list,
    frame_area: float,
    bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    """Find the largest contour that looks like a card and extract 4 corners."""
    min_area_ratio = 0.05 if bbox else 0.08
    candidates: list[tuple[np.ndarray, float]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * min_area_ratio:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 20:
            continue
        for eps_factor in [0.01, 0.015, 0.02, 0.03, 0.05]:
            approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
            if 4 <= len(approx) <= 8:
                pts = approx.reshape(-1, 2).astype(np.float32)
                if len(pts) > 4:
                    hull = cv2.convexHull(pts)
                    hull_peri = cv2.arcLength(hull, True)
                    approx2 = cv2.approxPolyDP(hull, 0.02 * hull_peri, True)
                    if len(approx2) == 4:
                        pts = approx2.reshape(4, 2).astype(np.float32)
                    else:
                        continue
                w = np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])
                h = np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])
                if w < 1 or h < 1:
                    continue
                aspect = max(w, h) / min(w, h)
                aspect_err = abs(aspect - ID_ASPECT_RATIO) / ID_ASPECT_RATIO
                if aspect_err > 0.45:
                    continue
                score = area / frame_area * (1 - aspect_err)
                candidates.append((pts, score))
                break

    if not candidates and contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area >= frame_area * min_area_ratio:
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            pts = box.astype(np.float32)
            w = np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])
            h = np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])
            if w > 1 and h > 1:
                aspect = max(w, h) / min(w, h)
                aspect_err = abs(aspect - ID_ASPECT_RATIO) / ID_ASPECT_RATIO
                if aspect_err <= 0.45:
                    score = area / frame_area * (1 - aspect_err)
                    candidates.append((pts, score))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def _order_corners(pts: np.ndarray) -> CardCorners:
    """Order 4 points as [TL, TR, BR, BL] using sum(x+y) and diff(x-y)."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(s)]
    rect[3] = pts[np.argmax(diff)]
    return CardCorners(
        tl=tuple(rect[0]),
        tr=tuple(rect[1]),
        br=tuple(rect[2]),
        bl=tuple(rect[3]),
    )


# ── Stage 3: Geometric Measurements ─────────────────────────────
def _compute_measurements(
    corners: CardCorners,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float, float, float]:
    """Return (angle, offset_dx, offset_dy, skew_ratio, coverage, aspect_ratio)."""
    tl = np.array(corners.tl)
    tr = np.array(corners.tr)
    br = np.array(corners.br)
    bl = np.array(corners.bl)

    angle = float(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0])))
    card_cx = (tl[0] + tr[0] + br[0] + bl[0]) / 4
    card_cy = (tl[1] + tr[1] + br[1] + bl[1]) / 4
    placeholder_cx = frame_w / 2
    placeholder_cy = frame_h / 2
    offset_dx = card_cx - placeholder_cx
    offset_dy = card_cy - placeholder_cy

    top_width = np.linalg.norm(tr - tl)
    bottom_width = np.linalg.norm(br - bl)
    skew_ratio = float(top_width / (bottom_width + 1e-6))

    card_area = 0.5 * abs(
        (tl[0]*tr[1] + tr[0]*br[1] + br[0]*bl[1] + bl[0]*tl[1])
        - (tl[1]*tr[0] + tr[1]*br[0] + br[1]*bl[0] + bl[1]*tl[0])
    )
    placeholder_w = frame_w * 0.75
    placeholder_h = placeholder_w / ID_ASPECT_RATIO
    placeholder_area = placeholder_w * placeholder_h
    coverage = float(card_area / placeholder_area)

    card_w = (top_width + bottom_width) / 2
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)
    card_h = (left_h + right_h) / 2
    aspect_ratio = float(card_w / (card_h + 1e-6))

    return angle, offset_dx, offset_dy, skew_ratio, coverage, aspect_ratio


def _build_issues(
    angle: float,
    offset_dx: float,
    offset_dy: float,
    skew_ratio: float,
    coverage: float,
    aspect_ratio: float,
    frame_w: int,
    frame_h: int,
) -> list[str]:
    issues: list[str] = []
    if abs(angle) >= 5:
        direction = "clockwise" if angle > 0 else "counter-clockwise"
        issues.append(f"Card is tilted {abs(angle):.1f}° {direction} — rotate it slightly")
    if abs(offset_dx) > frame_w * 0.08:
        direction = "right" if offset_dx > 0 else "left"
        issues.append(f"Move card to the {direction}")
    if abs(offset_dy) > frame_h * 0.08:
        direction = "down" if offset_dy > 0 else "up"
        issues.append(f"Move card {direction}")
    if not (0.85 < skew_ratio < 1.15):
        issues.append("Hold the camera directly above the card — avoid angling")
    if coverage < 0.75:
        issues.append("Move card closer to the camera")
    elif coverage > 1.10:
        issues.append("Move card further from the camera")
    aspect_err = abs(aspect_ratio - ID_ASPECT_RATIO) / ID_ASPECT_RATIO
    if aspect_err > 0.15:
        issues.append("Card edges not fully visible — check all 4 corners are in frame")
    return issues


# ── Main Analysis with Fallback Chain ───────────────────────────
def analyze_frame(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
    use_sam: bool = True,
) -> GeometryResult:
    """Analyze a single frame and return geometry measurements.

    Fallback chain:
      1. SAM segmentation → mask_to_quad
      2. Classical Canny contours → approxPolyDP/minAreaRect
      3. YOLO bbox corners (rectangular approximation)

    Args:
        frame: BGR image
        bbox: Optional YOLO bbox (x1, y1, x2, y2) to narrow search
        use_sam: Whether to attempt SAM (disabled for stream intermediate frames)
    """
    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_h * frame_w

    # ── Try SAM first ────────────────────────────────────────────
    mask: np.ndarray | None = None
    pts: np.ndarray | None = None
    source = "none"

    if use_sam and bbox is not None:
        mask = segment_card_sam(frame, list(bbox))
        if mask is not None:
            pts = mask_to_quad(mask)
            if pts is not None:
                source = "sam"
                logger.debug("SAM found quad from mask")

    # ── Fallback: classical contours ──────────────────────────
    if pts is None:
        gray, edges, contours = _preprocess(frame)
        if bbox is not None:
            bx1, by1, bx2, by2 = bbox
            filtered = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                        filtered.append(cnt)
            contours = filtered if filtered else contours
        pts = _detect_contour_quad(contours, frame_area, bbox)
        if pts is not None:
            source = "contour"

    # ── Final fallback: YOLO bbox corners ─────────────────────
    if pts is None and bbox is not None:
        bx1, by1, bx2, by2 = bbox
        pts = np.array([
            [bx1, by1],
            [bx2, by1],
            [bx2, by2],
            [bx1, by2],
        ], dtype=np.float32)
        source = "bbox_fallback"

    if pts is None:
        return GeometryResult(
            detected=False,
            corners=None,
            angle=0.0,
            offset_dx=0.0,
            offset_dy=0.0,
            skew_ratio=1.0,
            coverage=0.0,
            aspect_ratio=1.0,
            issues=["No card detected — ensure all 4 corners are visible"],
            ready_to_capture=False,
            source="none",
            mask=None,
        )

    corners = _order_corners(pts)
    angle, offset_dx, offset_dy, skew_ratio, coverage, aspect_ratio = _compute_measurements(
        corners, frame_w, frame_h
    )
    issues = _build_issues(angle, offset_dx, offset_dy, skew_ratio, coverage, aspect_ratio, frame_w, frame_h)
    ready = len(issues) == 0

    logger.debug("Geometry source: %s angle=%.1f ready=%s", source, angle, ready)

    return GeometryResult(
        detected=True,
        corners=corners,
        angle=angle,
        offset_dx=offset_dx,
        offset_dy=offset_dy,
        skew_ratio=skew_ratio,
        coverage=coverage,
        aspect_ratio=aspect_ratio,
        issues=issues,
        ready_to_capture=ready,
        source=source,
        mask=mask,
    )


# ── Stage 4: Perspective Auto-Correction ──────────────────────
def correct_perspective(
    frame: np.ndarray,
    corners: CardCorners,
    target_w: int = TARGET_LONG,
    target_h: int = TARGET_SHORT,
) -> np.ndarray:
    """Apply perspective transform to produce a flat, rectangular card image."""
    src_pts = corners.as_array()
    dst_pts = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(
        frame, M, (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected


# ── Visual Debug Overlay ────────────────────────────────────────
def draw_geometry_overlay(
    frame: np.ndarray,
    result: GeometryResult,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Draw actual card outline (from mask if available), corners, brackets, and issue text."""
    overlay = frame.copy()
    if not result.detected or result.corners is None:
        cv2.putText(overlay, "NO CARD", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return overlay

    LINE_COLOR = (0, 230, 255)       # bright cyan
    CORNER_COLOR = (0, 150, 255)     # darker cyan
    DOT_COLOR = (50, 255, 50)        # green
    RED = (0, 0, 255)
    GREEN = (50, 255, 50)
    YELLOW = (0, 200, 255)

    # Edge color based on state
    if result.ready_to_capture:
        edge_color = GREEN
    elif result.issues:
        edge_color = RED
    else:
        edge_color = YELLOW

    # ── Draw actual card outline from SAM mask (if available) ──
    draw_mask = mask if mask is not None else result.mask
    if draw_mask is not None:
        contours, _ = cv2.findContours(draw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, LINE_COLOR, thickness=3)
    else:
        # Fallback: polygon between 4 corners
        pts = np.array([result.corners.tl, result.corners.tr,
                        result.corners.br, result.corners.bl], dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=LINE_COLOR, thickness=3)

    # ── Corner dots ────────────────────────────────────────────
    corners_arr = np.array([result.corners.tl, result.corners.tr,
                            result.corners.br, result.corners.bl], dtype=np.int32)
    for (cx, cy) in corners_arr:
        cv2.circle(overlay, (int(cx), int(cy)), 8, DOT_COLOR, -1)
        cv2.circle(overlay, (int(cx), int(cy)), 8, (255, 255, 255), 2)

    # ── Outward bracket arms at each corner ────────────────────
    n = len(corners_arr)
    for i in range(n):
        prev = corners_arr[(i - 1) % n]
        curr = corners_arr[i]
        nxt = corners_arr[(i + 1) % n]
        cx, cy = int(curr[0]), int(curr[1])
        v1 = np.array(prev) - np.array(curr)
        v2 = np.array(nxt) - np.array(curr)
        n1 = v1 / (np.linalg.norm(v1) + 1e-6)
        n2 = v2 / (np.linalg.norm(v2) + 1e-6)
        outward = -(n1 + n2)
        outward = outward / (np.linalg.norm(outward) + 1e-6)
        end = np.array([cx, cy]) + outward * 30
        perp = np.array([-outward[1], outward[0]])
        arm1 = end + perp * 18
        arm2 = end - perp * 18
        cv2.line(overlay, (cx, cy), tuple(end.astype(int)), CORNER_COLOR, 3)
        cv2.line(overlay, tuple(end.astype(int)), tuple(arm1.astype(int)), CORNER_COLOR, 3)
        cv2.line(overlay, tuple(end.astype(int)), tuple(arm2.astype(int)), CORNER_COLOR, 3)

    # ── Issue text ─────────────────────────────────────────────
    y0 = 30
    for issue in result.issues[:3]:
        cv2.putText(overlay, issue, (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
        y0 += 25

    if result.ready_to_capture:
        cv2.putText(overlay, "READY", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 3)
    elif result.source == "sam":
        cv2.putText(overlay, "SAM MASK", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_COLOR, 2)

    return overlay
