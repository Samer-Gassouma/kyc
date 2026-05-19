"""ROI-based field extraction for Tunisian CIN cards.

Runs on the SAM-corrected flat card (target 856×540).
All ROIs are defined as relative percentages so they remain correct
regardless of the actual input resolution before correction.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TARGET_W, TARGET_H = 856, 540

# ── Front Side ROI Map ──────────────────────────────────────────
FRONT_ROIS: dict[str, dict[str, Any]] = {
    "photo": {
        "box": (0.01, 0.30, 0.32, 0.95),
        "lang": None,
        "label": "Photo",
    },
    "id_number": {
        "box": (0.34, 0.31, 0.78, 0.43),
        "lang": ["en"],
        "label": "ID Number",
    },
    "last_name": {
        "box": (0.35, 0.45, 0.89, 0.57),
        "lang": ["ar"],
        "label": "اللقب",
    },
    "first_name": {
        "box": (0.37, 0.57, 0.89, 0.66),
        "lang": ["ar"],
        "label": "الاسم",
    },
    "father_lineage": {
        "box": (0.35, 0.67, 0.99, 0.76),
        "lang": ["ar"],
        "label": "النسب",
    },
    "date_of_birth": {
        "box": (0.35, 0.76, 0.82, 0.85),
        "lang": ["ar", "en"],
        "label": "تاريخ الولادة",
    },
    "place_of_birth": {
        "box": (0.35, 0.86, 0.91, 0.97),
        "lang": ["ar"],
        "label": "مكانها",
    },
}

# ── Back Side ROI Map ───────────────────────────────────────────
BACK_ROIS: dict[str, dict[str, Any]] = {
    "mother_name": {
        "box": (0.28, 0.03, 0.79, 0.17),
        "lang": ["ar"],
        "label": "اسم ولقب الأم",
    },
    "profession": {
        "box": (0.20, 0.18, 0.59, 0.29),
        "lang": ["ar"],
        "label": "المهنة",
    },
    "address_line1": {
        "box": (0.02, 0.30, 0.56, 0.41),
        "lang": ["ar", "en"],
        "label": "العنوان",
    },
    "address_line2": {
        "box": (0.02, 0.42, 0.68, 0.52),
        "lang": ["ar"],
        "label": "المدينة",
    },
    "issue_date": {
        "box": (0.00, 0.53, 0.33, 0.64),
        "lang": ["ar", "en"],
        "label": "تاريخ الإصدار",
    },
    # blood_type → removed
    # fingerprint: x > 0.62 → excluded
    # serial 20130044 → excluded
    # barcode → excluded
}

# ── Arabic month name → number mapping ─────────────────────────
# Includes common OCR misspellings (EasyOCR on CIN font is noisy)
ARABIC_MONTHS = {
    "جانفي": "01",
    "يناير": "01",
    "فيفري": "02",
    "فبراير": "02",
    "مارس": "03",
    "أفريل": "04",
    "ابريل": "04",
    "نيسان": "04",
    "ماي": "05",
    "مايو": "05",
    "جوان": "06",
    "يونيو": "06",
    "جويلية": "07",
    "يوليو": "07",
    "اوت": "08",
    "أوت": "08",
    "اغسطس": "08",
    "سبتمبر": "09",
    "سىتمبر": "09",  # fuzzy variant
    "أكتوبر": "10",
    "اكتوبر": "10",
    "نوفمبر": "11",
    "ديسمبر": "12",
    "دسسد": "12",  # fuzzy variant
}


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def fuzzy_match_month(text: str) -> str | None:
    """Find closest Arabic month name in text (exact → fuzzy fallback)."""
    # Exact / known misspelling match first
    for name, num in ARABIC_MONTHS.items():
        if name in text:
            return num

    # Fuzzy fallback: find best (lowest Levenshtein distance) match
    best_num = None
    best_dist = 999
    tl = len(text)
    for name, num in ARABIC_MONTHS.items():
        nl = len(name)
        for i in range(tl):
            for j in range(max(1, nl - 2), nl + 3):
                if i + j > tl:
                    break
                substr = text[i : i + j]
                dist = _levenshtein(substr, name)
                if dist < best_dist:
                    best_dist = dist
                    best_num = num
    return best_num if best_dist <= 2 else None


FIELD_LABEL_PREFIXES = [
    "اللقب",
    "الاسم",
    "المهنة",
    "المهنه",
    "العنوان",
    "تاريخ الولادة",
    "مكانها",
    "ترش في",
    "تارخ الولادة",
    "اسم ولقب الأم",
    "النسب",
    "مكانه",
    "مكاتها",
]


def clean_arabic_noise(text: str) -> str:
    """
    Remove common EasyOCR noise from Arabic ID card text:
    - Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩) that aren't part of real content
    - Lone punctuation and special chars (؛ ، ! ؟ ~ = { } [ ] _ ` ')
    - Repeated whitespace
    - Leading/trailing garbage characters
    """
    # Remove Arabic-Indic digit sequences (OCR noise on background patterns)
    text = re.sub(r"[٠١٢٣٤٥٦٧٨٩]+", "", text)

    # Remove noise punctuation
    text = re.sub(r"[؛،!؟~={}\[\]_`'\"\\|/@#$%^&*]", "", text)

    # Remove lone single Arabic letters surrounded by spaces (OCR artifacts)
    text = re.sub(r"(?<!\w)[\u0600-\u06FF](?!\w)", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def extract_best_arabic_name(text: str) -> str:
    """
    From a noisy OCR string, extract the longest contiguous Arabic word sequence.
    Ignores digits, punctuation, and short fragments.
    Example: "اذرا ا قسومة ؛١٨١٨؛" → "قسومة"
    Example: "ر ن ٧٢ بن رضا بن علي ا ا" → "بن رضا بن علي"
    """
    text = clean_arabic_noise(text)
    # Extract all Arabic word sequences (2+ chars)
    arabic_words = re.findall(r"[\u0600-\u06FF]{2,}(?:\s+[\u0600-\u06FF]{2,})*", text)

    if not arabic_words:
        return text.strip()

    # Return the longest sequence found, stripping leading short fragments
    result = max(arabic_words, key=len).strip()
    # Strip leading 1-2 char words (OCR noise like "ا", "اذ")
    result = re.sub(r"^(?:[\u0600-\u06FF]{1,2}\s+)+", "", result).strip()
    # Strip known OCR garbage fragments that appear at start
    for garbage in ("اذرا ", "انا ", "ا ", "ال "):
        if result.startswith(garbage):
            result = result[len(garbage) :].strip()
    return result


def parse_arabic_date(text: str) -> str:
    """
    Extract date from noisy Arabic OCR string.
    Handles: "24 سبتمبر 2002 ١١٠١١٠ ٠١٠" → "2002-09-24"
    Strategy:
      1. Find 4-digit Western year (1900-2099)
      2. Find Arabic month name (with fuzzy variants)
      3. Find 1-2 digit Western day
    """
    # Step 1: extract 4-digit year (Western digits only)
    year_match = re.search(r"\b(19|20)\d{2}\b", text)
    if not year_match:
        return clean_arabic_noise(text)
    year = year_match.group(0)

    # Step 2: find month (exact first, fuzzy fallback)
    month_num = None
    for ar_month, num in ARABIC_MONTHS.items():
        if ar_month in text:
            month_num = num
            break
    if not month_num:
        month_num = fuzzy_match_month(text)

    if not month_num:
        return clean_arabic_noise(text)

    # Step 3: find day (1-2 Western digits, not the year)
    # Remove the year first to avoid confusion
    text_no_year = text.replace(year, "")
    day_match = re.search(r"\b(\d{1,2})\b", text_no_year)
    day = day_match.group(1).zfill(2) if day_match else "01"

    return f"{year}-{month_num}-{day}"


def preprocess_roi(
    roi: np.ndarray, lang: list[str] | None, roi_key: str = ""
) -> np.ndarray:
    """Adaptive preprocessing per ROI before OCR."""
    h, w = roi.shape[:2]
    upscaled = cv2.resize(roi, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    mean_brightness = float(np.mean(gray))
    if mean_brightness < 127:
        gray = cv2.bitwise_not(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Dedicated boost for father_lineage — sits on watermark/monument noise
    if roi_key == "father_lineage":
        alpha = 1.8  # contrast multiplier
        beta = -30  # darken background
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if lang and "ar" in lang:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.dilate(binary, kernel, iterations=1)

    return binary


def extract_roi(
    card: np.ndarray,
    roi_key: str,
    roi_def: dict[str, Any],
    reader: Any,
) -> dict[str, Any]:
    """Crop, preprocess, and OCR a single ROI."""
    h, w = card.shape[:2]
    x1p, y1p, x2p, y2p = roi_def["box"]
    x1, y1 = int(x1p * w), int(y1p * h)
    x2, y2 = int(x2p * w), int(y2p * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = card[y1:y2, x1:x2]

    if crop.size == 0:
        return {
            "field": roi_key,
            "label": roi_def["label"],
            "raw_text": "",
            "confidence": 0.0,
            "bbox": [x1, y1, x2, y2],
        }

    if roi_def.get("lang") is None:
        return {
            "field": roi_key,
            "label": roi_def["label"],
            "raw_text": "__IMAGE__",
            "confidence": 1.0,
            "bbox": [x1, y1, x2, y2],
            "crop": crop,
        }

    processed = preprocess_roi(crop, roi_def.get("lang"), roi_key=roi_key)

    try:
        results = reader.readtext(
            processed,
            detail=1,
            paragraph=False,
            text_threshold=0.5,
            low_text=0.3,
        )
    except Exception as exc:
        logger.error("EasyOCR failed on %s: %s", roi_key, exc)
        results = []

    if not results:
        return {
            "field": roi_key,
            "label": roi_def["label"],
            "raw_text": "",
            "confidence": 0.0,
            "bbox": [x1, y1, x2, y2],
        }

    full_text = " ".join(r[1] for r in results).strip()
    avg_conf = float(np.mean([r[2] for r in results]))

    return {
        "field": roi_key,
        "label": roi_def["label"],
        "raw_text": full_text,
        "confidence": round(avg_conf, 3),
        "bbox": [x1, y1, x2, y2],
    }


def parse_fields(raw: list[dict[str, Any]], side: str) -> dict[str, Any]:
    """Clean and structure raw OCR output."""
    data: dict[str, Any] = {}

    for result in raw:
        field = result["field"]
        text = result.get("raw_text", "").strip()

        if not text or text == "__IMAGE__":
            if field == "photo":
                pass  # handled separately
            continue

        # Strip field label prefix if OCR included it
        for prefix in FIELD_LABEL_PREFIXES:
            if prefix in text:
                text = text[text.index(prefix) + len(prefix) :].lstrip(": ").strip()
                break

        if field == "id_number":
            # Only keep Western digits, must be 8
            digits = re.sub(r"\D", "", text)
            # Take first 8-digit sequence
            match = re.search(r"\d{8}", digits)
            data["id_number"] = match.group(0) if match else digits[:8]
            data["id_number_valid"] = bool(
                re.match(r"^\d{8}$", data.get("id_number", ""))
            )

        elif field in (
            "last_name",
            "first_name",
            "place_of_birth",
            "mother_name",
            "profession",
        ):
            # Extract cleanest Arabic name sequence
            data[field] = extract_best_arabic_name(text)

        elif field == "father_lineage":
            # Must contain بن — extract the بن ... بن ... pattern
            cleaned = extract_best_arabic_name(text)
            # Ensure it starts with بن if present
            bn_match = re.search(r"بن[\s\u0600-\u06FF]+", cleaned)
            data[field] = bn_match.group(0).strip() if bn_match else cleaned

        elif field == "date_of_birth":
            data[field] = parse_arabic_date(text)

        elif field == "issue_date":
            # Try ISO first (already parsed upstream)
            if re.match(r"\d{4}-\d{2}-\d{2}", text):
                data[field] = text
            else:
                data[field] = parse_arabic_date(text)

        elif field in ("address_line1", "address_line2"):
            # Combine, remove label noise, keep Arabic + Western digits
            line = clean_arabic_noise(text)
            # Remove stray short words like "عدل" (OCR artifact for "العنوان")
            line = re.sub(r"\bعدل\b", "", line).strip()
            existing = data.get("address", "")
            data["address"] = (existing + " " + line).strip()

        else:
            data[field] = clean_arabic_noise(text)

    return data


def extract_barcode(card: np.ndarray) -> dict:
    """
    Extract and decode the barcode from the bottom of the back card.
    Barcode zone: bottom 15% of card, full width.
    """
    h, w = card.shape[:2]
    barcode_crop = card[int(h * 0.82) : int(h * 0.97), 0:w]

    upscaled = cv2.resize(
        barcode_crop, (w * 2, int(h * 0.15 * 2)), interpolation=cv2.INTER_LANCZOS4
    )
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    from pyzbar import pyzbar

    barcodes = pyzbar.decode(gray)

    if not barcodes:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        barcodes = pyzbar.decode(binary)

    if barcodes:
        raw = barcodes[0].data.decode("utf-8", errors="ignore")
        return {
            "barcode_raw": raw,
            "barcode_type": barcodes[0].type,
            "barcode_decoded": parse_cin_barcode(raw),
        }

    return {"barcode_raw": None, "barcode_type": None, "barcode_decoded": None}


def parse_cin_barcode(raw: str) -> dict:
    """
    Parse the Tunisian CIN barcode content.
    Format: id_number(8) + issue_date(8, DDMMYYYY or YYYYMMDD) + suffix
    """
    result: dict[str, Any] = {}
    result["raw"] = raw

    # Extract 8-digit ID number (first 8 digits are typically the CIN)
    id_match = re.search(r"^(\d{8})", raw)
    if id_match:
        result["id_number"] = id_match.group(1)

    # Look for a valid date after the ID number portion
    # Try DDMMYYYY pattern (common for Tunisian cards)
    tail = raw[8:] if len(raw) > 8 else ""
    ddmmyyyy = re.search(r"(\d{2})(\d{2})(\d{4})", tail)
    if ddmmyyyy:
        day, month, year = ddmmyyyy.group(1), ddmmyyyy.group(2), ddmmyyyy.group(3)
        if 1 <= int(day) <= 31 and 1 <= int(month) <= 12 and 1990 <= int(year) <= 2035:
            result["issue_date"] = f"{year}-{month}-{day}"
            return result

    # Try YYYYMMDD pattern
    yyyymmdd = re.search(r"(19\d{2}|20\d{2})(\d{2})(\d{2})", tail)
    if yyyymmdd:
        year, month, day = yyyymmdd.group(1), yyyymmdd.group(2), yyyymmdd.group(3)
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            result["issue_date"] = f"{year}-{month}-{day}"
            return result

    return result


def extract_card_fields(
    corrected_card: np.ndarray,
    side: str,
    reader: Any,
) -> dict[str, Any]:
    """Run ROI extraction on the SAM-corrected flat card.

    Uses batched EasyOCR: all OCR ROIs are sent in a single call
    instead of one-by-one, cutting OCR time by ~3x.
    """
    roi_map = FRONT_ROIS if side == "front" else BACK_ROIS

    h, w = corrected_card.shape[:2]
    if (w, h) != (TARGET_W, TARGET_H):
        corrected_card = cv2.resize(
            corrected_card,
            (TARGET_W, TARGET_H),
            interpolation=cv2.INTER_LANCZOS4,
        )

    face_crop: np.ndarray | None = None

    # Phase 1: crop and preprocess all ROIs
    ocr_batch: list[np.ndarray] = []
    ocr_meta: list[dict[str, Any]] = []
    photo_crop: np.ndarray | None = None

    for roi_key, roi_def in roi_map.items():
        x1p, y1p, x2p, y2p = roi_def["box"]
        x1, y1 = int(x1p * w), int(y1p * h)
        x2, y2 = int(x2p * w), int(y2p * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = corrected_card[y1:y2, x1:x2]

        if crop.size == 0:
            ocr_batch.append(np.zeros((32, 32), dtype=np.uint8))
            ocr_meta.append(
                {
                    "field": roi_key,
                    "label": roi_def["label"],
                    "bbox": [x1, y1, x2, y2],
                    "lang": roi_def.get("lang"),
                }
            )
            continue

        if roi_def.get("lang") is None:
            photo_crop = crop
            ocr_batch.append(np.zeros((32, 32), dtype=np.uint8))
            ocr_meta.append(
                {
                    "field": roi_key,
                    "label": roi_def["label"],
                    "bbox": [x1, y1, x2, y2],
                    "lang": None,
                    "crop": crop,
                }
            )
            continue

        processed = preprocess_roi(crop, roi_def.get("lang"), roi_key=roi_key)
        ocr_batch.append(processed)
        ocr_meta.append(
            {
                "field": roi_key,
                "label": roi_def["label"],
                "bbox": [x1, y1, x2, y2],
                "lang": roi_def.get("lang"),
            }
        )

    # Phase 2: batch OCR on all text ROIs at once
    try:
        batch_results = reader.readtext(
            ocr_batch,
            detail=1,
            paragraph=False,
            text_threshold=0.5,
            low_text=0.3,
        )
    except Exception as exc:
        logger.error("Batch EasyOCR failed: %s", exc)
        batch_results = [[] for _ in ocr_meta]

    # Phase 3: build structured results per ROI
    raw_results: list[dict[str, Any]] = []
    for meta, roi_ocr in zip(ocr_meta, batch_results):
        roi_key = meta["field"]
        label = meta["label"]
        bbox = meta["bbox"]

        if meta.get("lang") is None:
            result = {
                "field": roi_key,
                "label": label,
                "raw_text": "__IMAGE__",
                "confidence": 1.0,
                "bbox": bbox,
            }
            if meta.get("crop") is not None:
                result["crop"] = meta["crop"]
                photo_crop = meta["crop"]
            raw_results.append(result)
            continue

        if not roi_ocr:
            raw_results.append(
                {
                    "field": roi_key,
                    "label": label,
                    "raw_text": "",
                    "confidence": 0.0,
                    "bbox": bbox,
                }
            )
            continue

        full_text = " ".join(r[1] for r in roi_ocr).strip()
        avg_conf = float(np.mean([r[2] for r in roi_ocr]))
        raw_results.append(
            {
                "field": roi_key,
                "label": label,
                "raw_text": full_text,
                "confidence": round(avg_conf, 3),
                "bbox": bbox,
            }
        )

    if photo_crop is not None:
        face_crop = photo_crop
        logger.info("[roi] photo crop extracted: shape=%s", face_crop.shape)

    # Fallback: if photo ROI missed, try face detection on the whole card
    if side == "front" and face_crop is None:
        logger.info("[roi] photo ROI missed, trying face detection fallback")
        face_crop = _detect_and_crop_face(corrected_card)
        if face_crop is not None:
            logger.info(
                "[roi] face detection fallback succeeded: shape=%s", face_crop.shape
            )

    structured = parse_fields(raw_results, side)

    if side == "back":
        barcode_result = extract_barcode(corrected_card)
        structured["barcode"] = barcode_result

        bc_decoded = barcode_result.get("barcode_decoded") or {}
        if bc_decoded.get("id_number"):
            structured["id_number_from_barcode"] = bc_decoded["id_number"]
        if bc_decoded.get("issue_date"):
            structured["issue_date"] = bc_decoded["issue_date"]

    return {
        "side": side,
        "fields": structured,
        "raw_ocr": [{k: v for k, v in r.items() if k != "crop"} for r in raw_results],
        "face_crop": face_crop,
    }


def _detect_and_crop_face(card: np.ndarray) -> np.ndarray | None:
    """Detect a face in the corrected card and crop it as fallback."""
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

    # Try Haar cascade first (usually available with OpenCV)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        if len(faces) > 0:
            # Pick the largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            # Add 20% padding
            pad_x = int(w * 0.2)
            pad_y = int(h * 0.3)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(card.shape[1], x + w + pad_x)
            y2 = min(card.shape[0], y + h + pad_y)
            return card[y1:y2, x1:x2]
    except Exception as exc:
        logger.warning("Haar face detection failed: %s", exc)

    # Fallback: skin-tone blob detection in left half of card
    try:
        h, w = card.shape[:2]
        left_half = card[:, : w // 2]
        hsv = cv2.cvtColor(left_half, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([25, 180, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        # Find largest skin blob
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                x, y, bw, bh = cv2.boundingRect(largest)
                pad = int(max(bw, bh) * 0.2)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w // 2, x + bw + pad)
                y2 = min(h, y + bh + pad)
                return left_half[y1:y2, x1:x2]
    except Exception as exc:
        logger.warning("Skin-tone fallback failed: %s", exc)

    return None


def draw_roi_debug(card: np.ndarray, side: str) -> np.ndarray:
    """Draw all ROI boxes on the card for visual verification."""
    debug = card.copy()
    roi_map = FRONT_ROIS if side == "front" else BACK_ROIS
    h, w = debug.shape[:2]

    colors: dict[str, tuple[int, int, int]] = {
        "id_number": (0, 255, 0),
        "photo": (255, 0, 0),
        "last_name": (0, 165, 255),
        "first_name": (0, 165, 255),
        "date_of_birth": (255, 255, 0),
        "address_line1": (255, 0, 255),
        "address_line2": (255, 0, 255),
        "issue_date": (128, 0, 128),
        "mother_name": (0, 255, 128),
        "profession": (128, 128, 0),
    }

    for key, roi in roi_map.items():
        if roi.get("lang") is None:
            color = (255, 0, 0)
        else:
            color = colors.get(key, (200, 200, 200))
        x1p, y1p, x2p, y2p = roi["box"]
        x1, y1 = int(x1p * w), int(y1p * h)
        x2, y2 = int(x2p * w), int(y2p * h)
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            debug, key, (x1 + 4, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )

    return debug
