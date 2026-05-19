"""
OCR post-processing cleaner using mT5-small.
Takes noisy EasyOCR output per field and returns clean structured text.
Runs fully locally on CPU, ~300MB RAM, ~100ms per field.
"""

from __future__ import annotations
import re
import logging
from typing import Any
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Use AraT5 for Arabic fields, mT5-small for mixed fields
ARABIC_MODEL = "UBC-NLP/AraT5-base"   # ~850MB — good Arabic understanding
SMALL_MODEL = "google/mt5-small"      # ~300MB — multilingual fallback

_model = None
_tokenizer = None


FINETUNED_MODEL = "/home/ivan/kyc/backend/weights/ocr_cleaner_finetuned"

def _load_model():
    global _model, _tokenizer
    if _model is None:
        import os
        if os.path.exists(FINETUNED_MODEL):
            logger.info("Loading fine-tuned OCR cleaner model...")
            _tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
            _model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_MODEL)
        else:
            logger.info("Loading base OCR cleaner model (mT5-small)...")
            _tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL)
            _model = AutoModelForSeq2SeqLM.from_pretrained(SMALL_MODEL)
        _model.eval()
        logger.info("OCR cleaner model loaded")
    return _model, _tokenizer


# Field-specific prompt templates
FIELD_PROMPTS = {
    "last_name": (
        "arabic",
        "استخرج اسم العائلة فقط من النص التالي بدون أي رموز أو أرقام: {text}"
    ),
    "first_name": (
        "arabic",
        "استخرج الاسم الشخصي فقط من النص التالي بدون أي رموز أو أرقام: {text}"
    ),
    "father_lineage": (
        "arabic",
        "استخرج سلسلة النسب التي تبدأ بـ'بن' من النص التالي: {text}"
    ),
    "date_of_birth": (
        "date",
        "استخرج تاريخ الميلاد من النص التالي واكتبه بصيغة YYYY-MM-DD: {text}"
    ),
    "place_of_birth": (
        "arabic",
        "استخرج اسم المدينة أو المكان فقط من النص التالي: {text}"
    ),
    "mother_name": (
        "arabic",
        "استخرج اسم الأم فقط من النص التالي بدون أي رموز: {text}"
    ),
    "profession": (
        "arabic",
        "استخرج المهنة فقط من النص التالي: {text}"
    ),
    "address": (
        "mixed",
        "نظف العنوان التالي واحذف الرموز الزائدة مع الإبقاء على الأرقام والكلمات: {text}"
    ),
    "issue_date": (
        "date",
        "استخرج تاريخ الإصدار من النص التالي واكتبه بصيغة YYYY-MM-DD: {text}"
    ),
}


def clean_field(field: str, noisy_text: str, max_new_tokens: int = 32) -> str:
    """
    Clean a single OCR field using the language model.
    Returns cleaned text or original if model fails.
    """
    if not noisy_text or noisy_text == "__IMAGE__":
        return noisy_text

    if field not in FIELD_PROMPTS:
        return _fallback_clean(noisy_text)

    field_type, prompt_template = FIELD_PROMPTS[field]
    prompt = prompt_template.format(text=noisy_text)

    try:
        model, tokenizer = _load_model()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,           # beam search for better quality
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Post-validate based on field type
        return _validate_output(field, field_type, result, noisy_text)

    except Exception as e:
        logger.error("Model clean failed for %s: %s", field, e)
        return _fallback_clean(noisy_text)


def _validate_output(field: str, field_type: str, result: str, original: str) -> str:
    """
    Validate model output makes sense — reject if clearly wrong.
    Falls back to regex cleaner if output is garbage.
    """
    if field_type == "date":
        # Must match YYYY-MM-DD
        if re.match(r"\d{4}-\d{2}-\d{2}", result):
            return result
        # Try to extract from result
        match = re.search(r"\d{4}-\d{2}-\d{2}", result)
        return match.group(0) if match else _fallback_clean(original)

    if field_type == "arabic":
        # Must contain Arabic characters
        if re.search(r"[\u0600-\u06FF]{2,}", result):
            return result
        # Model output has no Arabic — fall back
        return _fallback_clean(original)

    return result  # mixed fields — trust the model


def _fallback_clean(text: str) -> str:
    """
    Lightweight regex fallback when model fails.
    Extracts longest Arabic sequence or cleans noise.
    """
    # Remove Arabic-Indic numerals and noise punctuation
    text = re.sub(r"[٠١٢٣٤٥٦٧٨٩]+", "", text)
    text = re.sub(r"[؛،!؟~={}\[\]_`'\"]", "", text)
    # Extract longest Arabic word sequence
    arabic_seqs = re.findall(
        r"[\u0600-\u06FF]{2,}(?:\s+[\u0600-\u06FF]{2,})*", text
    )
    if arabic_seqs:
        return max(arabic_seqs, key=len).strip()
    return re.sub(r"\s+", " ", text).strip()


def clean_all_fields(raw_fields: dict[str, Any]) -> dict[str, Any]:
    """
    Run cleaner on all fields in parallel using ThreadPoolExecutor.
    ID number and barcode fields are skipped (they're already clean digits).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    SKIP_FIELDS = {"id_number", "id_number_valid", "barcode_raw",
                   "barcode_decoded", "barcode_type", "photo"}

    cleaned = dict(raw_fields)
    fields_to_clean = {
        k: v for k, v in raw_fields.items()
        if k not in SKIP_FIELDS and isinstance(v, str) and v
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(clean_field, field, text): field
            for field, text in fields_to_clean.items()
        }
        for future in as_completed(futures):
            field = futures[future]
            try:
                cleaned[field] = future.result()
            except Exception as e:
                logger.error("Clean failed for %s: %s", field, e)

    return cleaned
