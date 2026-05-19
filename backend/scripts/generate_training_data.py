"""
Synthetic training data generator for OCR cleaner fine-tuning.
Generates (noisy_input, clean_output) pairs per field type.
"""

import random
import json
import re


# Noise injection functions
def inject_arabic_indic(text: str, prob: float = 0.3) -> str:
    """Randomly insert Arabic-Indic digit noise."""
    indic = "٠١٢٣٤٥٦٧٨٩"
    result = []
    for char in text:
        result.append(char)
        if random.random() < prob:
            result.append(random.choice(indic))
    return "".join(result)


def inject_punctuation_noise(text: str, prob: float = 0.2) -> str:
    noise_chars = "؛،!؟~=_`'|"
    result = []
    for char in text:
        if random.random() < prob:
            result.append(random.choice(noise_chars))
        result.append(char)
    return "".join(result)


def inject_label_prefix(text: str, field: str) -> str:
    labels = {
        "last_name": ["اللقب:", "اللقب :", "الفب:"],
        "first_name": ["الاسم:", "الامد:", "الام:"],
        "father_lineage": ["النسب:", "ا النسب"],
        "date_of_birth": ["تاريخ الولادة:", "تارخ الولادة:"],
        "place_of_birth": ["مكانها:", "مكاتها:", "جانها:"],
    }
    prefix_list = labels.get(field, [])
    if prefix_list and random.random() < 0.5:
        return random.choice(prefix_list) + " " + text
    return text


def add_noise(text: str, field: str) -> str:
    text = inject_label_prefix(text, field)
    text = inject_arabic_indic(text, prob=0.2)
    text = inject_punctuation_noise(text, prob=0.15)
    return text


# Clean ground truth samples — expand this list with real card data
CLEAN_SAMPLES = {
    "last_name": [
        "قسومة", "عامري", "بن سالم", "الطرابلسي", "بوعزيزي",
        "الشابي", "المنصوري", "بن يوسف", "العيادي", "الهمامي",
    ],
    "first_name": [
        "سامر", "أنيس", "لمياء", "محمد", "فاطمة",
        "خالد", "سارة", "يوسف", "مريم", "علي",
    ],
    "father_lineage": [
        "بن رضا بن علي", "بن مصطفى بن عثمان",
        "بن محمد بن الحبيب", "بن علي بن يوسف",
        "بن صالح بن محمود", "بنت عمر بن سالم",
    ],
    "place_of_birth": [
        "المنستير", "الوسلاتية", "تونس", "صفاقس",
        "سوسة", "القيروان", "نابل", "بنزرت",
    ],
    "profession": [
        "طالب", "مهندس", "طبيب", "أستاذ", "موظف",
        "تاجر", "محامي", "ممرض", "عامل", "متقاعد",
    ],
}

FIELD_PROMPTS_TRAINING = {
    "last_name": "استخرج اسم العائلة فقط من النص التالي بدون أي رموز أو أرقام: {text}",
    "first_name": "استخرج الاسم الشخصي فقط من النص التالي بدون أي رموز أو أرقام: {text}",
    "father_lineage": "استخرج سلسلة النسب التي تبدأ بـ'بن' من النص التالي: {text}",
    "date_of_birth": "استخرج تاريخ الميلاد من النص التالي واكتبه بصيغة YYYY-MM-DD: {text}",
    "place_of_birth": "استخرج اسم المدينة أو المكان فقط من النص التالي: {text}",
    "profession": "استخرج المهنة فقط من النص التالي: {text}",
}


def generate_dataset(n_samples: int = 2000) -> list[dict]:
    dataset = []
    for field, clean_list in CLEAN_SAMPLES.items():
        per_field = n_samples // len(CLEAN_SAMPLES)
        for _ in range(per_field):
            clean = random.choice(clean_list)
            noisy = add_noise(clean, field)
            prompt_template = FIELD_PROMPTS_TRAINING[field]
            dataset.append({
                "input": prompt_template.format(text=noisy),
                "output": clean,
                "field": field,
            })
    return dataset


if __name__ == "__main__":
    data = generate_dataset(n_samples=3000)
    with open("training_data/ocr_clean_pairs.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(data)} training pairs")
