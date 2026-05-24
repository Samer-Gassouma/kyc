#!/usr/bin/env python3
"""End-to-end pipeline test with real models and test images."""

from __future__ import annotations

import os
import sys
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test_data")


def find_test_images():
    imgs = {}
    if not os.path.isdir(TEST_IMAGES_DIR):
        print(f"  [warn] test_data dir not found at {TEST_IMAGES_DIR}")
        return imgs
    for f in sorted(os.listdir(TEST_IMAGES_DIR)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(TEST_IMAGES_DIR, f)
            imgs[f] = cv2.imread(path)
    return imgs


def test_yolo_detection(images: dict):
    print("\n── YOLO Detection ──")
    from models.yolo_detector import detect_frame

    for name, img in images.items():
        t0 = time.time()
        result = detect_frame(img)
        dt = (time.time() - t0) * 1000
        det = result["detected"]
        conf = result["confidence"]
        issues = result.get("issues", [])
        ready = result.get("ready_to_capture", False)
        print(f"  {name:30s} detected={det}  conf={conf:.2f}  ready={ready}  "
              f"issues={len(issues)}  {dt:.0f}ms")
        if issues:
            for iss in issues[:3]:
                print(f"    - {iss}")


def test_rcnn_validation(images: dict):
    print("\n── R-CNN Validation ──")
    from models.rcnn_validator import validate_capture

    for name, img in images.items():
        t0 = time.time()
        result = validate_capture(img, side="front")
        dt = (time.time() - t0) * 1000
        passed = result["validation_passed"]
        conf = result["confidence"]
        reason = result.get("rejection_reason")
        qd = result.get("quality_details", {})
        print(f"  {name:30s} passed={passed}  conf={conf:.2f}  "
              f"blur={qd.get('blur_score', 0):.0f}  {dt:.0f}ms")
        if reason:
            print(f"    reject: {reason}")


def test_ocr(images: dict):
    print("\n── OCR / MRZ ──")
    from tasks.ocr_task import _run_easyocr, _run_mrz_parse

    for name, img in images.items():
        t0 = time.time()
        ocr = _run_easyocr(img)
        dt_ocr = (time.time() - t0) * 1000

        t1 = time.time()
        mrz = _run_mrz_parse(img)
        dt_mrz = (time.time() - t1) * 1000

        texts = [r["text"] for r in ocr[:5]]
        print(f"  {name:30s}  ocr_items={len(ocr)}  {dt_ocr:.0f}ms  mrz={'YES' if mrz else 'NO'}  {dt_mrz:.0f}ms")
        if texts:
            print(f"    top text: {texts[:3]}")
        if mrz:
            print(f"    MRZ: {mrz}")


def test_quality_checker(images: dict):
    print("\n── Quality Checker ──")
    from models.quality_checker import check_quality

    for name, img in images.items():
        t0 = time.time()
        result = check_quality(img)
        dt = (time.time() - t0) * 1000
        print(f"  {name:30s} pass={result['quality_passed']}  "
              f"blur={result.get('blur_score', 0):.0f}  "
              f"issues={result.get('issues', [])}  {dt:.0f}ms")


def main():
    print("=" * 60)
    print("KYC Pipeline End-to-End Test")
    print("=" * 60)

    images = find_test_images()
    if not images:
        # Create a synthetic test image
        print("  No test images found, creating synthetic card image...")
        card = np.ones((400, 640, 3), dtype=np.uint8) * 220
        cv2.rectangle(card, (50, 50), (590, 350), (200, 200, 200), -1)
        cv2.rectangle(card, (50, 50), (590, 350), (0, 0, 0), 2)
        cv2.putText(card, "REPUBLIC OF TUNISIA", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(card, "National ID Card", (100, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
        cv2.putText(card, "12345678", (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(card, "DOB: 01/01/1990", (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        images["synthetic_card.jpg"] = card

    print(f"\n  Found {len(images)} test image(s): {list(images.keys())}")

    test_yolo_detection(images)
    test_rcnn_validation(images)
    test_quality_checker(images)
    test_ocr(images)

    print("\n" + "=" * 60)
    print("Pipeline test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
