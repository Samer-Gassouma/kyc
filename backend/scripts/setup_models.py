#!/usr/bin/env python3
"""Download / generate all model weights needed by the KYC pipeline.

Run once after install:
    python scripts/setup_models.py

Models set up:
  1. YOLOv8n — general object detection (ultralytics auto-download)
  2. Faster R-CNN ResNet-50 FPN — pretrained COCO (torchvision)
  3. MobileNetV3-Small — exported to ONNX for in-browser quality check
  4. SAM ViT-B — card segmentation
  5. EasyOCR language models — pre-downloaded
"""

from __future__ import annotations

import os
import shutil
import sys

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
FRONTEND_MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "id-capture", "public", "models"
)


def ensure_dirs():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(FRONTEND_MODELS_DIR, exist_ok=True)


# ── 1. YOLOv8n ────────────────────────────────────────────────────
def setup_yolo():
    dst = os.path.join(WEIGHTS_DIR, "yolov8n.pt")
    if os.path.exists(dst):
        print(f"  [skip] YOLO already at {dst}")
        return

    print("  Downloading YOLOv8n from ultralytics...")
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # auto-downloads to current dir
    # Move the downloaded file to weights/
    default_path = "yolov8n.pt"
    if os.path.exists(default_path):
        shutil.move(default_path, dst)
    elif hasattr(model, "ckpt_path") and os.path.exists(model.ckpt_path):
        shutil.copy2(model.ckpt_path, dst)
    print(f"  [done] YOLO saved to {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


# ── 2. Faster R-CNN ResNet-50 FPN (pretrained COCO) ───────────────
def setup_rcnn():
    dst = os.path.join(WEIGHTS_DIR, "rcnn_coco.pt")
    if os.path.exists(dst):
        print(f"  [skip] R-CNN already at {dst}")
        return

    print("  Downloading Faster R-CNN ResNet-50 FPN (COCO pretrained)...")
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    torch.save(model.state_dict(), dst)
    print(f"  [done] R-CNN saved to {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


# ── 3. MobileNetV3-Small → ONNX (quality classifier) ─────────────
def setup_quality_onnx():
    onnx_dst = os.path.join(WEIGHTS_DIR, "quality_classifier.onnx")
    frontend_dst = os.path.join(FRONTEND_MODELS_DIR, "quality_classifier.onnx")

    if os.path.exists(onnx_dst) and os.path.getsize(onnx_dst) > 5000:
        print(f"  [skip] Quality ONNX already at {onnx_dst}")
        if not os.path.exists(frontend_dst) or os.path.getsize(frontend_dst) < 5000:
            shutil.copy2(onnx_dst, frontend_dst)
        return

    print("  Exporting MobileNetV3-Small to ONNX (quality classifier)...")
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    # Load pretrained MobileNetV3-Small
    base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # Replace classifier head: 576 features → 4 outputs
    # [blur_score, glare_score, orientation_idx, brightness_idx]
    base.classifier = nn.Sequential(
        nn.Linear(576, 128),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(128, 4),
        nn.Sigmoid(),  # keep outputs in [0, 1]
    )

    base.eval()

    # Export to ONNX (single file, all weights embedded)
    dummy = torch.randn(1, 3, 224, 224)
    # Export to a temp path first
    tmp_path = onnx_dst + ".tmp"
    torch.onnx.export(
        base,
        dummy,
        tmp_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
    )

    # Convert external data to single-file ONNX
    import onnx
    model_proto = onnx.load(tmp_path, load_external_data=True)
    onnx.save_model(model_proto, onnx_dst, save_as_external_data=False)
    # Cleanup temp files
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    tmp_data = tmp_path + ".data"
    if os.path.exists(tmp_data):
        os.remove(tmp_data)
    # Also remove any stale .data file next to output
    stale_data = onnx_dst + ".data"
    if os.path.exists(stale_data):
        os.remove(stale_data)

    sz = os.path.getsize(onnx_dst) / 1e6
    print(f"  [done] ONNX exported ({sz:.2f} MB, single file)")

    # Attempt INT8 quantization for smaller browser payload
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantized_dst = onnx_dst.replace(".onnx", "_q.onnx")
        quantize_dynamic(onnx_dst, quantized_dst, weight_type=QuantType.QUInt8)
        os.replace(quantized_dst, onnx_dst)
        print(f"  [done] Quantized to {os.path.getsize(onnx_dst) / 1e6:.2f} MB")
    except Exception as e:
        print(f"  [info] Quantization skipped ({e.__class__.__name__}), using fp32 model")

    # Copy to frontend
    shutil.copy2(onnx_dst, frontend_dst)
    print(f"  [done] Quality ONNX: {onnx_dst} → {frontend_dst}")


# ── 4. SAM ViT-B (card segmentation) ──────────────────────────────
def setup_sam():
    dst = os.path.join(WEIGHTS_DIR, "sam_vit_b.pth")
    if os.path.exists(dst):
        print(f"  [skip] SAM already at {dst}")
        return

    print("  Downloading SAM ViT-B (~375 MB)...")
    import urllib.request
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    urllib.request.urlretrieve(url, dst)
    print(f"  [done] SAM saved to {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


# ── 6. EasyOCR language models ────────────────────────────────────
def setup_easyocr():
    print("  Pre-downloading EasyOCR models (en, fr)...")
    try:
        import easyocr

        reader = easyocr.Reader(["en", "fr"], gpu=True, verbose=False)
        print("  [done] EasyOCR models cached")
    except Exception as e:
        print(f"  [warn] EasyOCR setup issue: {e}")


def main():
    print("=" * 60)
    print("KYC Model Setup")
    print("=" * 60)

    ensure_dirs()

    print("\n[1/5] YOLOv8n (document detection)")
    setup_yolo()

    print("\n[2/5] Faster R-CNN ResNet-50 FPN (COCO)")
    setup_rcnn()

    print("\n[3/5] MobileNetV3-Small → ONNX (quality classifier)")
    setup_quality_onnx()

    print("\n[4/5] SAM ViT-B (card segmentation)")
    setup_sam()

    print("\n[5/5] EasyOCR language models")
    setup_easyocr()

    print("\n" + "=" * 60)
    print("All models ready. Weights directory:")
    for f in sorted(os.listdir(WEIGHTS_DIR)):
        fp = os.path.join(WEIGHTS_DIR, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp) / 1e6
            print(f"  {f:40s} {sz:8.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
