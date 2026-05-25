#!/usr/bin/env python3
"""Download / prepare face pipeline model weights.

Run once after install:
    python scripts/setup_face_models.py

Models:
  1. InsightFace buffalo_l — auto-downloaded by insightface on first use
  2. Silent-Face Anti-Spoofing ONNX — download .pth, import real arch, export ONNX
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import urllib.request

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
FRONTEND_MODELS = os.path.join(
    os.path.dirname(__file__), "..", "..", "id-capture", "public", "models"
)

SILENT_FACE_PTH_URL = (
    "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/"
    "master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
)
SILENT_FACE_REPO = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git"
ONNX_OUTPUT_NAME = "silent_face.onnx"
MIN_FILE_SIZE = 500_000  # 500 KB


def setup_silent_face():
    """Download MiniFASNetV2 .pth and convert to ONNX via legacy exporter."""
    dst = os.path.join(FRONTEND_MODELS, ONNX_OUTPUT_NAME)
    if os.path.exists(dst) and os.path.getsize(dst) > MIN_FILE_SIZE:
        print(f"  [skip] Silent-Face ONNX already at {dst} "
              f"({os.path.getsize(dst) / 1e6:.1f} MB)")
        _verify_onnx(dst)
        return

    os.makedirs(FRONTEND_MODELS, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # ── Download .pth ──────────────────────────────────────────────
    pth_path = os.path.join(WEIGHTS_DIR, "2.7_80x80_MiniFASNetV2.pth")
    if not os.path.exists(pth_path):
        print("  Downloading Silent-Face .pth weights...")
        try:
            urllib.request.urlretrieve(SILENT_FACE_PTH_URL, pth_path)
            sz = os.path.getsize(pth_path) / 1e6
            print(f"  [done] Downloaded ({sz:.1f} MB) → {pth_path}")
        except Exception as e:
            print(f"  [error] Download failed: {e}")
            print(f"  [info] Manual DL: {SILENT_FACE_PTH_URL}")
            print(f"  [info] Place at: {pth_path}")
            sys.exit(1)
    else:
        print(f"  [skip] .pth already cached ({os.path.getsize(pth_path) / 1e6:.1f} MB)")

    # ── Clone repo for model source ─────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = os.path.join(tmp, "repo")
        print("  Cloning Silent-Face repo for model architecture...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none",
                 SILENT_FACE_REPO, repo_path],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  [error] git clone failed: {e.stderr.decode()}")
            sys.exit(1)

        model_lib_dir = os.path.join(repo_path, "src", "model_lib")
        if not os.path.isdir(model_lib_dir):
            print("  [error] model_lib directory not found in repo")
            sys.exit(1)

        sys.path.insert(0, model_lib_dir)

        try:
            print("  Converting MiniFASNetV2 → ONNX (legacy exporter, opset 11)...")
            _convert_to_onnx(pth_path, dst)
        finally:
            sys.path.remove(model_lib_dir)

    # ── Verify ─────────────────────────────────────────────────────
    sz = os.path.getsize(dst)
    if sz < MIN_FILE_SIZE:
        print(f"  [error] ONNX file too small: {sz} bytes (expected > {MIN_FILE_SIZE})")
        sys.exit(1)

    print(f"  [done] Silent-Face ONNX exported ({sz / 1e6:.2f} MB) → {dst}")
    _verify_onnx(dst)


def _convert_to_onnx(pth_path: str, onnx_dst: str):
    """Load MiniFASNetV2 weights, export to ONNX with legacy exporter (opset 11)."""
    import torch
    from MiniFASNet import MiniFASNetV2

    # Load state dict, strip 'module.' prefix (from DataParallel training)
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    clean_sd = {}
    for k, v in state_dict.items():
        clean_sd[k.removeprefix("module.")] = v

    # conv6_kernel=(5,5) matches the 80x80 input variant
    model = MiniFASNetV2(
        embedding_size=128,
        conv6_kernel=(5, 5),
        drop_p=0.0,
        num_classes=3,       # live, print attack, replay attack
        img_channel=3,
    )
    model.load_state_dict(clean_sd, strict=True)
    model.eval()

    dummy = torch.randn(1, 3, 80, 80)

    torch.onnx.export(
        model,
        dummy,
        onnx_dst,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


def _verify_onnx(onnx_path: str):
    """Confirm the ONNX model loads and runs correctly."""
    try:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]
        print(f"  [verify] Input: {inp.name} {inp.shape}, Output: {out.name} {out.shape}")

        dummy = np.random.randn(1, 3, 80, 80).astype(np.float32)
        result = session.run(None, {"input": dummy})
        probs = np.exp(result[0] - result[0].max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        print(f"  [verify] Output classes (live/print/replay): {probs[0].tolist()}")
        print("  [verify] ONNX model loads and runs correctly")
    except ImportError:
        print("  [verify] onnxruntime not installed, skipping inference check")
    except Exception as e:
        print(f"  [verify] Warning: ONNX check failed: {e}")


def print_insightface_info():
    print("\n  InsightFace buffalo_l — auto-downloaded on first use.")
    print("  The model loads automatically when the backend starts.")
    print("  Weights are cached in ~/.insightface/models/buffalo_l/")
    print("  To preload: start the backend and hit any /api/face/* endpoint.")


def main():
    print("=" * 60)
    print("Face Pipeline Model Setup")
    print("=" * 60)

    print("\n[1/2] Silent-Face Anti-Spoofing ONNX")
    setup_silent_face()

    print("\n[2/2] InsightFace ArcFace (buffalo_l)")
    print_insightface_info()

    print("\n" + "=" * 60)
    print("Face models ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
