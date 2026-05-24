# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time Tunisian CIN (national ID card) capture and verification system. Frontend is a Next.js 16 camera app with in-browser ONNX quality checks. Backend is FastAPI with YOLO, SAM, EasyOCR, and Celery for async ML processing.

## Commands

### Backend (Python/FastAPI)

```bash
cd backend
source .venv/bin/activate

# Development server
uvicorn main:app --host 0.0.0.0 --port 8000

# Celery worker (OCR + face match run here)
PYTHONPATH=/home/ivan/kyc/backend celery -A celery_worker worker --loglevel=info

# Download model weights (~600 MB, one-time)
python scripts/setup_models.py
```

### Frontend (Next.js 16)

```bash
cd id-capture
npm run dev       # development server on port 3000
npm run build     # production build
npm run lint      # ESLint
```

### Full stack (Docker)

```bash
docker-compose up --build   # spins up Redis, MinIO, backend, Celery worker, frontend
```

## Architecture

```
Browser (Next.js + ONNX) ──→ FastAPI Backend ──→ Redis ──→ Celery Worker
                                   │                     │
                                   ▼                     ▼
                             S3/MinIO (encrypted)    EasyOCR + ROI extraction
```

### Backend (`backend/`)

- **`main.py`** — FastAPI app, mounts all routers, preloads ML models on startup
- **`core/config.py`** — Pydantic Settings from `.env` (all config via environment variables)
- **`core/db.py`** — SQLAlchemy models: `Capture`, `KYCResult`, `LivenessSession`, `ExtractionSession`
- **`core/auth.py`** — Dual auth: API key (`X-API-Key` header) with JWT fallback (`Authorization: Bearer` or `?token=` query param)
- **`core/storage.py`** — S3-compatible storage with AES-256-GCM encryption
- **`celery_worker.py`** — Celery app + task definitions (`tasks.ocr`, `tasks.face_match`)
- **`routers/stream.py`** — WebSocket `/ws/stream`: receives JPEG frames, returns YOLO + SAM geometry (SAM runs every 5th frame for perf)
- **`routers/capture.py`** — REST: submit/validate captures, poll OCR status, serve face crops
- **`routers/extract.py`** — REST: start async extraction (front+back), poll status with Redis Pub/Sub or DB polling fallback
- **`routers/liveness.py`** — `/api/kyc/start`, `/ws/liveness/{id}`, `/api/kyc/finalize/{id}`, `/api/kyc/status/{id}`
- **`routers/gallery.py`** — Upload endpoint with auto-detection, crop, rectify, validate
- **`routers/auth.py`** — JWT token issuance for frontend test sessions
- **`models/yolo_detector.py`** — YOLOv8 + contour-based document detection (hybrid: YOLO → contour refinement → pure contour fallback)
- **`models/geometry.py`** — SAM ViT-B segmentation → mask-to-quad → geometric measurements (angle, skew, coverage) → perspective correction. Fallback chain: SAM → classical contours → bbox corners
- **`models/liveness.py`** — Passive ONNX liveness: RetinaFace DNN detection → fr_liveness.onnx real/fake check. Auto-brightness + CLAHE preprocessing
- **`models/roi_extractor.py`** — Region-of-interest field extraction for Tunisian CIN: predefined ROI boxes (relative coordinates) → adaptive preprocessing per field → parallel EasyOCR via ThreadPoolExecutor → Arabic text parsing (dates, names, CIN validation). Includes barcode decoding (pyzbar)
- **`models/quality_checker.py`** — Laplacian blur check, glare detection, brightness analysis
- **`models/rcnn_validator.py`** — Faster R-CNN (COCO pretrained) for document presence validation with quality checks
- **`models/card_rectifier.py`** — Multi-method rotation estimation (FFT + projection + Hough, consensus voting) with before/after validation
- **`models/ocr_cleaner.py`** — mT5/AraT5-based OCR post-processing for Arabic fields with regex fallback
- **`tasks/ocr_task.py`** — Celery task handler: runs ROI extraction on corrected card, persists results, publishes Redis Pub/Sub notification
- **`tasks/face_match_task.py`** — Celery task: DeepFace face matching between ID photo and liveness selfie
- **`scripts/setup_models.py`** — Downloads all model weights (YOLO, R-CNN, SAM, FaceNet, ONNX quality classifier)

### Frontend (`id-capture/`)

- **Next.js 16** with App Router, TypeScript, Tailwind CSS 4
- Pages: `/` (home), `/kyc` (full KYC flow), `/extract` (CIN data extraction), `/liveness` (standalone liveness)
- **`components/kyc/`** — Camera overlay, capture review, ID capture step, liveness step, quality indicator, step progress
- **`hooks/`** — `useAutoCapture`, `useCamera`, `useCaptureStatus`, `useIDWebSocket`, `useONNXQuality`
- **`lib/`** — API base URL, frame encoder, ONNX model loader
- **Key dependency**: Next.js 16 has breaking changes — when writing frontend code, consult `id-capture/node_modules/next/dist/docs/` for the current API
- API calls proxy to backend via Next.js rewrites (`/api/*` → `http://localhost:8000/api/*`)

### Key ML Pipeline

1. **Document detection**: YOLOv8 finds card → SAM segments pixel mask → contour extracts 4 corners → perspective warp to 856×540 flat card
2. **OCR**: Predefined ROI boxes cropped → adaptive preprocessing (CLAHE, upscaling, Arabic-aware dilation) → parallel EasyOCR (Arabic + English) → field-specific parsing (Arabic name extraction, date parsing with fuzzy month matching)
3. **Liveness**: RetinaFace DNN detects face → ONNX fr_liveness classifies real/spoof → 12 consecutive "real" frames needed to pass → selfie saved for face match
4. **Face match**: CIN photo (extracted from front card) vs liveness selfie via DeepFace Facenet

### Data Flow

Images are encrypted (AES-256-GCM) before S3 upload. S3 keys are stored in the `captures` table. OCR results stored as JSON in `kyc_results`. Celery workers notify completion via Redis Pub/Sub on `kyc:capture:{id}:done` channel. The `/api/extract/start` endpoint fires both sides in parallel, waits for both workers (Redis Pub/Sub with DB polling fallback), then merges front+back fields.

### Notes

- Model weights in `backend/weights/` are gitignored — run `scripts/setup_models.py` to download
- `backend/.env` is gitignored — copy from `.env.example`
- The `id-capture` subproject uses `faceplugin-face-recognition-js` for browser-side ONNX inference
- Redis is used as both Celery broker and Pub/Sub channel for cross-process communication
- SAM is expensive — in the WebSocket stream it only runs every 5th frame; the gallery/capture routers run it every time
