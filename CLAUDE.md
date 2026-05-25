# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time Tunisian CIN (national ID card) capture and verification system. Frontend is a Next.js 16 camera app with in-browser MediaPipe FaceMesh + Silent-Face liveness. Backend is FastAPI with YOLO, SAM, InsightFace ArcFace, EasyOCR, and Celery for async ML processing. Face embeddings stored in PostgreSQL + pgvector.

## Commands

### Backend (Python/FastAPI)

```bash
cd backend
source .venv/bin/activate

# Development server
uvicorn main:app --host 0.0.0.0 --port 8000

# Celery worker (OCR tasks run here)
PYTHONPATH=/home/ivan/kyc/backend celery -A celery_worker worker --loglevel=info

# Download model weights (~600 MB, one-time)
python scripts/setup_models.py

# Download face pipeline models
python scripts/setup_face_models.py
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
docker-compose up --build   # spins up PostgreSQL, Redis, MinIO, backend, Celery worker, frontend
```

## Architecture

```
Browser (Next.js)           FastAPI Backend            PostgreSQL + pgvector
├─ MediaPipe FaceMesh       ├─ InsightFace ArcFace     └─ face_profiles (vector(512))
├─ Silent-Face Liveness     ├─ YOLO + SAM              
└─ ONNX quality classifier  ├─ EasyOCR + ROI           
                             ├─ Redis ── Celery Worker  
                             └─ S3/MinIO (encrypted)    
```

### Backend (`backend/`)

- **`main.py`** — FastAPI app, mounts all routers, preloads ML models on startup
- **`core/config.py`** — Pydantic Settings from `.env` (all config via environment variables)
- **`core/db.py`** — SQLAlchemy models (SQLite): `Capture`, `KYCResult`, `ExtractionSession`
- **`core/pg_db.py`** — Async SQLAlchemy + pgvector models (PostgreSQL): `User`, `FaceProfile` with ivfflat cosine index
- **`core/auth.py`** — Dual auth: API key (`X-API-Key` header) with JWT fallback (`Authorization: Bearer` or `?token=` query param)
- **`core/storage.py`** — S3-compatible storage with AES-256-GCM encryption
- **`celery_worker.py`** — Celery app + task definitions (`tasks.ocr`)
- **`routers/stream.py`** — WebSocket `/ws/stream`: receives JPEG frames, returns YOLO + SAM geometry (SAM runs every 5th frame for perf)
- **`routers/capture.py`** — REST: submit/validate captures, poll OCR status, serve face crops
- **`routers/extract.py`** — REST: start async extraction (front+back), poll status with Redis Pub/Sub or DB polling fallback
- **`routers/face.py`** — REST: POST `/api/face/enroll` (InsightFace embedding + pgvector store), POST `/api/face/verify` (cosine similarity ≥ 0.35), GET `/api/face/profile/{user_id}`
- **`routers/gallery.py`** — Upload endpoint with auto-detection, crop, rectify, validate
- **`routers/auth.py`** — JWT token issuance for frontend test sessions
- **`models/yolo_detector.py`** — YOLOv8 + contour-based document detection (hybrid: YOLO → contour refinement → pure contour fallback)
- **`models/geometry.py`** — SAM ViT-B segmentation → mask-to-quad → geometric measurements (angle, skew, coverage) → perspective correction. Fallback chain: SAM → classical contours → bbox corners
- **`models/face.py`** — InsightFace ArcFace (`buffalo_l`) encoder: face detection, alignment, 512-d embedding generation
- **`models/roi_extractor.py`** — Region-of-interest field extraction for Tunisian CIN: predefined ROI boxes (relative coordinates) → adaptive preprocessing per field → parallel EasyOCR via ThreadPoolExecutor → Arabic text parsing (dates, names, CIN validation). Includes barcode decoding (pyzbar)
- **`models/quality_checker.py`** — Laplacian blur check, glare detection, brightness analysis
- **`models/rcnn_validator.py`** — Faster R-CNN (COCO pretrained) for document presence validation with quality checks
- **`models/card_rectifier.py`** — Multi-method rotation estimation (FFT + projection + Hough, consensus voting) with before/after validation
- **`models/ocr_cleaner.py`** — mT5/AraT5-based OCR post-processing for Arabic fields with regex fallback
- **`tasks/ocr_task.py`** — Celery task handler: runs ROI extraction on corrected card, persists results, publishes Redis Pub/Sub notification
- **`scripts/setup_models.py`** — Downloads model weights (YOLO, R-CNN, SAM, ONNX quality classifier)
- **`scripts/setup_face_models.py`** — Downloads Silent-Face ONNX model for client-side liveness; InsightFace auto-downloads on first use

### Frontend (`id-capture/`)

- **Next.js 16** with App Router, TypeScript, Tailwind CSS 4
- Pages: `/` (home), `/kyc` (full KYC flow), `/extract` (CIN data extraction)
- **`components/kyc/`** — Camera overlay, capture review, ID capture step, face scan step (MediaPipe + Silent-Face), quality indicator, step progress
- **`hooks/`** — `useAutoCapture`, `useCamera`, `useCaptureStatus`, `useIDWebSocket`, `useONNXQuality`, `useMediaPipeFace`
- **`lib/`** — API base URL, frame encoder, ONNX model loader, Silent-Face liveness wrapper
- **Key dependency**: Next.js 16 has breaking changes — when writing frontend code, consult `id-capture/node_modules/next/dist/docs/` for the current API
- API calls proxy to backend via Next.js rewrites (`/api/*` → `http://localhost:8000/api/*`)

### Key ML Pipeline

1. **Document detection**: YOLOv8 finds card → SAM segments pixel mask → contour extracts 4 corners → perspective warp to 856×540 flat card
2. **OCR**: Predefined ROI boxes cropped → adaptive preprocessing (CLAHE, upscaling, Arabic-aware dilation) → parallel EasyOCR (Arabic + English) → field-specific parsing (Arabic name extraction, date parsing with fuzzy month matching)
3. **Face enrollment**: CIN face photo extracted from front card → auto-enrolled via InsightFace ArcFace → 512-d embedding stored in PostgreSQL pgvector
4. **Liveness + Verification**: MediaPipe FaceMesh extracts 468 3D landmarks client-side → Silent-Face Anti-Spoofing checks liveness in-browser → verified frame sent to server → InsightFace ArcFace generates 512-d embedding → pgvector cosine similarity query (threshold ≥ 0.35) → match result returned

### Data Flow

Images are encrypted (AES-256-GCM) before S3 upload. S3 keys are stored in the `captures` table. OCR results stored as JSON in `kyc_results`. Celery workers notify completion via Redis Pub/Sub on `kyc:capture:{id}:done` channel. The `/api/extract/start` endpoint fires both sides in parallel, waits for both workers (Redis Pub/Sub with DB polling fallback), then merges front+back fields.

### Notes

- Model weights in `backend/weights/` are gitignored — run `scripts/setup_models.py` to download
- Face models: run `scripts/setup_face_models.py` for Silent-Face ONNX; InsightFace auto-downloads buffalo_l on first use
- `backend/.env` is gitignored — copy from `.env.example`
- PostgreSQL with pgvector is required for face identity storage; SQLite remains for KYC operational data
- Redis is used as both Celery broker and Pub/Sub channel for cross-process communication
- SAM is expensive — in the WebSocket stream it only runs every 5th frame; the gallery/capture routers run it every time
