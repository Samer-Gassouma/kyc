# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time Tunisian CIN (national ID card) capture and verification system. Frontend is a Next.js 16 camera app with in-browser MediaPipe FaceMesh + blink-based liveness. Backend is FastAPI with YOLO, SAM, InsightFace ArcFace, EasyOCR, and Celery for async ML processing. Face embeddings stored in PostgreSQL + pgvector.

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
Browser (Next.js)              FastAPI Backend            PostgreSQL + pgvector
├─ MediaPipe FaceLandmarker    ├─ InsightFace ArcFace     └─ face_profiles (vector(512))
├─ Blink liveness (blendshapes)├─ YOLO + SAM              
├─ 3D face mesh (Three.js)     ├─ EasyOCR + ROI           
└─ Static card guide            ├─ Redis ── Celery Worker  
                                └─ S3/MinIO (encrypted)    
```

## KYC Flow

```
front_id → back_id → face_scan → extraction (background) → completed
```

1. **Card capture**: Static guide frame overlay. User aligns card, taps Capture. Preview → Retake or Use this photo. No auto-detection. No quality rejection — all images accepted.
2. **Face scan**: MediaPipe FaceLandmarker detects face, draws colored mesh overlay (eyes=blue, nose=purple, lips=pink, brows=yellow, jaw=green). Auto-captures after 2s hold. Blink detection via `eyeBlinkLeft`/`eyeBlinkRight` blendshapes for liveness. 3D face mesh displayed after capture (Three.js, OrbitControls, auto-rotate).
3. **Extraction**: Started as fire-and-forget after face scan passes. Runs YOLO → perspective correction → EasyOCR → ROI extraction in Celery worker. User gets session ID immediately, can check results later on `/extract` page.

## Face Pipeline

| Layer | Technology | Role |
|---|---|---|
| Face detection | **MediaPipe FaceLandmarker** | 468 3D landmarks + blendshapes |
| Liveness | **Blink detection** via blendshapes | eyeBlinkLeft/Right > 0.35 → open→close→open cycle |
| Identity embedding | **InsightFace ArcFace** (buffalo_l) | 512-d face vector, server-side |
| Vector storage | **PostgreSQL + pgvector** | Cosine similarity, ivfflat index |
| Match threshold | **0.35** (configurable: `FACE_MATCH_THRESHOLD`) | Calibrated on CFP dataset |
| 3D viewer | **Three.js** with OrbitControls | Colored mesh, auto-rotate |

### Backend (`backend/`)

- **`main.py`** — FastAPI app, mounts all routers, preloads ML models on startup
- **`core/config.py`** — Pydantic Settings from `.env`. Key: `FACE_MATCH_THRESHOLD=0.35`, `pg_database_url`
- **`core/db.py`** — SQLAlchemy models (SQLite): `Capture`, `KYCResult`, `ExtractionSession`
- **`core/pg_db.py`** — Async SQLAlchemy + pgvector (PostgreSQL): `User`, `FaceProfile` with ivfflat cosine index
- **`core/auth.py`** — Dual auth: API key + JWT fallback
- **`core/storage.py`** — S3-compatible storage with AES-256-GCM encryption
- **`celery_worker.py`** — Celery app + task `tasks.ocr`
- **`routers/stream.py`** — WebSocket `/ws/stream`: YOLO + SAM geometry for real-time card detection
- **`routers/capture.py`** — POST `/api/capture/validate` — **all images accepted unconditionally** (validation bypassed)
- **`routers/extract.py`** — POST `/api/extract/start` — async front+back processing, **no quality checks**. Poll status at `/api/extract/status/{id}`
- **`routers/face.py`** — POST `/api/face/enroll`, POST `/api/face/verify`, POST `/api/face/verify-against-document`, GET `/api/face/profile/{user_id}`
- **`routers/gallery.py`** — Upload endpoint with auto-detection
- **`routers/auth.py`** — JWT token issuance
- **`models/yolo_detector.py`** — YOLOv8 document detection
- **`models/geometry.py`** — SAM ViT-B segmentation → perspective correction
- **`models/face.py`** — InsightFace ArcFace encoder (512-d embeddings)
- **`models/roi_extractor.py`** — ROI field extraction for Tunisian CIN (EasyOCR, Arabic parsing, barcode)
- **`models/quality_checker.py`** — Quality checks (bypassed in production — not called)
- **`models/rcnn_validator.py`** — **Passthrough** — always returns `validation_passed: True`
- **`models/card_rectifier.py`** — Rotation estimation
- **`models/ocr_cleaner.py`** — mT5/AraT5 OCR post-processing
- **`tasks/ocr_task.py`** — Celery task: ROI extraction, persist results, Redis Pub/Sub notification
- **`scripts/setup_models.py`** — Downloads YOLO, R-CNN, SAM, ONNX quality classifier weights
- **`scripts/setup_face_models.py`** — InsightFace auto-download; Silent-Face ONNX conversion (legacy)
- **`scripts/test_face_pipeline.py`** — CFP dataset integration tests for face pipeline
- **`tests/test_face_api.py`** — API integration tests for enroll/verify/profile endpoints

### Frontend (`id-capture/`)

- **Next.js 16** with App Router, TypeScript, Tailwind CSS 4
- Pages: `/` (home), `/kyc` (full KYC flow), `/face` (standalone face test), `/extract` (CIN extraction)
- **`components/kyc/`**
  - `IDCaptureStep.tsx` — Manual capture with static guide frame, preview/confirm, no quality rejection
  - `Face3DViewer.tsx` — Three.js 3D face mesh viewer (colored regions, OrbitControls, auto-rotate)
  - `StepProgress.tsx` — 3-step progress indicator (Front ID, Back ID, Face Scan)
  - `FaceScanStep.tsx` — Standalone face scan component (legacy, mostly replaced by inline logic in KYC page)
- **`hooks/`**
  - `useFaceDetection.ts` — MediaPipe FaceLandmarker, region edges, triangle indices, face crop utility
  - `useCardDetection.ts` — Brightness-based card detection (legacy, not used in current flow)
  - `useCamera.ts` — Generic camera hook
  - `useAutoCapture.ts`, `useCaptureStatus.ts`, `useIDWebSocket.ts`, `useONNXQuality.ts` — Legacy utilities
- **`lib/`** — API base URL, frame encoder
- API calls proxy to backend via Next.js rewrites (`/api/*` → `http://localhost:8000/api/*`)

### Key ML Pipeline

1. **Document detection**: YOLOv8 finds card → SAM segments → perspective warp to 856×540 flat card
2. **OCR**: ROI boxes cropped → EasyOCR (Arabic + English) → field-specific parsing
3. **Face detection (client)**: MediaPipe FaceLandmarker → 468 3D landmarks + blendshapes (blink detection)
4. **Face enrollment (server)**: InsightFace ArcFace → 512-d embedding → pgvector storage
5. **Face verification**: Cosine similarity query via pgvector (threshold 0.35)

### Data Flow

Images are encrypted (AES-256-GCM) before S3 upload. S3 keys stored in `captures` table. OCR results stored as JSON in `kyc_results`. Celery workers notify completion via Redis Pub/Sub on `kyc:capture:{id}:done` channel.

### Notes

- Model weights in `backend/weights/` are gitignored — run `scripts/setup_models.py`
- InsightFace auto-downloads `buffalo_l` on first use (~275 MB)
- PostgreSQL with pgvector required for face identity; SQLite for KYC operational data
- **All image quality checks are bypassed** — `rcnn_validator.py` returns passthrough, `extract.py` has no quality gates
- Redis is Celery broker + Pub/Sub channel
- SAM expensive — WebSocket stream runs every 5th frame; capture/extract routers run it every time
- Face match threshold calibrated to 0.35 based on CFP dataset (99% TPR, 0% FPR at 100 identities)
- `face_profiles.landmarks_3d` stores 468-point 3D landmark arrays as JSONB
- No Silent-Face ONNX required — liveness is blink-based via MediaPipe blendshapes
- `/face` is a standalone testing page; `/kyc` is the production KYC flow
