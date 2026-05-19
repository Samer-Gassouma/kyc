# KYC ID Capture & Verification System

Real-time Tunisian CIN (Carte d'Identité Nationale) capture and verification.

- **Frontend**: Next.js app for ID capture, liveness detection, and CIN extraction
- **Backend**: FastAPI with YOLO, SAM, EasyOCR, Celery for async processing
- **Auth**: API key for service-to-service; optional JWT for frontend testing

## Architecture

```
Browser (Next.js)          →  FastAPI Backend
├── ONNX Quality Check          ├── YOLO + SAM (card detection & flattening)
├── Camera Stream                ├── EasyOCR + MRZ (async via Celery)
└── Auto-capture Logic           ├── Face crop extraction
                                 └── S3-compatible encrypted storage
```

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Redis
- MinIO or S3-compatible storage

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy env and set your secrets
cp .env.example .env
# Edit .env — set API_KEYS, JWT_SECRET, AES_ENCRYPTION_KEY

# Download model weights (~600 MB)
python scripts/setup_models.py

# Start Redis and MinIO (or use docker-compose)
# Then start the server:
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Celery Worker

```bash
cd backend
source .venv/bin/activate
PYTHONPATH=/home/ivan/kyc/backend celery -A celery_worker worker --loglevel=info
```

### 3. Frontend

```bash
cd id-capture
npm install
npm run dev
```

### Docker (full stack)

```bash
docker-compose up --build
```

## Model Weights

Run `python scripts/setup_models.py` once after install. It downloads:

| Model | Size | Purpose |
|---|---|---|
| `yolov8n.pt` | 6 MB | Document detection |
| `rcnn_coco.pt` | 160 MB | Document validation |
| `sam_vit_b.pth` | 358 MB | Card segmentation / flattening |
| `facenet_vggface2.pt` | 107 MB | Face embedding |
| `quality_classifier.onnx` | 4 MB | In-browser quality check |

Weights are gitignored. Do **not** commit them.

## Environment Variables

See `backend/.env.example` and `id-capture/.env.local.example`.

Key backend vars:
- `API_KEYS` — comma-separated keys for service-to-service auth
- `JWT_SECRET` — for frontend test sessions
- `REDIS_URL` — Celery broker
- `S3_BUCKET`, `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`
- `AES_ENCRYPTION_KEY` — 32-byte hex for capture encryption
- `DATABASE_URL` — SQLite by default

## API Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /api/extract/start` | API Key or JWT | Start async front+back extraction |
| `GET /api/extract/status/{id}` | API Key or JWT | Poll extraction progress & results |
| `GET /api/capture/{id}/face-crop` | API Key or JWT | Download face crop JPEG |
| `POST /api/auth/token` | None | Get test JWT |
| `GET /health` | None | Health check |

## Extraction Flow

1. **Upload** front + back images → returns `session_id`
2. **Async processing** — Celery workers run OCR on both sides in parallel
3. **Poll status** — `GET /api/extract/status/{session_id}`
4. **Results** — merged front/back fields + face crop URL

## GitHub Cleanup Notes

- `backend/weights/` — model binaries, auto-downloaded via script
- `backend/.env` — secrets, never commit (`.env.example` provided)
- `*.db` — SQLite databases, generated at runtime
- `__pycache__/` — cleaned before push
