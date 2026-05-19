"""Application configuration — all values driven by environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Model weights ──────────────────────────────────────────────
    yolo_weights_path: str = "./weights/yolov9_id.pt"
    yolo_conf_threshold: float = 0.25
    rcnn_weights_path: str = "./weights/rcnn_id.pt"
    onnx_quality_model: str = "./weights/quality_classifier.onnx"

    # ── Redis / Celery ─────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── S3 / MinIO ─────────────────────────────────────────────────
    s3_bucket: str = "kyc-captures"
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"

    # ── Auth ───────────────────────────────────────────────────────
    jwt_secret: str = "change_me_in_production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_keys: str = ""  # comma-separated e.g. "sk_prod_abc,sk_staging_xyz"

    # ── CORS ───────────────────────────────────────────────────────
    allowed_origins: str = "*"

    # ── Database ───────────────────────────────────────────────────
    database_url: str = "sqlite:///./kyc.db"

    # ── Encryption ─────────────────────────────────────────────────
    aes_encryption_key: str = "0123456789abcdef0123456789abcdef"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
