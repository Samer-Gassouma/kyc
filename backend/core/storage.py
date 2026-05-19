"""S3-compatible storage wrapper with AES-256 encryption."""

from __future__ import annotations

import io
import os
import uuid

import boto3
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.config import settings


def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )


def _encrypt(data: bytes) -> tuple[bytes, bytes]:
    """AES-256-GCM encrypt. Returns (nonce + ciphertext, nonce)."""
    key = bytes.fromhex(settings.aes_encryption_key)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, data, None)
    return nonce + ct, nonce


def _decrypt(blob: bytes) -> bytes:
    """AES-256-GCM decrypt. Expects nonce (12 bytes) prepended to ciphertext."""
    key = bytes.fromhex(settings.aes_encryption_key)
    aesgcm = AESGCM(key)
    nonce = blob[:12]
    ct = blob[12:]
    return aesgcm.decrypt(nonce, ct, None)


def upload_encrypted(data: bytes, prefix: str = "captures") -> str:
    """Encrypt data and upload to S3. Returns the object key."""
    encrypted, _ = _encrypt(data)
    object_key = f"{prefix}/{uuid.uuid4().hex}.enc"
    client = _get_s3_client()
    client.put_object(
        Bucket=settings.s3_bucket,
        Key=object_key,
        Body=io.BytesIO(encrypted),
        ContentLength=len(encrypted),
    )
    return object_key


def download_decrypted(object_key: str) -> bytes:
    """Download from S3 and decrypt."""
    client = _get_s3_client()
    resp = client.get_object(Bucket=settings.s3_bucket, Key=object_key)
    blob = resp["Body"].read()
    return _decrypt(blob)
