"""InsightFace ArcFace encoder for 512-d face embedding generation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_encoder: FaceEncoder | None = None


class FaceEncoder:
    """Lazy-loads InsightFace buffalo_l model for face detection + embedding."""

    def __init__(self):
        self._app: Any = None

    def _load(self):
        if self._app is not None:
            return
        import insightface

        self._app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1)  # -1 = CPU
        logger.info("InsightFace buffalo_l loaded")

    def encode(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Detect face, extract embedding and aligned crop.

        Returns (512-d normalized embedding, aligned 112x112 face) or None.
        """
        self._load()

        # InsightFace expects BGR
        faces = self._app.get(image)

        if not faces:
            return None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        embedding = face.normed_embedding  # already L2-normalized, shape (512,)
        if embedding is None:
            return None

        aligned = face.normed_face  # aligned 112x112 BGR crop, or None

        return (embedding.astype(np.float32), aligned)

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two normalized embeddings."""
        return float(np.dot(emb1, emb2))


def get_face_encoder() -> FaceEncoder:
    global _encoder
    if _encoder is None:
        _encoder = FaceEncoder()
    return _encoder
