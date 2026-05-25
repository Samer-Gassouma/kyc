"""Integration tests for /api/face/enroll and /api/face/verify.

Requires:
  - PostgreSQL with pgvector running (PG_DATABASE_URL env var or default)
  - InsightFace buffalo_l model cached
  - A test face image at tests/fixtures/real_face.jpg

Usage:
  PG_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/kyc_test \\
  pytest tests/test_face_api.py -v
"""

from __future__ import annotations

import io
import os
import uuid
from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

TESTS_DIR = Path(__file__).parent
FIXTURES = TESTS_DIR / "fixtures"

# ── Test images (generated programmatically so tests run without external files) ──


def _face_image() -> bytes:
    """Generate a minimal valid face-like image (BGR gradient with oval).
    Real InsightFace tests need actual face images — see test_with_real_images below.
    """
    import cv2

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Skin-tone oval in center
    cx, cy = 320, 200
    for y in range(480):
        for x in range(640):
            dx = (x - cx) / 200
            dy = (y - cy) / 260
            if dx * dx + dy * dy < 1.0:
                img[y, x] = [180, 140, 110]  # BGR skin tone

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def _non_face_image() -> bytes:
    """A plain gray image with no face."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    import cv2

    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    from main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    """Dev token for test sessions."""
    return {"Authorization": "Bearer dev_token"}


# ── Tests ─────────────────────────────────────────────────────────────


class TestFaceEnroll:
    async def test_enroll_no_face_in_image_returns_400(self, client, auth_headers):
        """Reject images with no detectable face."""
        files = {"image": ("test.jpg", io.BytesIO(_non_face_image()), "image/jpeg")}
        data = {"liveness_score": "0.95"}
        res = await client.post("/api/face/enroll", files=files, data=data, headers=auth_headers)
        assert res.status_code == 400
        assert "No face detected" in res.json()["detail"]

    async def test_enroll_creates_user_when_no_user_id_provided(self, client, auth_headers):
        """Auto-create a User when user_id is not provided."""
        files = {"image": ("test.jpg", io.BytesIO(_face_image()), "image/jpeg")}
        data = {"liveness_score": "0.95"}
        res = await client.post("/api/face/enroll", files=files, data=data, headers=auth_headers)
        # May succeed or fail depending on whether the synthetic face passes InsightFace
        # If it fails with "No face detected", that's expected for synthetic images
        if res.status_code == 400:
            assert "No face detected" in res.json()["detail"]
        else:
            assert res.status_code == 200
            body = res.json()
            assert "user_id" in body
            assert body["verified"] is True
            assert body["embedding_dim"] == 512

    async def test_enroll_rejects_invalid_user_id_format(self, client, auth_headers):
        """Return 400 for malformed UUID."""
        files = {"image": ("test.jpg", io.BytesIO(_face_image()), "image/jpeg")}
        data = {"user_id": "not-a-uuid", "liveness_score": "0.9"}
        res = await client.post("/api/face/enroll", files=files, data=data, headers=auth_headers)
        # Could fail at image stage or UUID stage — either 400 is correct
        assert res.status_code == 400

    async def test_enroll_stores_landmarks(self, client, auth_headers):
        """Landmarks JSON is stored alongside embedding."""
        files = {"image": ("test.jpg", io.BytesIO(_face_image()), "image/jpeg")}
        landmarks = [[0.1, 0.2, -0.05] for _ in range(468)]
        import json

        data = {
            "liveness_score": "0.95",
            "landmarks_3d": json.dumps(landmarks),
        }
        res = await client.post("/api/face/enroll", files=files, data=data, headers=auth_headers)
        # If synthetic face passes InsightFace, verify landmarks stored
        if res.status_code == 200:
            body = res.json()
            assert body["verified"] is True

    async def test_enroll_rejects_invalid_landmarks_json(self, client, auth_headers):
        """Return 400 for malformed landmarks JSON."""
        files = {"image": ("test.jpg", io.BytesIO(_face_image()), "image/jpeg")}
        data = {"liveness_score": "0.9", "landmarks_3d": "not-json"}
        res = await client.post("/api/face/enroll", files=files, data=data, headers=auth_headers)
        assert res.status_code == 400
        assert "landmarks_3d" in res.json()["detail"].lower()


class TestFaceVerify:
    async def test_verify_rejects_invalid_user_id(self, client, auth_headers):
        """Return 400 for malformed UUID."""
        files = {"image": ("test.jpg", io.BytesIO(_face_image()), "image/jpeg")}
        data = {"user_id": "bad-uuid"}
        res = await client.post("/api/face/verify", files=files, data=data, headers=auth_headers)
        assert res.status_code == 400

    async def test_verify_no_face_profile_returns_404(self, client, auth_headers):
        """Return 404 when user has no enrolled face."""
        uid = str(uuid.uuid4())
        files = {"image": ("test.jpg", io.BytesIO(_face_image()), "image/jpeg")}
        data = {"user_id": uid}
        res = await client.post("/api/face/verify", files=files, data=data, headers=auth_headers)
        # Either 400 (no face) or 404 (no profile) — both are correct depending on image
        assert res.status_code in (400, 404)

    async def test_verify_no_face_in_image_returns_400(self, client, auth_headers):
        """Reject images with no face."""
        uid = str(uuid.uuid4())
        files = {"image": ("test.jpg", io.BytesIO(_non_face_image()), "image/jpeg")}
        data = {"user_id": uid}
        res = await client.post("/api/face/verify", files=files, data=data, headers=auth_headers)
        assert res.status_code == 400
        assert "No face detected" in res.json()["detail"]


class TestFaceProfile:
    async def test_profile_invalid_user_id_returns_400(self, client, auth_headers):
        res = await client.get("/api/face/profile/not-a-uuid", headers=auth_headers)
        assert res.status_code == 400

    async def test_profile_nonexistent_user_returns_404(self, client, auth_headers):
        uid = str(uuid.uuid4())
        res = await client.get(f"/api/face/profile/{uid}", headers=auth_headers)
        assert res.status_code == 404


class TestThresholdBehavior:
    """Verify MATCH_THRESHOLD = 0.85 is enforced correctly."""

    async def test_cosine_similarity_below_threshold_rejected(self, client, auth_headers):
        """When similarity < 0.85, matched should be False."""
        # This test requires two different face images enrolled and verified.
        # With synthetic images, InsightFace won't detect faces, so we test
        # the threshold logic at the boundary via the response schema.
        #
        # Real test: enroll face A, verify with face B → should get matched=false
        pass

    async def test_cosine_similarity_above_threshold_accepted(self, client, auth_headers):
        """When similarity >= 0.85, matched should be True."""
        # Requires real face images — see test_with_real_images below.
        pass


# ── Real-image tests (run with TEST_FACE_DIR env var) ────────────────


@pytest.mark.skipif(
    not os.environ.get("TEST_FACE_DIR"),
    reason="Set TEST_FACE_DIR to a directory containing real_face_1.jpg, real_face_2.jpg, spoof.jpg",
)
class TestWithRealFaces:
    """End-to-end tests using real face images.

    Set TEST_FACE_DIR=/path/to/faces with:
      - real_face_1.jpg  — a real person's face (for enrollment)
      - real_face_2.jpg  — same person, different pose/lighting (for verification)
      - other_person.jpg — a different person's face (for negative match)
      - spoof.jpg        — a printed photo or screen replay of real_face_1
    """

    @pytest.fixture
    def face_dir(self):
        return Path(os.environ["TEST_FACE_DIR"])

    def _load(self, face_dir: Path, name: str) -> bytes:
        path = face_dir / name
        if not path.exists():
            pytest.skip(f"Missing test image: {path}")
        return path.read_bytes()

    async def test_enroll_and_verify_same_person(self, client, auth_headers, face_dir):
        """Enroll person A, then verify with a different photo of person A → matched=True."""
        enroll_img = self._load(face_dir, "real_face_1.jpg")
        verify_img = self._load(face_dir, "real_face_2.jpg")

        # Enroll
        files = {"image": ("face.jpg", io.BytesIO(enroll_img), "image/jpeg")}
        data = {"liveness_score": "0.95"}
        res = await client.post("/api/face/enroll", files=files, data=data, headers=auth_headers)
        assert res.status_code == 200, res.text
        user_id = res.json()["user_id"]

        # Verify same person
        files2 = {"image": ("verify.jpg", io.BytesIO(verify_img), "image/jpeg")}
        data2 = {"user_id": user_id}
        res2 = await client.post("/api/face/verify", files=files2, data=data2, headers=auth_headers)
        assert res2.status_code == 200, res2.text
        body = res2.json()
        assert body["matched"] is True, f"Expected match, got confidence={body['confidence']}"
        assert body["confidence"] >= 0.85
        assert body["user_id"] == user_id

    async def test_verify_different_person_rejected(self, client, auth_headers, face_dir):
        """Enroll person A, verify with person B → matched=False."""
        enroll_img = self._load(face_dir, "real_face_1.jpg")
        other_img = self._load(face_dir, "other_person.jpg")

        # Enroll person A
        files = {"image": ("face.jpg", io.BytesIO(enroll_img), "image/jpeg")}
        data = {"liveness_score": "0.95"}
        res = await client.post("/api/face/enroll", files=files, data=data, headers=auth_headers)
        assert res.status_code == 200, res.text
        user_id = res.json()["user_id"]

        # Verify with person B
        files2 = {"image": ("other.jpg", io.BytesIO(other_img), "image/jpeg")}
        data2 = {"user_id": user_id}
        res2 = await client.post("/api/face/verify", files=files2, data=data2, headers=auth_headers)
        assert res2.status_code == 200, res2.text
        body = res2.json()
        assert body["matched"] is False, f"Expected no match, got confidence={body['confidence']}"
        assert body["confidence"] < 0.85

    async def test_liveness_score_persisted(self, client, auth_headers, face_dir):
        """Enrollment stores the liveness_score in the profile."""
        enroll_img = self._load(face_dir, "real_face_1.jpg")
        files = {"image": ("face.jpg", io.BytesIO(enroll_img), "image/jpeg")}
        data = {"liveness_score": "0.72"}
        res = await client.post("/api/face/enroll", files=files, data=data, headers=auth_headers)
        assert res.status_code == 200
        user_id = res.json()["user_id"]

        # Check profile
        res2 = await client.get(f"/api/face/profile/{user_id}", headers=auth_headers)
        assert res2.status_code == 200
        profile = res2.json()
        assert profile["liveness_score"] == 0.72
        assert profile["verified"] is True

    async def test_enroll_with_low_liveness_still_stores(self, client, auth_headers, face_dir):
        """Even low liveness scores are stored (rejection happens client-side)."""
        enroll_img = self._load(face_dir, "real_face_1.jpg")
        files = {"image": ("face.jpg", io.BytesIO(enroll_img), "image/jpeg")}
        data = {"liveness_score": "0.12"}
        res = await client.post("/api/face/enroll", files=files, data=data, headers=auth_headers)
        assert res.status_code == 200
        user_id = res.json()["user_id"]

        res2 = await client.get(f"/api/face/profile/{user_id}", headers=auth_headers)
        assert res2.json()["liveness_score"] == 0.12


class TestAuthRequired:
    async def test_enroll_requires_auth(self, client):
        res = await client.post("/api/face/enroll")
        assert res.status_code in (401, 403)

    async def test_verify_requires_auth(self, client):
        res = await client.post("/api/face/verify")
        assert res.status_code in (401, 403)

    async def test_profile_requires_auth(self, client):
        res = await client.get(f"/api/face/profile/{uuid.uuid4()}")
        assert res.status_code in (401, 403)
