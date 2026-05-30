"""PostgreSQL + pgvector async engine and models for face identity storage."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from core.config import settings
from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

engine = create_async_engine(settings.pg_database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# ── Raw asyncpg pool for kyc_sessions (raw SQL, not ORM) ─────────────
import asyncpg

_raw_pool: asyncpg.Pool | None = None


async def _get_raw_pool() -> asyncpg.Pool:
    global _raw_pool
    if _raw_pool is None:
        # Derive asyncpg DSN from the SQLAlchemy URL
        # pg_database_url looks like: postgresql+asyncpg://user:pass@host:port/db
        dsn = settings.pg_database_url
        if dsn.startswith("postgresql+asyncpg://"):
            dsn = "postgresql://" + dsn.split("://", 1)[1]
        _raw_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10)
    return _raw_pool


async def get_raw_db():
    """Yield a raw asyncpg connection for kyc_sessions operations."""
    pool = await _get_raw_pool()
    async with pool.acquire() as conn:
        yield conn


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class FaceProfile(Base):
    __tablename__ = "face_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    embedding = mapped_column(Vector(512), nullable=False)
    landmarks_3d = mapped_column(JSONB, nullable=True)
    liveness_score: Mapped[float] = mapped_column(Float, nullable=False)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


async def init_pg_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        await conn.run_sync(Base.metadata.create_all)

        # ── kyc_sessions table (raw DDL, not managed by ORM) ──────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS kyc_sessions (
                id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id                 UUID,
                status                    TEXT DEFAULT 'in_progress',

                -- liveness
                liveness_passed           BOOLEAN DEFAULT FALSE,
                liveness_score            FLOAT,

                -- phone
                phone_number              TEXT,
                phone_verified            BOOLEAN DEFAULT FALSE,
                phone_otp_hash            TEXT,
                phone_otp_expires_at      TIMESTAMPTZ,
                phone_otp_attempts        INT DEFAULT 0,

                -- document
                document_country          TEXT DEFAULT 'TN',
                document_type             TEXT DEFAULT 'id_card',
                document_front_s3_key     TEXT,
                document_back_s3_key      TEXT,
                document_data             JSONB,

                -- face
                face_user_id              UUID,
                face_document_match       BOOLEAN DEFAULT FALSE,
                face_document_similarity  FLOAT,

                -- email
                email                     TEXT,
                email_verified            BOOLEAN DEFAULT FALSE,
                email_otp_hash            TEXT,
                email_otp_expires_at      TIMESTAMPTZ,
                email_otp_attempts        INT DEFAULT 0,

                -- result
                rejection_reasons         JSONB,
                completed_at              TIMESTAMPTZ,
                created_at                TIMESTAMPTZ DEFAULT NOW(),
                updated_at                TIMESTAMPTZ DEFAULT NOW()
            )
        """))

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_face_profiles_embedding "
                "ON face_profiles USING ivfflat (embedding vector_cosine_ops) "
                "WITH (lists = 100)"
            )
        )
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_kyc_sessions_status
            ON kyc_sessions (status)
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_kyc_sessions_cin
            ON kyc_sessions ((document_data->>'cin'))
        """))


async def get_pg_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
