import uuid
from datetime import datetime

from sqlalchemy import (
    String, DateTime, Float, Text, Boolean, ForeignKey,
    UniqueConstraint, Index, CheckConstraint, JSON,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

# Native Postgres ENUM types are avoided on purpose (per Phase 1 requirements)
# — they're painful to alter later. Allowed values are enforced with a
# CHECK constraint instead, so `ALTER TABLE ... ADD VALUE` is never needed.
STRETCH_CATEGORIES = ("none", "low", "medium", "high")
FIT_STYLES = ("slim", "regular", "relaxed", "oversized")

# Generic JSON, upgraded to native JSONB on Postgres — works against both the
# production Postgres DB and SQLite (used in tests) without a variant error.
_JSONType = JSON().with_variant(JSONB(), "postgresql")


class BodyProfile(Base):
    __tablename__ = "body_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # Optional link to a try-on job; nullable so a fit estimate can be
    # requested standalone. No FK-level cascade — jobs lifecycle is untouched.
    job_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("jobs.id"), nullable=True)

    height_cm: Mapped[float] = mapped_column(Float)
    bust_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    waist_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    hips_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    shoulder_cm: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_body_profiles_job_id", "job_id"),
    )


class GarmentSizeChart(Base):
    __tablename__ = "garment_size_charts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    garment_category: Mapped[str] = mapped_column(String(64))
    size_label: Mapped[str] = mapped_column(String(16))

    bust_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    waist_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    hips_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    shoulder_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    length_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    sleeve_cm: Mapped[float | None] = mapped_column(Float, nullable=True)

    stretch_category: Mapped[str] = mapped_column(String(16), default="none")
    fit_style: Mapped[str] = mapped_column(String(16), default="regular")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("garment_category", "size_label", name="uq_garment_category_size"),
        Index("ix_garment_size_charts_category", "garment_category"),
        CheckConstraint(
            f"stretch_category IN {STRETCH_CATEGORIES}", name="ck_garment_stretch_category"
        ),
        CheckConstraint(
            f"fit_style IN {FIT_STYLES}", name="ck_garment_fit_style"
        ),
    )


class FitEstimate(Base):
    __tablename__ = "fit_estimates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("jobs.id"), nullable=True)
    body_profile_id: Mapped[str] = mapped_column(String(36), ForeignKey("body_profiles.id"), nullable=False)

    selected_size: Mapped[str] = mapped_column(String(16))
    recommended_size: Mapped[str | None] = mapped_column(String(16), nullable=True)
    body_shape: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)

    regional_fit: Mapped[dict] = mapped_column(_JSONType)
    summary: Mapped[str] = mapped_column(Text)
    is_approximate: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_fit_estimates_job_id", "job_id"),
        Index("ix_fit_estimates_body_profile_id", "body_profile_id"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "selected_size": self.selected_size,
            "recommended_size": self.recommended_size,
            "body_shape": self.body_shape,
            "confidence": self.confidence,
            "regional_fit": self.regional_fit,
            "summary": self.summary,
            "is_approximate": self.is_approximate,
            "created_at": self.created_at.isoformat(),
        }
