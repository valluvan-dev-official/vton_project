import json
import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import String, DateTime, Float, Text, Enum, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class JobStatus(str, PyEnum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.pending)
    person_image_path: Mapped[str] = mapped_column(String(512))
    garment_image_path: Mapped[str] = mapped_column(String(512))            # primary (best-quality) garment image
    garment_image_paths: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list — all angles submitted
    garment_size: Mapped[str] = mapped_column(String(8), default="M")
    person_size_estimate: Mapped[str | None] = mapped_column(String(8), nullable=True)
    result_image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    saved_as_training: Mapped[bool] = mapped_column(Boolean, default=False)
    user_consent: Mapped[bool] = mapped_column(Boolean, default=True)
    # Phase 1 shadow-mode fit analysis (api/app/services/fit_analysis) —
    # JSON-serialized FitAnalysisOut, or NULL if analysis wasn't computed /
    # failed / this row predates the feature. Never read by inference; see
    # fit_analysis/__init__.py docstring. Nullable and additive so existing
    # rows and clients that don't know about it are unaffected.
    fit_analysis_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self, result_url: str | None = None) -> dict:
        fit_analysis = None
        if self.fit_analysis_json:
            try:
                fit_analysis = json.loads(self.fit_analysis_json)
            except (TypeError, ValueError):
                fit_analysis = None

        return {
            "id": self.id,
            "status": self.status.value,
            "result_url": result_url,
            "quality_score": self.quality_score,
            "garment_size": self.garment_size,
            "person_size_estimate": self.person_size_estimate,
            "saved_as_training": self.saved_as_training,
            "user_consent": self.user_consent,
            "error_message": self.error_message,
            "fit_analysis": fit_analysis,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
