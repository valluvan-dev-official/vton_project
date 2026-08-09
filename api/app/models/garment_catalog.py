"""Production merchant garment size-chart storage (Phase 2).

Distinct from app/models/fit.py's GarmentSizeChart on purpose:
GarmentSizeChart is keyed by (garment_category, size_label) only — one row
per size *label*, shared across every merchant. That's unusable here: two
merchants selling an "XL" t-shirt legitimately measure differently, and
this table exists specifically to keep them apart. Keyed by
(merchant, sku, size_label) instead — one row per merchant's specific
product's specific size.

Merchant-provided measurements are the only authoritative source consumed
by FitEngine in production — see api/app/services/garment_catalog.py and
api/app/services/fit_analysis/garment_measurements.py's DEFAULT_TSHIRT_SIZE_CHART
docstring (dev/test fallback only, gated off by default).

Normalization policy (Phase 2 review)
--------------------------------------
merchant/sku/size_label matching is CASE-INSENSITIVE and whitespace-
insensitive: "AcmeApparel", "acmeapparel", and "  Acme Apparel " must all
resolve to the same logical merchant, or an accidental case/whitespace
typo silently creates a duplicate "phantom" product with its own
(possibly stale or missing) size chart instead of updating the real one.

merchant_key/sku_key/size_label_key below are the actual matching/
uniqueness columns — lowercased, internal-whitespace-collapsed, derived
automatically from merchant/sku/size_label via the @validates hooks so
every write path (this app's service layer, a future admin tool, a raw
ORM session) gets consistent normalization, not just callers that
remember to call a helper. merchant/sku/size_label themselves keep their
original (trimmed) casing for display — the API never lowercases what a
merchant typed.

The UNIQUE constraint is on the *_key columns, not the display columns —
this is enforced by the database itself (not just application code), so
even a concurrent write or a bypass of the service layer can't create a
case/whitespace-variant duplicate of the same logical product+size.
"""
import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, validates

from app.database import Base
from app.models.fit import FIT_STYLES, STRETCH_CATEGORIES


def normalize_key(value: str | None) -> str:
    """Canonical comparison form: trim, collapse internal whitespace runs
    to a single space, lowercase. Used for merchant/sku/size_label
    matching everywhere (model validators here, and
    api/app/services/garment_catalog.py's query filters) — the single
    source of truth for what "the same merchant/SKU/size" means."""
    return " ".join((value or "").split()).lower()


class MerchantGarmentSizeChart(Base):
    __tablename__ = "merchant_garment_size_charts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Free-text on purpose — "arbitrary merchant size labels" (spec) means
    # size_label isn't restricted to XS-XXL (e.g. "EU 42", "Size 10", "One Size").
    # Display value: trimmed, original casing preserved.
    merchant: Mapped[str] = mapped_column(String(128))
    sku: Mapped[str] = mapped_column(String(128))
    size_label: Mapped[str] = mapped_column(String(32))

    # Matching/uniqueness columns — see module docstring. Always derived
    # from the display columns above via @validates; never set directly.
    merchant_key: Mapped[str] = mapped_column(String(128))
    sku_key: Mapped[str] = mapped_column(String(128))
    size_label_key: Mapped[str] = mapped_column(String(32))

    garment_category: Mapped[str | None] = mapped_column(String(64), nullable=True)

    chest_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    shoulder_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    waist_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    hip_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    length_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    sleeve_length_cm: Mapped[float | None] = mapped_column(Float, nullable=True)

    stretch_category: Mapped[str] = mapped_column(String(16), default="none")
    fit_style: Mapped[str] = mapped_column(String(16), default="regular")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("merchant_key", "sku_key", "size_label_key", name="uq_merchant_sku_size_key"),
        Index("ix_merchant_garment_size_charts_merchant_sku_key", "merchant_key", "sku_key"),
        CheckConstraint(
            f"stretch_category IN {STRETCH_CATEGORIES}", name="ck_merchant_garment_stretch_category"
        ),
        CheckConstraint(
            f"fit_style IN {FIT_STYLES}", name="ck_merchant_garment_fit_style"
        ),
    )

    @validates("merchant")
    def _validate_merchant(self, key, value):
        value = (value or "").strip()
        self.merchant_key = normalize_key(value)
        return value

    @validates("sku")
    def _validate_sku(self, key, value):
        value = (value or "").strip()
        self.sku_key = normalize_key(value)
        return value

    @validates("size_label")
    def _validate_size_label(self, key, value):
        value = (value or "").strip()
        self.size_label_key = normalize_key(value)
        return value

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "merchant": self.merchant,
            "sku": self.sku,
            "size_label": self.size_label,
            "garment_category": self.garment_category,
            "chest_cm": self.chest_cm,
            "shoulder_cm": self.shoulder_cm,
            "waist_cm": self.waist_cm,
            "hip_cm": self.hip_cm,
            "length_cm": self.length_cm,
            "sleeve_length_cm": self.sleeve_length_cm,
            "stretch_category": self.stretch_category,
            "fit_style": self.fit_style,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
