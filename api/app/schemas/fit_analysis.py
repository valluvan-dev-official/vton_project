"""Pydantic response shape for the Phase-1 shadow-mode fit_analysis block
embedded (optionally) in the job status/result response. Purely additive —
existing clients that don't know this key exists are unaffected; see
app/models/job.py Job.to_dict().
"""
from typing import Optional

from pydantic import BaseModel


class FitAnalysisOut(BaseModel):
    selected_size: str
    estimated_person_size: Optional[str] = None
    overall_fit: str
    chest_fit: str
    shoulder_fit: str
    length_fit: str
    confidence: float
    explanation: str
    measurement_source: str
    is_shadow_mode: bool = True
