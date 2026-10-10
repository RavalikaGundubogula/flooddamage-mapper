from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class AnalysisStatus(str, Enum):
    """Lifecycle statuses for an analysis job."""
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisRecord(BaseModel):
    """In-memory domain model representing an analysis job."""
    job_id: str
    status: AnalysisStatus = AnalysisStatus.CREATED
    aoi: Dict[str, Any]
    flood_date: str
    report_language: str = "en"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
