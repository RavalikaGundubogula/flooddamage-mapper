from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    """Granular stages of the end-to-end disaster analysis pipeline."""
    CREATED = "created"
    VALIDATING = "validating"
    DISCOVERING_SATELLITES = "discovering_satellites"
    SELECTING_SCENES = "selecting_scenes"
    PREPARING_ARD = "preparing_ard"
    RUNNING_SEGMENTATION = "running_segmentation"
    RUNNING_GEOSPATIAL = "running_geospatial"
    RUNNING_CONNECTIVITY = "running_connectivity"
    BUILDING_RESULTS = "building_results"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    COMPLETED_WITH_LIMITATIONS = "completed_with_limitations"
    BLOCKED = "blocked"
    FAILED = "failed"


class PipelineRunRequest(BaseModel):
    """User request payload to initiate an end-to-end flood damage analysis pipeline."""
    aoi: Optional[Union[Dict[str, Any], List[float]]] = Field(
        None,
        description="GeoJSON Polygon, Feature, or Bounding Box [min_lon, min_lat, max_lon, max_lat]. If omitted, defaults to Trishuli Nepal case study.",
    )
    flood_date: str = Field(
        "2026-08-26",
        description="Date of the flood event (YYYY-MM-DD or DD-MM-YYYY).",
        examples=["2026-08-26"],
    )
    report_language: str = Field(
        "en",
        description="Language for the situation report ('en' for English, 'ne' for Nepali).",
        examples=["en", "ne"],
    )
    demo_mode: bool = Field(
        False,
        description="If True, runs pipeline with pre-computed/synthetic demonstration layers for full UI visualization without CDSE credentials.",
    )
    comparison_date: Optional[str] = Field(
        None,
        description="Optional explicit post-event comparison satellite date (YYYY-MM-DD). If omitted, search window is used.",
    )
    reference_date: Optional[str] = Field(
        None,
        description="Optional explicit pre-event reference baseline satellite date (YYYY-MM-DD).",
    )
    pre_event_window_days: int = Field(
        15,
        description="Number of days prior to the flood event to search for baseline satellite imagery.",
    )
    post_event_window_days: int = Field(
        7,
        description="Number of days following the flood event to search for disaster satellite imagery.",
    )


class StageLogEntry(BaseModel):
    """Historical stage log entry during pipeline execution."""
    stage: PipelineStage
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: str
    progress_percent: int


class PipelineStatusResponse(BaseModel):
    """Comprehensive pipeline execution status for polling and dashboard display."""
    job_id: str
    status: str
    current_stage: PipelineStage
    progress_percent: int
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_blocked: bool = False
    blocker: Optional[str] = None
    error_message: Optional[str] = None
    demo_mode: bool = False
    flood_date: str = "2026-08-26"
    report_language: str = "en"
    stages_history: List[StageLogEntry] = Field(default_factory=list)
    available_layers: List[str] = Field(default_factory=list)
    satellite_status: Optional[str] = None
    segmentation_status: Optional[str] = None
    geospatial_status: Optional[str] = None
    connectivity_status: Optional[str] = None
    report_status: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
