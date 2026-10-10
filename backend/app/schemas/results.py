from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FactStatus(str, Enum):
    """Factuality and epistemic status of a reported metric."""
    VERIFIED = "verified"
    ESTIMATED = "estimated"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    NOT_RUN = "not_run"


class SectionStatus(str, Enum):
    """Operational status of a major analysis pipeline section."""
    AVAILABLE = "available"
    ESTIMATED = "estimated"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    NOT_RUN = "not_run"


class ReportLanguage(str, Enum):
    """Supported report generation languages."""
    EN = "en"
    NE = "ne"


class FactItem(BaseModel):
    """
    Factual wrapper for an individual reportable value.
    Ensures that values, calculation methods, sources, and blocked states
    are strictly delineated so the LLM cannot hallucinate truth.
    """
    value: Optional[Any] = None
    status: FactStatus
    source: str = Field(..., description="Upstream component that produced the value")
    method: Optional[str] = Field(default=None, description="Algorithmic or spatial measurement method")
    reason: Optional[str] = Field(default=None, description="Explanation when status is blocked or unavailable")


class AnalysisMetadata(BaseModel):
    """Top-level execution metadata for the analysis job."""
    job_id: str
    aoi: Dict[str, Any]
    flood_date: str
    processing_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    report_language: str = "en"
    source_data_status: str = Field(..., description="'calibrated_raster_available', 'satellite_binary_data_blocked', or 'test_fixture'")


class SatelliteResults(BaseModel):
    """Module 2 & 3: Satellite scene selection and ARD compilation metrics."""
    status: SectionStatus
    sentinel_1_used: bool = False
    sentinel_2_used: bool = False
    pre_event_scene: Optional[str] = None
    post_event_scene: Optional[str] = None
    orbit_direction: Optional[str] = None
    relative_orbit: Optional[int] = None
    acquisition_dates: List[str] = Field(default_factory=list)
    data_availability: str = "unknown"
    reason: Optional[str] = None


class SegmentationResults(BaseModel):
    """Module 4: AI Flood & Debris semantic segmentation results."""
    status: SectionStatus
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    training_dataset: Optional[str] = None
    flood_area_km2: Optional[float] = None
    flood_percentage: Optional[float] = None
    flood_pixel_count: Optional[int] = None
    polygon_count: Optional[int] = None
    confidence_information: Optional[str] = None
    reason: Optional[str] = None


class GeospatialResults(BaseModel):
    """Module 5: Geospatial infrastructure spatial exposure overlay."""
    status: SectionStatus
    flood_region_count: Optional[int] = None
    affected_buildings: Optional[Dict[str, Any]] = None
    affected_roads: Optional[Dict[str, Any]] = None
    affected_bridges: Optional[Dict[str, Any]] = None
    change_zone_count: Optional[int] = None
    estimated_exposure_metrics: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


class ConnectivityResults(BaseModel):
    """Module 6: Road network routing, settlement isolation, and hospital access."""
    status: SectionStatus
    road_count: Optional[int] = None
    blocked_road_count: Optional[int] = None
    partially_affected_road_count: Optional[int] = None
    isolated_settlement_count: Optional[int] = None
    flood_isolated_settlement_count: Optional[int] = None
    pre_existingly_disconnected_count: Optional[int] = None
    hospital_access_lost_count: Optional[int] = None
    hospital_access_preserved_count: Optional[int] = None
    detour_statistics: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


class ProvenanceInfo(BaseModel):
    """Data provenance, attribution, and validation boundaries."""
    satellite_sources: List[str] = Field(
        default_factory=lambda: ["Copernicus Sentinel-1 SAR", "Copernicus Sentinel-2 MSI"]
    )
    osm_source: str = "OpenStreetMap contributors (Pre-event infrastructure baseline)"
    model_source: str = "FloodUNet (PyTorch 4-stage residual U-Net)"
    training_dataset: str = "Kuro Siwo Benchmark (NeurIPS 2024 / Orion-AI-Lab)"
    attribution_information: List[str] = Field(
        default_factory=lambda: [
            "© Copernicus Sentinel data (2026)",
            "© OpenStreetMap contributors",
            "Kuro Siwo dataset / Orion-AI-Lab",
        ]
    )
    validation_sources: List[str] = Field(
        default_factory=lambda: [
            "Copernicus Emergency Management Service (EMSR927 - validation only)"
        ]
    )
    emsr927_note: str = (
        "Copernicus EMSR927 is strictly reserved as an external post-event validation reference "
        "and is NOT an analysis input."
    )


class StructuredAnalysisResult(BaseModel):
    """
    Consolidated, stable machine-readable record combining outputs from Modules 1 through 6.
    Acts as the single factual source of truth for situation reporting and dashboard visualization.
    """
    job_id: str
    overall_status: str = Field(..., description="'completed', 'completed_with_limitations', 'satellite_binary_data_blocked', or 'failed'")
    source_data_status: str
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: AnalysisMetadata
    satellite: SatelliteResults
    segmentation: SegmentationResults
    geospatial: GeospatialResults
    connectivity: ConnectivityResults
    provenance: ProvenanceInfo = Field(default_factory=ProvenanceInfo)
    limitations: List[str] = Field(default_factory=list)
    facts: Dict[str, FactItem] = Field(default_factory=dict, description="Factuality-classified key-value registry")


# ============================================================================
# Situation Report Models
# ============================================================================

class ReportSection(BaseModel):
    """Individual section of a humanitarian situation report."""
    section_id: str
    title: str
    content: str
    key_metrics: Dict[str, Any] = Field(default_factory=dict)
    caveats: List[str] = Field(default_factory=list)


class ReportMetadata(BaseModel):
    """Execution and audit metadata for the generated situation report."""
    report_id: str
    job_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    language: str
    report_status: str = Field(..., description="'generated_llm', 'fallback', 'llm_unavailable', or 'blocked'")
    llm_used: bool = False
    model_name: Optional[str] = None
    structured_result_version: str = "1.0.0"
    data_status: str
    limitations_present: bool = True


class SituationReport(BaseModel):
    """Structured multilingual humanitarian situation report."""
    title: str
    executive_summary: str
    sections: List[ReportSection]
    metadata: ReportMetadata
    disclaimer: str = (
        "This report is generated from satellite observations and OpenStreetMap road network models. "
        "All damage, inundation, and road blockage statements represent geodetic estimates, not confirmed "
        "physical ground truth or structural destruction."
    )


class ReportGenerateRequest(BaseModel):
    """Options for situation report generation."""
    language: str = Field(default="en", description="Target ISO language code ('en' or 'ne')")
    force_fallback: bool = Field(default=False, description="Bypass LLM and generate deterministic rule-based report")
    llm_provider: Optional[str] = Field(default=None, description="Optional LLM provider override ('mock', 'gemini', 'openai')")


class ReportResponse(BaseModel):
    """API response envelope for situation report endpoints."""
    job_id: str
    language: str
    report_status: str
    llm_used: bool
    report: SituationReport
    metadata: ReportMetadata
