from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.satellite import SceneMetadata


class SelectedScenePair(BaseModel):
    """Pair of before/after satellite scenes selected for flood change detection."""
    satellite: str = Field(..., description="Constellation: 'SENTINEL-1' or 'SENTINEL-2'")
    before_scene: SceneMetadata = Field(..., description="Pre-flood reference baseline scene")
    after_scene: SceneMetadata = Field(..., description="Post-flood event disaster scene")
    orbit_compatibility: bool = Field(
        ...,
        description="True if both scenes share identical relative orbit track and look geometry (SAR) or projection"
    )
    relative_orbit: Optional[int] = Field(None, description="Shared relative orbit track number (Sentinel-1)")
    orbit_direction: Optional[str] = Field(None, description="Shared pass direction: ASCENDING or DESCENDING")
    temporal_separation_days: float = Field(..., description="Total days elapsed between before and after acquisitions")
    before_days_from_event: float = Field(..., description="Days between before scene and flood event (negative)")
    after_days_from_event: float = Field(..., description="Days between flood event and after scene (positive)")
    quality_notes: str = Field(..., description="Technical notes regarding product type, polarization, and cloud metrics")


class SceneSelectionResponse(BaseModel):
    """Structured response containing optimal before/after scene pairs for analysis."""
    status: str = Field(..., description="'selected' if viable pair found, 'insufficient_data' otherwise")
    event_date: str = Field(..., description="Event/flood date YYYY-MM-DD")
    selected_strategy: str = Field(..., description="Strategy chosen by router ('sentinel-1', 'sentinel-2', etc.)")
    selection_reason: str = Field(..., description="Detailed explanation of the pair selection criteria")
    sentinel_1_selection: Optional[SelectedScenePair] = Field(
        None,
        description="Selected Sentinel-1 SAR before/after pair, if applicable"
    )
    sentinel_2_selection: Optional[SelectedScenePair] = Field(
        None,
        description="Selected Sentinel-2 optical before/after pair, if applicable"
    )
    warnings: List[str] = Field(default_factory=list, description="Operational warnings or quality flags")
    aoi: Dict[str, Any] = Field(..., description="Normalized GeoJSON Polygon area of interest")
