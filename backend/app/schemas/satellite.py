from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SceneMetadata(BaseModel):
    """Normalized metadata for a single satellite acquisition scene."""
    product_id: str = Field(..., description="Unique product identifier (e.g. SAFE package name)")
    collection: str = Field(..., description="Satellite constellation/collection (e.g. SENTINEL-1, SENTINEL-2)")
    platform: Optional[str] = Field(None, description="Specific spacecraft (e.g. S1A, S1B, S2A, S2B)")
    acquisition_datetime: datetime = Field(..., description="Acquisition timestamp in UTC")
    geometry: Optional[Dict[str, Any]] = Field(None, description="Footprint geometry of the scene (GeoJSON)")
    cloud_cover: Optional[float] = Field(None, description="Cloud cover percentage (0-100) for optical imagery")
    orbit_direction: Optional[str] = Field(None, description="ASCENDING or DESCENDING")
    relative_orbit: Optional[int] = Field(None, description="Track/Relative orbit number")
    polarization: Optional[str] = Field(None, description="Radar polarization (e.g. VV, VH, VV&VH)")
    product_type: str = Field(..., description="Product type (e.g. IW_GRDH_1S, S2MSI2A, S2MSI1C)")
    catalog_id: str = Field(default="copernicus_cdse", description="Identifier of the discovery catalog source")
    extra_attributes: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Raw metadata attributes")


class Sentinel1Candidate(BaseModel):
    """Sentinel-1 scene enriched with flood change detection suitability flags."""
    scene: SceneMetadata
    timing: str = Field(..., description="'pre_event' or 'post_event'")
    days_from_event: float = Field(..., description="Temporal offset in days from the flood event date")
    is_grd: bool = Field(..., description="Whether product is Ground Range Detected (GRD), the standard for SAR flood mapping")
    relative_orbit: Optional[int] = Field(None, description="Relative orbit track for geometric consistency")
    orbit_direction: Optional[str] = Field(None, description="ASCENDING or DESCENDING")
    polarization: Optional[str] = Field(None, description="Polarization channels (e.g. VV&VH)")


class Sentinel1DiscoveryResponse(BaseModel):
    """Discovery summary and candidate scenes for Sentinel-1."""
    collection: str = "SENTINEL-1"
    flood_date: str
    temporal_window: Dict[str, str] = Field(..., description="Search window start and end dates")
    total_candidates: int
    pre_event_count: int
    post_event_count: int
    grd_candidate_count: int
    candidates: List[Sentinel1Candidate]


class Sentinel2Candidate(BaseModel):
    """Sentinel-2 optical scene enriched with cloud cover and quality suitability."""
    scene: SceneMetadata
    timing: str = Field(..., description="'pre_event' or 'post_event'")
    days_from_event: float = Field(..., description="Temporal offset in days from the flood event date")
    cloud_cover: Optional[float] = Field(None, description="Cloud cover percentage (0-100)")
    is_l2a: bool = Field(..., description="Whether product is Level-2A (Bottom of Atmosphere surface reflectance)")
    optical_usability: str = Field(
        ...,
        description="'clear' (<20% cloud), 'moderate' (20-50% cloud), or 'cloudy' (>50% cloud)"
    )


class Sentinel2DiscoveryResponse(BaseModel):
    """Discovery summary and candidate scenes for Sentinel-2."""
    collection: str = "SENTINEL-2"
    flood_date: str
    temporal_window: Dict[str, str] = Field(..., description="Search window start and end dates")
    total_candidates: int
    pre_event_count: int
    post_event_count: int
    usable_candidate_count: int = Field(..., description="Number of scenes with <= 50% cloud cover")
    average_cloud_cover: Optional[float] = Field(None, description="Average cloud cover across all discovered scenes")
    candidates: List[Sentinel2Candidate]


class SatelliteDiscoveryRequest(BaseModel):
    """User request for testing satellite candidate discovery."""
    aoi: Optional[Any] = Field(
        None,
        description="Optional AOI. Defaults to Nepal Trishuli/Bhote Koshi region if omitted."
    )
    flood_date: str = Field(
        default="2024-08-26",
        description="Event date in YYYY-MM-DD format (default: 2024-08-26 Nepal floods)"
    )
    days_before: int = Field(default=15, ge=1, le=60, description="Pre-event window in days")
    days_after: int = Field(default=7, ge=0, le=30, description="Post-event window in days")
    max_results: int = Field(default=20, ge=1, le=100, description="Max candidate scenes to return")


class DiscoveryQuery(BaseModel):
    """Search parameters for satellite scene discovery."""
    aoi: Dict[str, Any] = Field(..., description="GeoJSON Polygon area of interest")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD or ISO timestamp")
    end_date: str = Field(..., description="End date in YYYY-MM-DD or ISO timestamp")
    collection: str = Field(..., description="Target collection: 'SENTINEL-1' or 'SENTINEL-2'")
    max_results: int = Field(default=20, ge=1, le=100, description="Max candidate scenes to return")


class DiscoveryResult(BaseModel):
    """Results returned from satellite scene discovery."""
    collection: str
    query_start_date: str
    query_end_date: str
    total_found: int
    scenes: List[SceneMetadata]


class SatelliteSubsystemStatus(BaseModel):
    """Health and configuration status of the satellite discovery subsystem."""
    status: str = Field(..., description="Operational status: operational, degraded, or offline")
    catalog_name: str = Field(..., description="Name of the primary satellite catalog")
    catalog_endpoint: str = Field(..., description="Base API endpoint of the catalog")
    supported_satellites: List[str] = Field(..., description="List of supported satellite collections")
    auth_configured: bool = Field(..., description="Whether authentication credentials are provided")
    cache_enabled: bool = Field(default=True, description="Whether query caching is enabled")


class Sentinel1Suitability(BaseModel):
    """Suitability assessment for Sentinel-1 radar change detection."""
    is_suitable: bool
    total_candidates: int
    grd_candidates: int
    pre_event_count: int
    post_event_count: int
    has_geometry_pairs: bool
    assessment: str


class Sentinel2Suitability(BaseModel):
    """Suitability assessment for Sentinel-2 optical analysis."""
    is_suitable: bool
    total_candidates: int
    usable_candidates: int
    pre_event_count: int
    post_event_count: int
    average_cloud_cover: Optional[float]
    best_post_event_cloud_cover: Optional[float]
    assessment: str


class RoutingDecisionResponse(BaseModel):
    """Structured decision produced by the Smart Satellite Router."""
    strategy: str = Field(
        ...,
        description="'sentinel-1', 'sentinel-2', 'sentinel-1-and-sentinel-2', or 'no-suitable-data'"
    )
    reason: str = Field(..., description="Detailed justification for the chosen satellite strategy")
    flood_date: str
    aoi: Dict[str, Any]
    sentinel_1: Sentinel1Suitability
    sentinel_2: Sentinel2Suitability
    warnings: List[str] = Field(default_factory=list)

