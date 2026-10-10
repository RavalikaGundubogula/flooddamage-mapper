from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BuildingExposureClassification(str, Enum):
    """Estimated building exposure category based on flood mask spatial overlap."""
    UNAFFECTED = "unaffected"
    POTENTIALLY_AFFECTED = "potentially_affected"
    HEAVILY_AFFECTED = "heavily_affected"


class RoadExposureClassification(str, Enum):
    """Estimated road segment exposure category based on flood mask intersection."""
    UNAFFECTED = "unaffected"
    POTENTIALLY_BLOCKED = "potentially_blocked"
    HEAVILY_AFFECTED = "heavily_affected"


class BridgeExposureClassification(str, Enum):
    """Estimated bridge exposure category based on flood mask intersection."""
    UNAFFECTED = "unaffected"
    POTENTIALLY_AFFECTED = "potentially_affected"
    HEAVILY_AFFECTED = "heavily_affected"


class FloodRegionProperties(BaseModel):
    """Properties for an individual polygonized floodwater extent region."""
    region_id: str
    area_m2: float
    area_km2: float
    centroid: List[float] = Field(..., description="[longitude, latitude] of centroid")
    bounding_box: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    mean_confidence: Optional[float] = None
    source_job_id: str


class ChangeZoneProperties(BaseModel):
    """Properties for a heuristic terrain change / possible debris candidate zone."""
    zone_id: str
    change_type: str = "possible_change_zone"
    area_m2: float
    area_km2: float
    centroid: List[float]
    heuristic_score: float
    change_features_used: List[str]
    source_job_id: str
    disclaimer: str = "Probabilistic terrain change candidate requiring field validation; not confirmed debris."


class BuildingImpact(BaseModel):
    """Building spatial exposure metrics against flood inundation."""
    building_id: str
    osm_id: Optional[str] = None
    name: Optional[str] = None
    building_type: str = "yes"
    area_m2: float
    flooded_overlap_m2: float
    overlap_percentage: float
    distance_to_flood_m: float
    classification: BuildingExposureClassification
    centroid: List[float]
    geometry: Dict[str, Any]


class RoadImpact(BaseModel):
    """Road segment spatial exposure metrics against flood inundation."""
    road_id: str
    osm_id: Optional[str] = None
    name: Optional[str] = None
    highway_type: str = "road"
    total_length_m: float
    flooded_length_m: float
    affected_percentage: float
    nearest_flood_region_id: Optional[str] = None
    classification: RoadExposureClassification
    geometry: Dict[str, Any]


class BridgeImpact(BaseModel):
    """Bridge spatial exposure metrics against flood inundation."""
    bridge_id: str
    osm_id: Optional[str] = None
    name: Optional[str] = None
    bridge_type: str = "bridge"
    total_length_m: float
    flooded_length_m: float
    affected_percentage: float
    nearest_flood_region_id: Optional[str] = None
    classification: BridgeExposureClassification
    geometry: Dict[str, Any]


class BuildingAssetSummary(BaseModel):
    """Aggregated exposure summary for OpenStreetMap buildings."""
    total: int
    unaffected: int
    potentially_affected: int
    heavily_affected: int
    total_area_m2: float
    affected_area_m2: float


class RoadAssetSummary(BaseModel):
    """Aggregated exposure summary for OpenStreetMap road segments."""
    total_segments: int
    unaffected: int
    potentially_blocked: int
    heavily_affected: int
    total_length_km: float
    affected_length_km: float


class BridgeAssetSummary(BaseModel):
    """Aggregated exposure summary for OpenStreetMap bridges."""
    total: int
    unaffected: int
    potentially_affected: int
    heavily_affected: int
    total_length_km: float
    affected_length_km: float


class AssetSummary(BaseModel):
    """Structured infrastructure exposure summary."""
    buildings: BuildingAssetSummary
    roads: RoadAssetSummary
    bridges: BridgeAssetSummary


class FloodExtentMetrics(BaseModel):
    """Geodetic and pixel-level metrics for segmented flood water."""
    total_aoi_area_km2: float
    valid_analysis_area_km2: float
    flooded_pixel_count: int
    total_valid_pixels: int
    flooded_area_m2: float
    flooded_area_km2: float
    flooded_percentage: float
    flood_region_count: int
    largest_flood_region_km2: float
    smallest_flood_region_km2: float
    mean_flood_region_km2: float


class ChangeZoneMetrics(BaseModel):
    """Metrics for heuristic terrain change and potential debris zones."""
    status: str = Field(..., description="'available' or 'insufficient_data'")
    total_change_zones: int
    total_area_km2: float
    largest_zone_km2: float
    message: str


class GeospatialAnalysisRequest(BaseModel):
    """Configuration options for Module 5 geospatial overlay analysis."""
    job_id: Optional[str] = Field(default=None, description="Analysis job identifier with completed segmentation.")
    min_flood_region_area_m2: float = Field(
        default=100.0,
        ge=0.0,
        description="Minimum polygon area in square meters; smaller noise components are filtered.",
    )
    building_affected_threshold_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Minimum flood overlap percentage to classify building as potentially_affected.",
    )
    building_heavy_threshold_pct: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Minimum flood overlap percentage to classify building as heavily_affected.",
    )
    road_blocked_threshold_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Minimum flood intersection percentage to classify road as potentially_blocked.",
    )
    road_heavy_threshold_pct: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Minimum flood intersection percentage to classify road as heavily_affected.",
    )
    bridge_affected_threshold_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Minimum flood intersection percentage to classify bridge as potentially_affected.",
    )
    bridge_heavy_threshold_pct: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Minimum flood intersection percentage to classify bridge as heavily_affected.",
    )
    osm_geojson_path: Optional[str] = Field(
        default=None,
        description="Optional local path to pre-downloaded OpenStreetMap GeoJSON file for offline execution.",
    )


class GeospatialAnalysisResponse(BaseModel):
    """Comprehensive Module 5 geospatial asset damage & exposure results."""
    job_id: str
    status: str = Field(..., description="'completed', 'satellite_binary_data_blocked', or 'failed'")
    source_data_status: str = Field(
        ...,
        description="'calibrated_raster_available', 'satellite_binary_data_blocked', or 'test_fixture'",
    )
    osm_status: str = Field(..., description="'available' or 'unavailable'")
    aoi: Dict[str, Any]
    crs: str = "EPSG:4326"
    flood_metrics: FloodExtentMetrics
    change_zone_metrics: ChangeZoneMetrics
    asset_summary: AssetSummary
    flood_regions: Dict[str, Any] = Field(..., description="GeoJSON FeatureCollection of discrete flood polygons")
    change_zones: Dict[str, Any] = Field(..., description="GeoJSON FeatureCollection of candidate terrain change zones")
    buildings: Dict[str, Any] = Field(..., description="GeoJSON FeatureCollection of evaluated OSM buildings")
    roads: Dict[str, Any] = Field(..., description="GeoJSON FeatureCollection of evaluated OSM roads")
    bridges: Dict[str, Any] = Field(..., description="GeoJSON FeatureCollection of evaluated OSM bridges")
    attribution: List[str] = Field(
        default_factory=lambda: [
            "© Copernicus Sentinel data",
            "© OpenStreetMap contributors",
            "Kuro Siwo dataset / Orion-AI-Lab",
        ]
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: str
