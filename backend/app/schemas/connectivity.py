from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RoadEdgeStatus(str, Enum):
    """Estimated road segment status based on spatial flood exposure."""
    ESTIMATED_OPEN = "estimated_open"
    PARTIALLY_AFFECTED = "partially_affected"
    ESTIMATED_BLOCKED = "estimated_blocked"


class SettlementIsolationStatus(str, Enum):
    """Rigorous classification of settlement isolation."""
    UNAFFECTED = "unaffected"
    FLOOD_ISOLATED = "flood_isolated"
    PRE_EXISTINGLY_DISCONNECTED = "pre_existingly_disconnected"


class DetourClassification(str, Enum):
    """Detour severity for settlements that remain connected post-flood."""
    NO_CHANGE = "no_change"
    MINOR_DETOUR = "minor_detour"
    MAJOR_DETOUR = "major_detour"
    DISCONNECTED = "disconnected"


class HospitalAccessStatus(str, Enum):
    """Hospital accessibility classification from settlement road network."""
    HOSPITAL_ACCESS_PRESERVED = "hospital_access_preserved"
    HOSPITAL_ACCESS_LOST = "hospital_access_lost"
    NO_BASELINE_HOSPITAL_ACCESS = "no_baseline_hospital_access"


class RoadNetworkSummary(BaseModel):
    """Aggregated metrics for the pre-event and post-flood road network graph."""
    total_nodes: int = Field(..., description="Total intersection and terminal nodes in road graph")
    total_edges: int = Field(..., description="Total directed/traversable road segment edges")
    open_edges: int = Field(..., description="Road edges unaffected by floodwaters")
    partially_affected_edges: int = Field(..., description="Edges partially affected (< threshold) remaining traversable")
    blocked_edges: int = Field(..., description="Edges estimated blocked (>= threshold) removed in post-flood graph")
    baseline_components: int = Field(..., description="Number of connected components in pre-event road network")
    post_flood_components: int = Field(..., description="Number of connected components after removing blocked edges")
    newly_disconnected_components: int = Field(default=0, description="Additional components formed by flood fragmentation")


class SettlementSummary(BaseModel):
    """Summary of settlement connectivity before and after flood impact."""
    total: int = Field(..., description="Total settlements evaluated within or near AOI")
    baseline_connected: int = Field(..., description="Settlements with a valid pre-event route to destination")
    flood_isolated: int = Field(..., description="Settlements connected in baseline but disconnected post-flood")
    pre_existingly_disconnected: int = Field(..., description="Settlements lacking pre-event route (not flood-caused)")
    unaffected: int = Field(..., description="Settlements connected with no significant route change (<= 5%)")
    minor_detour: int = Field(..., description="Settlements connected requiring minor detour (> 5% and <= 25%)")
    major_detour: int = Field(..., description="Settlements connected requiring major detour (> 25%)")


class HospitalAccessSummary(BaseModel):
    """Summary of healthcare accessibility changes across settlements."""
    settlements_with_baseline_access: int = Field(..., description="Settlements with road access to at least one hospital pre-event")
    access_preserved: int = Field(..., description="Settlements retaining road access to a hospital post-flood")
    hospital_access_lost: int = Field(..., description="Settlements that lost hospital access due to estimated blocked roads")
    no_baseline_access: int = Field(..., description="Settlements that had no pre-event road route to any hospital")


class SettlementConnectivityDetail(BaseModel):
    """Individual settlement route reachability, detour metrics, and hospital access."""
    settlement_id: str
    name: Optional[str] = None
    latitude: float
    longitude: float
    place_type: str = "settlement"
    population: Optional[int] = Field(default=None, description="Reported OSM population only; None if unavailable")
    nearest_node_id: Optional[str] = None
    distance_to_nearest_node_m: Optional[float] = None
    destination_id: Optional[str] = None
    destination_name: Optional[str] = None
    destination_selection_reason: Optional[str] = None
    baseline_connected: bool
    post_flood_connected: bool
    status: SettlementIsolationStatus
    baseline_distance_km: Optional[float] = None
    post_flood_distance_km: Optional[float] = None
    additional_distance_km: Optional[float] = None
    detour_percentage: Optional[float] = None
    detour_classification: DetourClassification
    nearest_hospital_id: Optional[str] = None
    nearest_hospital_name: Optional[str] = None
    baseline_hospital_access: bool
    post_flood_hospital_access: bool
    hospital_access_lost: bool
    hospital_access_status: HospitalAccessStatus
    baseline_hospital_distance_km: Optional[float] = None
    post_flood_hospital_distance_km: Optional[float] = None


class HospitalDetail(BaseModel):
    """OpenStreetMap hospital facility identified in or near the AOI."""
    hospital_id: str
    name: Optional[str] = None
    latitude: float
    longitude: float
    amenity_type: str = "hospital"
    nearest_node_id: Optional[str] = None
    distance_to_nearest_node_m: Optional[float] = None
    osm_tags: Dict[str, Any] = Field(default_factory=dict)


class RoadEdgeDetail(BaseModel):
    """Individual graph edge with exposure metrics and blockage determination."""
    edge_id: str
    road_id: str
    osm_id: Optional[str] = None
    name: Optional[str] = None
    highway_type: str = "road"
    from_node: str
    to_node: str
    length_m: float
    flooded_length_m: float
    affected_percentage: float
    is_bridge: bool = False
    status: RoadEdgeStatus
    geometry: Dict[str, Any]


class ConnectivityAnalysisRequest(BaseModel):
    """Configuration options for Module 6 road network and settlement connectivity analysis."""
    job_id: Optional[str] = Field(default=None, description="Analysis job identifier with completed Module 5 geospatial results.")
    road_blocked_threshold_pct: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Minimum flood intersection percentage to classify road as estimated_blocked (default: 50%).",
    )
    detour_minor_threshold_pct: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description="Maximum detour percentage to classify route change as no_change (default: 5%).",
    )
    detour_major_threshold_pct: float = Field(
        default=25.0,
        ge=0.0,
        le=100.0,
        description="Threshold above which route detour is classified as major_detour (default: 25%).",
    )
    osm_geojson_path: Optional[str] = Field(
        default=None,
        description="Optional local path to pre-downloaded OpenStreetMap GeoJSON file containing settlements and hospitals.",
    )
    settlements_geojson_path: Optional[str] = Field(
        default=None,
        description="Optional separate local file containing settlement features.",
    )
    hospitals_geojson_path: Optional[str] = Field(
        default=None,
        description="Optional separate local file containing hospital features.",
    )


class ConnectivityAnalysisResponse(BaseModel):
    """Complete Module 6 road network, settlement isolation, and hospital accessibility analysis."""
    job_id: str
    status: str = Field(..., description="'completed', 'satellite_binary_data_blocked', or 'failed'")
    source_data_status: str = Field(..., description="'calibrated_raster_available', 'satellite_binary_data_blocked', or 'test_fixture'")
    crs: str = "EPSG:4326"
    blocking_threshold_pct: float = 50.0
    road_network: RoadNetworkSummary
    settlements: SettlementSummary
    hospital_access: HospitalAccessSummary
    settlement_details: List[SettlementConnectivityDetail] = Field(default_factory=list)
    hospital_details: List[HospitalDetail] = Field(default_factory=list)
    road_details: List[RoadEdgeDetail] = Field(default_factory=list)
    blocked_roads: Dict[str, Any] = Field(..., description="GeoJSON FeatureCollection of estimated blocked road segments")
    partially_affected_roads: Dict[str, Any] = Field(..., description="GeoJSON FeatureCollection of partially affected road segments")
    isolated_settlements: Dict[str, Any] = Field(..., description="GeoJSON FeatureCollection of flood-isolated settlements")
    hospital_access_lost_settlements: Dict[str, Any] = Field(..., description="GeoJSON FeatureCollection of settlements with lost hospital access")
    settlements_connectivity: Dict[str, Any] = Field(..., description="GeoJSON FeatureCollection of all settlements with connectivity metrics")
    attribution: List[str] = Field(
        default_factory=lambda: [
            "© Copernicus Sentinel data",
            "© OpenStreetMap contributors",
            "Kuro Siwo dataset / Orion-AI-Lab",
        ]
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: str
    disclaimer: str = (
        "This is a network accessibility estimate based on spatial flood exposure, "
        "not physical confirmation of road destruction or impassability."
    )
