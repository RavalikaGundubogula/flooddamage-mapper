from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.clipping import ClipSpecification
from app.schemas.optical_preprocessing import Sentinel2PreprocessResponse
from app.schemas.preprocessing import Sentinel1PreprocessResponse


class SpatialReferenceMetadata(BaseModel):
    """Spatial coordinate reference, grid matrix dimensions, and geographic bounds."""
    crs: str = Field(default="EPSG:4326")
    grid_shape: List[int] = Field(..., description="[height, width] pixel matrix")
    affine_transform: List[float] = Field(..., description="6-element affine transformation matrix")
    bounding_box: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    area_sqkm: float = Field(..., description="Ground coverage area in square kilometers")


class AnalysisReadyPackage(BaseModel):
    """
    Standardized Analysis-Ready Data (ARD) contract handed off to
    Module 4: AI Flood & Debris Segmentation.
    """
    job_id: str
    flood_date: str
    aoi: Dict[str, Any]
    status: str = Field(..., description="'ready', 'contract_ready', or 'failed'")
    strategy_used: str = Field(..., description="'sentinel-1', 'sentinel-2', or 'sentinel-1-and-sentinel-2'")
    spatial_reference: SpatialReferenceMetadata
    sar_ard: Optional[Sentinel1PreprocessResponse] = Field(None, description="Calibrated dual-pol SAR stack")
    optical_ard: Optional[Sentinel2PreprocessResponse] = Field(None, description="Spectral NDWI/MNDWI optical stack")
    channels_available: List[str] = Field(..., description="List of all ready-to-infer channels")
    recommended_segmentation_model: str = Field(
        ...,
        description="Downstream AI model recommendation: 'sar_unet_flood', 'multimodal_flood_debris', etc."
    )
    manifest_file: Optional[str] = Field(None, description="Local path to serialized ARD manifest JSON")
    created_at: datetime
    message: str


class PrepareArdRequest(BaseModel):
    """Request payload to trigger end-to-end Module 3 Analysis-Ready Data preparation."""
    aoi: Optional[Any] = Field(None, description="GeoJSON Polygon or Bounding Box (defaults to Nepal basin)")
    flood_date: str = Field(default="2024-08-26", description="Disaster event date YYYY-MM-DD")
    job_id: Optional[str] = Field(None, description="Optional job ID (generates UUID if omitted)")
    resolution_meters: float = Field(default=10.0, ge=1.0, le=100.0)
    speckle_filter_window: int = Field(default=3, ge=1, le=7)
    cloud_threshold_pct: float = Field(default=50.0, ge=0.0, le=100.0)
