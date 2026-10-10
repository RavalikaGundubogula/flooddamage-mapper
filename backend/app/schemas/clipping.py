from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.scene_selection import SelectedScenePair


class ClipSpecification(BaseModel):
    """Spatial and mathematical grid specification for cropping to an AOI."""
    aoi_geojson: Dict[str, Any] = Field(..., description="Target GeoJSON Polygon")
    bounding_box: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    target_crs: str = Field(default="EPSG:4326", description="Target Coordinate Reference System")
    pixel_resolution_meters: float = Field(default=10.0, description="Spatial resolution in ground meters")
    pixel_resolution_degrees: float = Field(..., description="Approximate spatial resolution in decimal degrees")
    dimensions: List[int] = Field(..., description="Grid matrix dimensions [height, width] in pixels")
    affine_transform: List[float] = Field(..., description="Affine transform coefficients [a, b, c, d, e, f]")
    aoi_area_sqkm: float = Field(..., description="Area of interest footprint in square kilometers")


class ClippedBandMetadata(BaseModel):
    """Metadata for an individual cropped raster measurement band."""
    band_name: str = Field(..., description="Band identifier (e.g. VV, VH, B04, B08, B11)")
    source_path: Optional[str] = Field(None, description="Path to original raw file or archive entry")
    clipped_path: Optional[str] = Field(None, description="Local path to cropped GeoTIFF")
    width: int = Field(..., description="Width in pixels")
    height: int = Field(..., description="Height in pixels")
    crs: str = Field(..., description="Coordinate Reference System")
    transform: List[float] = Field(..., description="Affine transform matrix")
    data_type: str = Field(default="float32", description="Data type of raster pixels")
    statistics: Optional[Dict[str, float]] = Field(None, description="Raster statistics (min, max, mean, std)")


class ClippedSceneResult(BaseModel):
    """Result of clipping a single satellite acquisition to the AOI."""
    product_id: str
    scene_type: str = Field(..., description="'before' or 'after'")
    collection: str = Field(..., description="SENTINEL-1 or SENTINEL-2")
    status: str = Field(..., description="'clipped', 'prepared_contract', or 'source_missing'")
    clip_specification: ClipSpecification
    bands: List[ClippedBandMetadata] = Field(default_factory=list)
    output_directory: Optional[str] = None
    spatial_alignment_verified: bool = Field(default=True)
    message: str


class PairClippingRequest(BaseModel):
    """Request to clip satellite scene pairs to an AOI."""
    scene_pair: Optional[SelectedScenePair] = None
    aoi: Optional[Dict[str, Any]] = None
    resolution_meters: float = Field(default=10.0, ge=1.0, le=100.0)
    job_id: Optional[str] = None


class PairClippingResponse(BaseModel):
    """Co-registered clipped raster pair ready for subsequent preprocessing and AI segmentation."""
    satellite: str
    aoi: Dict[str, Any]
    status: str = Field(..., description="'ready', 'specification_prepared', or 'failed'")
    before_scene_clip: ClippedSceneResult
    after_scene_clip: ClippedSceneResult
    identical_grid_alignment: bool = Field(..., description="True if before and after grids have matching dimensions and bounds")
    grid_shape: List[int] = Field(..., description="[height, width] of the aligned rasters")
    message: str
