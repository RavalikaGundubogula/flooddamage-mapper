from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.clipping import PairClippingResponse
from app.schemas.scene_selection import SelectedScenePair


class OpticalIndexStatistics(BaseModel):
    """Statistical distribution of an optical spectral index (e.g. NDWI, MNDWI)."""
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    water_pixel_count: int
    water_area_pct: float


class PreprocessedOpticalBand(BaseModel):
    """Metadata for an individual optical raster band or calculated spectral index."""
    band_name: str = Field(..., description="Band or index name: rgb, ndwi, mndwi, diff_ndwi, cloud_mask")
    timing: str = Field(..., description="'pre_event', 'post_event', or 'difference'")
    file_path: Optional[str] = Field(None, description="Local GeoTIFF path of optical band/index")
    width: int = Field(..., description="Raster width in pixels")
    height: int = Field(..., description="Raster height in pixels")
    description: str = Field(..., description="Description of the index or composite")
    statistics: Optional[OpticalIndexStatistics] = Field(None, description="Index statistics and water thresholding")


class Sentinel2PreprocessRequest(BaseModel):
    """Request payload for Sentinel-2 optical preprocessing and NDWI analysis."""
    scene_pair: Optional[SelectedScenePair] = None
    clipped_pair: Optional[PairClippingResponse] = None
    aoi: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    cloud_threshold_pct: float = Field(default=50.0, ge=0.0, le=100.0, description="Max allowable scene cloud cover")
    ndwi_threshold: float = Field(default=0.1, ge=-1.0, le=1.0, description="NDWI threshold for water delineation (typically 0.0 to 0.15)")


class Sentinel2PreprocessResponse(BaseModel):
    """Output package of calibrated Sentinel-2 optical indices and cloud masking."""
    product_id_pre: Optional[str] = None
    product_id_post: Optional[str] = None
    status: str = Field(..., description="'processed', 'contract_prepared', 'skipped_cloudy', or 'failed'")
    is_optical_useful: bool = Field(..., description="Whether optical data met quality and cloud thresholds")
    reason: str = Field(..., description="Detailed explanation of optical usability decision")
    channels: List[PreprocessedOpticalBand] = Field(default_factory=list)
    water_detection_summary: Dict[str, Any] = Field(default_factory=dict)
    grid_shape: List[int] = Field(default_factory=list)
    crs: str = Field(default="EPSG:4326")
    output_dir: Optional[str] = None
    message: str
