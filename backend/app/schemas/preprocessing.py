from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.clipping import PairClippingResponse
from app.schemas.scene_selection import SelectedScenePair


class SarChannelStatistics(BaseModel):
    """Statistical summary of a SAR backscatter channel in decibels (dB)."""
    min_db: float
    max_db: float
    mean_db: float
    std_db: float


class PreprocessedSarBand(BaseModel):
    """Metadata for an individual calibrated and filtered SAR raster band."""
    band_name: str = Field(..., description="Channel name: pre_vv, pre_vh, post_vv, post_vh, diff_vv, diff_vh")
    polarization: str = Field(..., description="Polarization channel: VV or VH or RATIO")
    timing: str = Field(..., description="'pre_event', 'post_event', or 'difference'")
    file_path: Optional[str] = Field(None, description="Local GeoTIFF path of preprocessed channel")
    width: int = Field(..., description="Raster width in pixels")
    height: int = Field(..., description="Raster height in pixels")
    unit: str = Field(default="dB", description="Radiometric unit: decibels (dB)")
    statistics: Optional[SarChannelStatistics] = Field(None, description="Radiometric distribution statistics")


class Sentinel1PreprocessRequest(BaseModel):
    """Request payload to calibrate, filter, and compute SAR difference channels."""
    scene_pair: Optional[SelectedScenePair] = None
    clipped_pair: Optional[PairClippingResponse] = None
    aoi: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    speckle_filter_window: int = Field(default=3, ge=1, le=7, description="Spatial filter window size (e.g. 3 for 3x3)")
    calibrate_to_db: bool = Field(default=True, description="Convert linear digital numbers to backscatter decibels (dB)")


class Sentinel1PreprocessResponse(BaseModel):
    """Multi-channel calibrated SAR package ready for AI flood and debris segmentation."""
    product_id_pre: str
    product_id_post: str
    status: str = Field(..., description="'calibrated', 'contract_prepared', or 'failed'")
    channels: List[PreprocessedSarBand]
    change_summary: Dict[str, Any] = Field(
        ...,
        description="Summary of flood-characteristic backscatter drop (negative dB difference)"
    )
    filter_applied: str = Field(..., description="Speckle reduction filter description")
    grid_shape: List[int] = Field(..., description="[height, width] of the analysis-ready raster stack")
    crs: str = Field(default="EPSG:4326")
    output_dir: str
    message: str
