from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SegmentationRequest(BaseModel):
    """Request payload to trigger AI segmentation on an ARD product."""
    job_id: Optional[str] = Field(default=None, description="Job identifier with prepared ARD data.")
    model_name: Optional[str] = Field(
        default="flood_unet_s1",
        description="Name or checkpoint of the semantic segmentation model to use.",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.05,
        le=0.95,
        description="Probability threshold for classifying pixels as flooded water.",
    )
    min_connected_pixels: int = Field(
        default=10,
        ge=1,
        description="Minimum pixel cluster size to filter isolated noise artifacts.",
    )


class SegmentationMetrics(BaseModel):
    """Validation metrics for semantic segmentation evaluation."""
    iou: float = Field(..., description="Intersection over Union (Jaccard Index).")
    dice_score: float = Field(..., description="Dice Similarity Coefficient / F1-score.")
    precision: float = Field(..., description="Precision: True Positives / (True Positives + False Positives).")
    recall: float = Field(..., description="Recall: True Positives / (True Positives + False Negatives).")


class AffectedAreaSummary(BaseModel):
    """Spatial metrics and flood extent calculations."""
    flood_pixel_count: int = Field(..., description="Number of pixels classified as flooded.")
    total_valid_pixels: int = Field(..., description="Total non-cloud, non-nodata pixels evaluated.")
    flood_percentage: float = Field(..., description="Percentage of valid AOI area covered by flood.")
    flood_area_m2: float = Field(..., description="Total flooded area in square meters.")
    flood_area_sqkm: float = Field(..., description="Total flooded area in square kilometers.")


class SegmentationResponse(BaseModel):
    """Full segmentation result payload including raster & vector representations."""
    job_id: str
    status: str = Field(..., description="'completed', 'failed', or 'insufficient_data'")
    model_name: str
    model_architecture: str = Field(
        default="FloodUNet (Encoder-Decoder with Skip Connections)",
        description="Architecture details of the neural network used.",
    )
    confidence_threshold: float
    summary: AffectedAreaSummary
    metrics: Optional[SegmentationMetrics] = Field(
        default=None,
        description="Evaluation metrics if validation ground truth was available.",
    )
    mask_raster_path: str = Field(
        ...,
        description="Absolute path to generated GeoTIFF flood mask.",
    )
    geojson_features_count: int = Field(
        ...,
        description="Number of vectorized flood polygon features.",
    )
    geojson: Dict[str, Any] = Field(
        ...,
        description="Vectorized flood polygons in GeoJSON FeatureCollection format.",
    )
    crs: str = Field(default="EPSG:4326")
    spatial_bounds: List[float] = Field(
        ...,
        description="Bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84.",
    )
    created_at: datetime
    message: str


class TrainModelRequest(BaseModel):
    """Parameters to launch model training on Kuro Siwo or local benchmark."""
    model_name: str = Field(default="flood_unet_s1", description="Identifier for model checkpoint.")
    dataset_dir: Optional[str] = Field(default=None, description="Custom dataset directory path.")
    epochs: int = Field(default=5, ge=1, le=100, description="Number of training epochs.")
    batch_size: int = Field(default=4, ge=1, le=32, description="Batch size for training.")
    learning_rate: float = Field(default=1e-3, gt=0, le=1.0, description="Adam optimizer learning rate.")
    seed: int = Field(default=42, description="Random seed for deterministic data splits.")


class TrainModelResponse(BaseModel):
    """Summary of model training run."""
    model_name: str
    checkpoint_path: str
    epochs_trained: int
    train_loss: float
    val_loss: float
    val_metrics: SegmentationMetrics
    samples_count: int
    train_split: int
    val_split: int
    dataset_source: str
    status: str
    trained_at: datetime
    message: str


class ModelInfo(BaseModel):
    """Metadata for registered model checkpoints."""
    model_name: str
    architecture: str
    checkpoint_path: str
    input_channels: int
    weights_size_bytes: int
    trained_at: Optional[datetime] = None
    best_val_dice: Optional[float] = None
    is_active: bool = True
