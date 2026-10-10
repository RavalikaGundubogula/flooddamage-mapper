from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.scene_selection import SelectedScenePair


class ProductDownloadRequest(BaseModel):
    """Request payload to initiate satellite product download."""
    product_id: Optional[str] = Field(None, description="Specific SAFE product name or CDSE product UUID")
    collection: Optional[str] = Field(None, description="SENTINEL-1 or SENTINEL-2 (optional if inferable from name)")
    product_ids: Optional[List[str]] = Field(None, description="List of product IDs or names to download")
    scene_pair: Optional[SelectedScenePair] = Field(None, description="SelectedScenePair from scene selector")


class ProductDownloadResult(BaseModel):
    """Result and status of downloading a single satellite product."""
    product_id: str = Field(..., description="Product SAFE name or identifier")
    cdse_id: Optional[str] = Field(None, description="Copernicus CDSE product GUID")
    collection: str = Field(..., description="Constellation: SENTINEL-1 or SENTINEL-2")
    status: str = Field(
        ...,
        description="'completed', 'cached', 'authentication_required', 'downloading', or 'failed'"
    )
    file_path: Optional[str] = Field(None, description="Local filesystem path to the downloaded archive or directory")
    file_size_bytes: Optional[int] = Field(None, description="Downloaded file size in bytes")
    checksum_sha256: Optional[str] = Field(None, description="Computed SHA-256 checksum of the downloaded file")
    download_url: Optional[str] = Field(None, description="Target download URL in Copernicus CDSE Zipper service")
    auth_configured: bool = Field(..., description="Whether valid Copernicus credentials are configured")
    message: str = Field(..., description="Status description or action instructions")


class DownloadSummaryResponse(BaseModel):
    """Aggregated status of requested satellite product downloads."""
    total_requested: int
    completed_count: int
    cached_count: int
    auth_required_count: int
    failed_count: int
    auth_configured: bool
    results: List[ProductDownloadResult]


class StoredDownloadItem(BaseModel):
    """Information regarding a raw satellite product archive stored locally."""
    product_id: str
    collection: str
    file_name: str
    file_path: str
    file_size_bytes: int
    created_at: datetime
    checksum_sha256: Optional[str] = None
    metadata_path: Optional[str] = None


class StoredDownloadsResponse(BaseModel):
    """Listing of satellite products cached on local disk."""
    total_products: int
    total_size_bytes: int
    sentinel1_count: int
    sentinel2_count: int
    products: List[StoredDownloadItem]
