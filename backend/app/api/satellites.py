from typing import Optional
from fastapi import APIRouter, Depends, Query
from app.schemas.satellite import (
    SatelliteDiscoveryRequest,
    SatelliteSubsystemStatus,
    Sentinel1DiscoveryResponse,
    Sentinel2DiscoveryResponse,
    RoutingDecisionResponse,
)
from app.schemas.clipping import (
    ClipSpecification,
    ClippedSceneResult,
    PairClippingRequest,
    PairClippingResponse,
)
from app.schemas.download import (
    ProductDownloadRequest,
    ProductDownloadResult,
    DownloadSummaryResponse,
    StoredDownloadsResponse,
)
from app.schemas.scene_selection import SceneSelectionResponse
from app.schemas.preprocessing import (
    Sentinel1PreprocessRequest,
    Sentinel1PreprocessResponse,
)
from app.schemas.optical_preprocessing import (
    Sentinel2PreprocessRequest,
    Sentinel2PreprocessResponse,
)
from app.schemas.ard import (
    AnalysisReadyPackage,
    PrepareArdRequest,
)
from app.services.ard_service import (
    AnalysisReadyDataService,
    get_ard_service,
)
from app.services.product_downloader import (
    ProductDownloaderService,
    get_product_downloader,
)
from app.services.spatial_clipper import (
    SpatialClippingService,
    get_spatial_clipper,
)
from app.services.sar_preprocessor import (
    Sentinel1PreprocessorService,
    get_sar_preprocessor,
)
from app.services.optical_preprocessor import (
    Sentinel2PreprocessorService,
    get_optical_preprocessor,
)
from app.services.satellite_service import (
    SatelliteDiscoveryService,
    get_satellite_service,
)
from app.services.scene_selector import (
    SceneSelectionService,
    get_scene_selection_service,
)

router = APIRouter(prefix="/satellites", tags=["Satellites"])


@router.get(
    "/status",
    response_model=SatelliteSubsystemStatus,
    summary="Satellite discovery subsystem status",
    description="Returns configuration, supported satellite platforms, and operational status of the catalog client.",
)
def get_status(
    service: SatelliteDiscoveryService = Depends(get_satellite_service),
) -> SatelliteSubsystemStatus:
    """Verify that the satellite discovery subsystem is initialized and reachable."""
    return service.get_status()


@router.get(
    "/auth/verify",
    summary="Verify CDSE authentication status and token acquisition",
    description="Validates CDSE credentials and token acquisition without exposing sensitive credentials.",
)
def verify_cdse_authentication(
    downloader: ProductDownloaderService = Depends(get_product_downloader),
) -> dict:
    """Safely verify CDSE OAuth authentication and token acquisition."""
    return downloader.verify_authentication()


# ============================================================================
# Sentinel-1 Discovery Endpoints
# ============================================================================

@router.get(
    "/sentinel-1",
    response_model=Sentinel1DiscoveryResponse,
    summary="Discover Sentinel-1 SAR candidate scenes (GET)",
    description=(
        "Searches for Sentinel-1 radar scenes around an event date. "
        "Defaults to Nepal Trishuli/Bhote Koshi river basin if no AOI is specified."
    ),
)
def get_sentinel_1_candidates(
    flood_date: str = Query(
        default="2024-08-26",
        description="Flood event date (YYYY-MM-DD)",
        examples=["2024-08-26"]
    ),
    days_before: int = Query(default=15, ge=1, le=60, description="Pre-event search window in days"),
    days_after: int = Query(default=7, ge=0, le=30, description="Post-event search window in days"),
    max_results: int = Query(default=20, ge=1, le=100, description="Maximum candidate scenes to return"),
    service: SatelliteDiscoveryService = Depends(get_satellite_service),
) -> Sentinel1DiscoveryResponse:
    """Discover candidate Sentinel-1 scenes using default or query-configured parameters."""
    return service.discover_sentinel_1(
        aoi=None,
        flood_date=flood_date,
        days_before=days_before,
        days_after=days_after,
        max_results=max_results,
    )


@router.post(
    "/sentinel-1",
    response_model=Sentinel1DiscoveryResponse,
    summary="Discover Sentinel-1 SAR candidate scenes (POST with custom AOI)",
    description="Searches for Sentinel-1 radar scenes with a custom GeoJSON Polygon or Bounding Box AOI.",
)
def post_sentinel_1_candidates(
    payload: SatelliteDiscoveryRequest,
    service: SatelliteDiscoveryService = Depends(get_satellite_service),
) -> Sentinel1DiscoveryResponse:
    """Discover candidate Sentinel-1 scenes for custom AOI and flood event window."""
    return service.discover_sentinel_1(
        aoi=payload.aoi,
        flood_date=payload.flood_date,
        days_before=payload.days_before,
        days_after=payload.days_after,
        max_results=payload.max_results,
    )


# ============================================================================
# Sentinel-2 Discovery Endpoints
# ============================================================================

@router.get(
    "/sentinel-2",
    response_model=Sentinel2DiscoveryResponse,
    summary="Discover Sentinel-2 optical candidate scenes (GET)",
    description=(
        "Searches for Sentinel-2 optical scenes around an event date and assesses cloud cover conditions. "
        "Defaults to Nepal Trishuli/Bhote Koshi river basin if no AOI is specified."
    ),
)
def get_sentinel_2_candidates(
    flood_date: str = Query(
        default="2024-08-26",
        description="Flood event date (YYYY-MM-DD)",
        examples=["2024-08-26"]
    ),
    days_before: int = Query(default=15, ge=1, le=60, description="Pre-event search window in days"),
    days_after: int = Query(default=7, ge=0, le=30, description="Post-event search window in days"),
    max_results: int = Query(default=20, ge=1, le=100, description="Maximum candidate scenes to return"),
    service: SatelliteDiscoveryService = Depends(get_satellite_service),
) -> Sentinel2DiscoveryResponse:
    """Discover candidate Sentinel-2 scenes and evaluate cloud cover suitability."""
    return service.discover_sentinel_2(
        aoi=None,
        flood_date=flood_date,
        days_before=days_before,
        days_after=days_after,
        max_results=max_results,
    )


@router.post(
    "/sentinel-2",
    response_model=Sentinel2DiscoveryResponse,
    summary="Discover Sentinel-2 optical candidate scenes (POST with custom AOI)",
    description="Searches for Sentinel-2 optical scenes with a custom GeoJSON Polygon or Bounding Box AOI.",
)
def post_sentinel_2_candidates(
    payload: SatelliteDiscoveryRequest,
    service: SatelliteDiscoveryService = Depends(get_satellite_service),
) -> Sentinel2DiscoveryResponse:
    """Discover candidate Sentinel-2 scenes for custom AOI and flood event window."""
    return service.discover_sentinel_2(
        aoi=payload.aoi,
        flood_date=payload.flood_date,
        days_before=payload.days_before,
        days_after=payload.days_after,
        max_results=payload.max_results,
    )


# ============================================================================
# Smart Satellite Router Endpoints
# ============================================================================

@router.get(
    "/route",
    response_model=RoutingDecisionResponse,
    summary="Compute recommended satellite strategy (GET)",
    description=(
        "Runs catalog discovery on Sentinel-1 and Sentinel-2, then invokes the Smart Satellite Router "
        "to recommend an optimal data strategy based on radar geometry and cloud cover."
    ),
)
def get_routing_decision(
    flood_date: str = Query(
        default="2024-08-26",
        description="Flood event date (YYYY-MM-DD)",
        examples=["2024-08-26"]
    ),
    days_before: int = Query(default=15, ge=1, le=60, description="Pre-event search window in days"),
    days_after: int = Query(default=7, ge=0, le=30, description="Post-event search window in days"),
    max_results: int = Query(default=20, ge=1, le=100, description="Maximum candidate scenes per constellation"),
    service: SatelliteDiscoveryService = Depends(get_satellite_service),
) -> RoutingDecisionResponse:
    """Run satellite discovery and compute routing decision with default Nepal parameters."""
    return service.route_analysis(
        aoi=None,
        flood_date=flood_date,
        days_before=days_before,
        days_after=days_after,
        max_results=max_results,
    )


@router.post(
    "/route",
    response_model=RoutingDecisionResponse,
    summary="Compute recommended satellite strategy (POST with custom AOI)",
    description=(
        "Runs catalog discovery for custom AOI and invokes the Smart Satellite Router "
        "to determine whether to use Sentinel-1, Sentinel-2, both, or report no suitable data."
    ),
)
def post_routing_decision(
    payload: SatelliteDiscoveryRequest,
    service: SatelliteDiscoveryService = Depends(get_satellite_service),
) -> RoutingDecisionResponse:
    """Run satellite discovery and compute routing decision for custom AOI."""
    return service.route_analysis(
        aoi=payload.aoi,
        flood_date=payload.flood_date,
        days_before=payload.days_before,
        days_after=payload.days_after,
        max_results=payload.max_results,
    )


# ============================================================================
# Scene Selection Endpoints (Module 3)
# ============================================================================

@router.get(
    "/select",
    response_model=SceneSelectionResponse,
    summary="Select optimal before/after satellite scenes (GET)",
    description=(
        "Executes discovery, routes optimal satellite data strategy, and selects the best "
        "before and after satellite scenes with compatible viewing geometry (SAR relative orbit) "
        "and quality criteria. Defaults to Nepal Trishuli/Bhote Koshi basin."
    ),
)
def get_scene_selection(
    flood_date: str = Query(
        default="2026-08-26",
        description="Flood event date (YYYY-MM-DD)",
        examples=["2026-08-26"]
    ),
    days_before: int = Query(default=15, ge=1, le=60, description="Pre-event search window in days"),
    days_after: int = Query(default=7, ge=0, le=30, description="Post-event search window in days"),
    max_results: int = Query(default=20, ge=1, le=100, description="Maximum candidate scenes per constellation"),
    selector: SceneSelectionService = Depends(get_scene_selection_service),
) -> SceneSelectionResponse:
    """Select before and after satellite scene pairs using default Nepal parameters."""
    return selector.select_scenes(
        aoi=None,
        flood_date=flood_date,
        days_before=days_before,
        days_after=days_after,
        max_results=max_results,
    )


@router.post(
    "/select",
    response_model=SceneSelectionResponse,
    summary="Select optimal before/after satellite scenes (POST with custom AOI)",
    description="Selects optimal before and after satellite scenes for custom AOI polygon or bounding box.",
)
def post_scene_selection(
    payload: SatelliteDiscoveryRequest,
    selector: SceneSelectionService = Depends(get_scene_selection_service),
) -> SceneSelectionResponse:
    """Select before and after satellite scene pairs for custom AOI."""
    return selector.select_scenes(
        aoi=payload.aoi,
        flood_date=payload.flood_date,
        days_before=payload.days_before,
        days_after=payload.days_after,
        max_results=payload.max_results,
    )


# ============================================================================
# Cache Inspection & Maintenance Endpoints
# ============================================================================

@router.get(
    "/cache/stats",
    summary="Get satellite discovery cache statistics",
    description="Returns the number of cached catalog queries and total disk space used.",
)
def get_cache_stats(
    service: SatelliteDiscoveryService = Depends(get_satellite_service),
) -> dict:
    """Return discovery cache usage statistics."""
    return service.cache.get_stats()


@router.post(
    "/cache/clear",
    summary="Clear satellite discovery cache",
    description="Clears all cached catalog discovery queries stored in data/cache.",
)
def clear_cache(
    service: SatelliteDiscoveryService = Depends(get_satellite_service),
) -> dict:
    """Purge discovery cache files from disk."""
    deleted_count = service.cache.clear()
    return {"status": "cleared", "deleted_files_count": deleted_count}


# ============================================================================
# Product Download & Raw Storage Endpoints (Module 3 Milestone 2)
# ============================================================================

@router.post(
    "/download",
    response_model=DownloadSummaryResponse,
    summary="Download satellite products from Copernicus CDSE",
    description=(
        "Retrieves raw satellite products for selected scene pairs or explicit product IDs. "
        "Checks local raw disk cache first. If Copernicus credentials are not configured, "
        "returns authentication_required status with setup instructions."
    ),
)
def download_satellite_products(
    payload: ProductDownloadRequest,
    downloader: ProductDownloaderService = Depends(get_product_downloader),
) -> DownloadSummaryResponse:
    """Download one or more satellite products or scene pairs."""
    return downloader.execute_download_request(payload)


@router.get(
    "/downloads",
    response_model=StoredDownloadsResponse,
    summary="List locally stored satellite product downloads",
    description="Returns a listing of all raw satellite products currently cached in data/raw.",
)
def list_downloaded_products(
    downloader: ProductDownloaderService = Depends(get_product_downloader),
) -> StoredDownloadsResponse:
    """List raw satellite archives and metadata files stored locally."""
    return downloader.list_downloaded_products()


@router.get(
    "/downloads/{product_id}",
    response_model=ProductDownloadResult,
    summary="Check status of specific satellite product download",
    description="Returns download status, disk path, checksum, and file size for a specific product ID.",
)
def get_product_download_status(
    product_id: str,
    collection: Optional[str] = Query(None, description="Optional collection filter: SENTINEL-1 or SENTINEL-2"),
    downloader: ProductDownloaderService = Depends(get_product_downloader),
) -> ProductDownloadResult:
    """Check download status of an individual product."""
    return downloader.download_product(product_id=product_id, collection=collection)


# ============================================================================
# AOI Clipping & Spatial Preparation Endpoints (Module 3 Milestone 3)
# ============================================================================

@router.get(
    "/clip/specification",
    response_model=ClipSpecification,
    summary="Compute AOI clipping spatial specification (GET)",
    description=(
        "Calculates the exact raster grid bounds, matrix dimensions (height x width), "
        "Affine transformation matrix, and resolution for a given AOI or default Nepal basin."
    ),
)
def get_clip_specification(
    min_lon: Optional[float] = Query(None, description="Bounding box min longitude"),
    min_lat: Optional[float] = Query(None, description="Bounding box min latitude"),
    max_lon: Optional[float] = Query(None, description="Bounding box max longitude"),
    max_lat: Optional[float] = Query(None, description="Bounding box max latitude"),
    resolution_meters: float = Query(default=10.0, ge=1.0, le=100.0, description="Spatial resolution in meters"),
    clipper: SpatialClippingService = Depends(get_spatial_clipper),
    sat_service: SatelliteDiscoveryService = Depends(get_satellite_service),
) -> ClipSpecification:
    """Compute clipping specification for bounding box or default Nepal AOI."""
    if min_lon is not None and min_lat is not None and max_lon is not None and max_lat is not None:
        aoi = {
            "type": "Polygon",
            "coordinates": [
                [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]
            ],
        }
    else:
        aoi = sat_service.get_default_nepal_aoi()

    return clipper.calculate_clip_specification(aoi, resolution_meters=resolution_meters)


@router.post(
    "/clip",
    response_model=PairClippingResponse,
    summary="Clip or prepare satellite scene pair to AOI (POST)",
    description=(
        "Cuts before and after satellite scenes to the user's AOI polygon or bounding box, "
        "ensuring strict pixel-for-pixel co-registration and identical grid dimensions for change detection."
    ),
)
def post_clip_pair(
    payload: PairClippingRequest,
    clipper: SpatialClippingService = Depends(get_spatial_clipper),
    sat_service: SatelliteDiscoveryService = Depends(get_satellite_service),
    selector: SceneSelectionService = Depends(get_scene_selection_service),
) -> PairClippingResponse:
    """Clip before and after satellite scenes to the target AOI."""
    aoi = payload.aoi or sat_service.get_default_nepal_aoi()

    pair = payload.scene_pair
    if not pair:
        selection_resp = selector.select_scenes(aoi=aoi, flood_date="2024-08-26")
        pair = selection_resp.sentinel_1_selection or selection_resp.sentinel_2_selection

    if not pair:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="No valid satellite scene pair provided or found for the specified AOI.",
        )

    return clipper.prepare_pair_clip(
        scene_pair=pair,
        aoi=aoi,
        job_id=payload.job_id,
        resolution_meters=payload.resolution_meters,
    )


# ============================================================================
# SAR Radiometric Calibration & Preprocessing Endpoints (Module 3 Milestone 4)
# ============================================================================

@router.post(
    "/preprocess/sentinel-1",
    response_model=Sentinel1PreprocessResponse,
    summary="Radiometrically calibrate and filter Sentinel-1 SAR pair (POST)",
    description=(
        "Converts dual-polarization (VV/VH) radar scenes to decibel (dB) backscatter values, "
        "applies spatial speckle noise reduction filtering, and calculates temporal difference channels "
        "optimized for AI flood and water extent segmentation."
    ),
)
def post_preprocess_sentinel_1(
    payload: Sentinel1PreprocessRequest,
    preprocessor: Sentinel1PreprocessorService = Depends(get_sar_preprocessor),
    clipper: SpatialClippingService = Depends(get_spatial_clipper),
    sat_service: SatelliteDiscoveryService = Depends(get_satellite_service),
    selector: SceneSelectionService = Depends(get_scene_selection_service),
) -> Sentinel1PreprocessResponse:
    """Preprocess Sentinel-1 SAR before/after pair."""
    if not payload.clipped_pair and not payload.scene_pair:
        aoi = payload.aoi or sat_service.get_default_nepal_aoi()
        selection_resp = selector.select_scenes(aoi=aoi, flood_date="2024-08-26")
        s1_pair = selection_resp.sentinel_1_selection
        if not s1_pair:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail="No compatible Sentinel-1 scene pair available for preprocessing.",
            )
        clipped_res = clipper.prepare_pair_clip(s1_pair, aoi=aoi, job_id=payload.job_id)
        payload.clipped_pair = clipped_res
        payload.scene_pair = s1_pair

    return preprocessor.preprocess_sentinel1_pair(payload)


# ============================================================================
# Optical NDWI & Spectral Index Preprocessing Endpoints (Module 3 Milestone 5)
# ============================================================================

@router.post(
    "/preprocess/sentinel-2",
    response_model=Sentinel2PreprocessResponse,
    summary="Compute Sentinel-2 NDWI / MNDWI optical water indices (POST)",
    description=(
        "Processes Sentinel-2 optical imagery into Normalized Difference Water Index (NDWI) "
        "and Modified NDWI (MNDWI) bands. Enforces cloud screening: automatically skips optical processing "
        "if cloud cover exceeds the threshold (e.g. monsoon storms > 50%), preserving system reliability."
    ),
)
def post_preprocess_sentinel_2(
    payload: Sentinel2PreprocessRequest,
    optical_service: Sentinel2PreprocessorService = Depends(get_optical_preprocessor),
    sat_service: SatelliteDiscoveryService = Depends(get_satellite_service),
    selector: SceneSelectionService = Depends(get_scene_selection_service),
) -> Sentinel2PreprocessResponse:
    """Preprocess Sentinel-2 optical pair or report cloud-screened status."""
    if not payload.clipped_pair and not payload.scene_pair:
        aoi = payload.aoi or sat_service.get_default_nepal_aoi()
        selection_resp = selector.select_scenes(aoi=aoi, flood_date="2024-08-26")
        s2_pair = selection_resp.sentinel_2_selection
        if not s2_pair:
            # If no optical pair was found or router skipped optical
            return Sentinel2PreprocessResponse(
                product_id_pre=None,
                product_id_post=None,
                status="skipped_cloudy",
                is_optical_useful=False,
                reason="No viable Sentinel-2 optical scenes with acceptable cloud cover found around event date.",
                channels=[],
                water_detection_summary={"optical_skipped": True},
                grid_shape=[3340, 2951],
                crs="EPSG:4326",
                output_dir=None,
                message="Optical preprocessing skipped: heavy monsoon cloud cover precludes reliable optical water extraction.",
            )
        payload.scene_pair = s2_pair

    return optical_service.preprocess_sentinel2_pair(payload)


# ============================================================================
# Analysis-Ready Data (ARD) Endpoints (Module 3 Milestone 6)
# ============================================================================

@router.post(
    "/prepare-ard",
    response_model=AnalysisReadyPackage,
    summary="Generate full Analysis-Ready Data (ARD) package (POST)",
    description=(
        "Executes the complete Module 3 pipeline: discovers scenes, selects optimal compatible pairs, "
        "clips to AOI boundary, radiometrically calibrates Sentinel-1 dual-pol channels, processes optical NDWI/MNDWI "
        "if usable, and compiles a co-registered multi-channel package ready for AI segmentation."
    ),
)
def post_prepare_ard(
    payload: PrepareArdRequest,
    ard_service: AnalysisReadyDataService = Depends(get_ard_service),
) -> AnalysisReadyPackage:
    """Execute end-to-end Module 3 ARD pipeline."""
    return ard_service.prepare_ard(payload)


@router.get(
    "/ard/{job_id}",
    response_model=AnalysisReadyPackage,
    summary="Get Analysis-Ready Data package by Job ID (GET)",
    description="Retrieves the generated ARD package, spatial reference, and channels for a specific analysis job.",
)
def get_ard_package(
    job_id: str,
    ard_service: AnalysisReadyDataService = Depends(get_ard_service),
) -> AnalysisReadyPackage:
    """Retrieve existing ARD package by job ID."""
    package = ard_service.get_ard_by_job_id(job_id)
    if not package:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"Analysis-Ready Data package for job {job_id} not found.",
        )
    return package


