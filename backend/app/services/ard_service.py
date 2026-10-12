import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import Settings, get_settings
from app.schemas.analysis import normalize_and_validate_aoi
from app.schemas.ard import (
    AnalysisReadyPackage,
    PrepareArdRequest,
    SpatialReferenceMetadata,
)
from app.schemas.optical_preprocessing import Sentinel2PreprocessRequest
from app.schemas.preprocessing import Sentinel1PreprocessRequest
from app.services.optical_preprocessor import (
    Sentinel2PreprocessorService,
    get_optical_preprocessor,
)
from app.services.product_downloader import (
    ProductDownloaderService,
    get_product_downloader,
)
from app.services.sar_preprocessor import (
    Sentinel1PreprocessorService,
    get_sar_preprocessor,
)
from app.services.satellite_service import (
    DEFAULT_NEPAL_AOI,
    SatelliteDiscoveryService,
    get_satellite_service,
)
from app.services.scene_selector import (
    SceneSelectionService,
    get_scene_selection_service,
)
from app.services.spatial_clipper import (
    SpatialClippingService,
    get_spatial_clipper,
)

logger = logging.getLogger(__name__)


class AnalysisReadyDataService:
    """
    Orchestration service for Module 3:
    Coordinates scene selection, access, spatial clipping, radiometric calibration,
    optical index calculation, and manifest generation to deliver a validated
    Analysis-Ready Data (ARD) package for Module 4 AI segmentation.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        sat_service: Optional[SatelliteDiscoveryService] = None,
        selector: Optional[SceneSelectionService] = None,
        downloader: Optional[ProductDownloaderService] = None,
        clipper: Optional[SpatialClippingService] = None,
        sar_preprocessor: Optional[Sentinel1PreprocessorService] = None,
        optical_preprocessor: Optional[Sentinel2PreprocessorService] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.sat_service = sat_service or get_satellite_service()
        self.selector = selector or get_scene_selection_service()
        self.downloader = downloader or get_product_downloader()
        self.clipper = clipper or get_spatial_clipper()
        self.sar_preprocessor = sar_preprocessor or get_sar_preprocessor()
        self.optical_preprocessor = optical_preprocessor or get_optical_preprocessor()
        self._ard_store: Dict[str, AnalysisReadyPackage] = {}

    def prepare_ard(self, request: PrepareArdRequest) -> AnalysisReadyPackage:
        """
        Execute the full end-to-end Module 3 processing chain and generate ARD package.
        """
        job_id = request.job_id or f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # 1. Resolve and normalize AOI
        if request.aoi is None:
            resolved_aoi = DEFAULT_NEPAL_AOI
        else:
            resolved_aoi = normalize_and_validate_aoi(request.aoi)

        # 2. Select optimal satellite scene pairs
        selection_resp = self.selector.select_scenes(
            aoi=resolved_aoi,
            flood_date=request.flood_date,
            days_before=request.pre_event_window_days,
            days_after=request.post_event_window_days,
        )

        s1_pair = selection_resp.sentinel_1_selection
        s2_pair = selection_resp.sentinel_2_selection
        strategy = selection_resp.selected_strategy

        # 3. Derive spatial clipping specification & co-registration
        clip_spec = self.clipper.calculate_clip_specification(
            aoi=resolved_aoi,
            resolution_meters=request.resolution_meters,
        )

        spatial_ref = SpatialReferenceMetadata(
            crs=clip_spec.target_crs,
            grid_shape=clip_spec.dimensions,
            affine_transform=clip_spec.affine_transform,
            bounding_box=clip_spec.bounding_box,
            area_sqkm=clip_spec.aoi_area_sqkm,
        )

        # 4. Process Sentinel-1 SAR if available
        sar_ard_resp = None
        channels: List[str] = []

        if s1_pair:
            clipped_s1 = self.clipper.prepare_pair_clip(
                scene_pair=s1_pair,
                aoi=resolved_aoi,
                job_id=job_id,
                resolution_meters=request.resolution_meters,
            )
            s1_req = Sentinel1PreprocessRequest(
                scene_pair=s1_pair,
                clipped_pair=clipped_s1,
                aoi=resolved_aoi,
                job_id=job_id,
                speckle_filter_window=request.speckle_filter_window,
                calibrate_to_db=True,
            )
            sar_ard_resp = self.sar_preprocessor.preprocess_sentinel1_pair(s1_req)
            for c in sar_ard_resp.channels:
                channels.append(c.band_name)

        # 5. Process Sentinel-2 Optical if available
        optical_ard_resp = None
        if s2_pair:
            clipped_s2 = self.clipper.prepare_pair_clip(
                scene_pair=s2_pair,
                aoi=resolved_aoi,
                job_id=job_id,
                resolution_meters=request.resolution_meters,
            )
            s2_req = Sentinel2PreprocessRequest(
                scene_pair=s2_pair,
                clipped_pair=clipped_s2,
                aoi=resolved_aoi,
                job_id=job_id,
                cloud_threshold_pct=request.cloud_threshold_pct,
            )
            optical_ard_resp = self.optical_preprocessor.preprocess_sentinel2_pair(s2_req)
            if optical_ard_resp.is_optical_useful:
                for c in optical_ard_resp.channels:
                    channels.append(c.band_name)

        # 6. Determine recommended segmentation AI model
        if sar_ard_resp and optical_ard_resp and optical_ard_resp.is_optical_useful:
            recommended_model = "multimodal_flood_debris"
        elif sar_ard_resp:
            recommended_model = "sar_unet_flood"
        elif optical_ard_resp and optical_ard_resp.is_optical_useful:
            recommended_model = "optical_water_segmenter"
        else:
            recommended_model = "heuristic_threshold_fallback"

        # 7. Serialize manifest JSON to disk
        out_dir = self.settings.get_data_path(f"{self.settings.DATA_PROCESSED_DIR}/{job_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "ard_manifest.json"

        package = AnalysisReadyPackage(
            job_id=job_id,
            flood_date=request.flood_date,
            aoi=resolved_aoi,
            status="ready" if (sar_ard_resp and sar_ard_resp.status == "calibrated") else "contract_ready",
            strategy_used=strategy,
            spatial_reference=spatial_ref,
            sar_ard=sar_ard_resp,
            optical_ard=optical_ard_resp,
            channels_available=channels,
            recommended_segmentation_model=recommended_model,
            manifest_file=str(manifest_path.resolve()),
            created_at=now,
            message=(
                f"Analysis-Ready Data package generated for job {job_id}. "
                f"Aligned on {spatial_ref.grid_shape[0]}x{spatial_ref.grid_shape[1]} grid ({spatial_ref.area_sqkm} sq km). "
                f"{len(channels)} channels prepared for {recommended_model} AI segmentation."
            ),
        )

        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(package.model_dump(mode="json"), f, indent=2)
        except Exception as e:
            logger.warning("Could not serialize manifest file: %s", e)

        # Store in memory
        self._ard_store[job_id] = package
        return package

    def get_ard_by_job_id(self, job_id: str) -> Optional[AnalysisReadyPackage]:
        """Retrieve stored ARD package by job ID or read from disk manifest."""
        if job_id in self._ard_store:
            return self._ard_store[job_id]

        # Attempt to read from disk
        manifest_file = self.settings.get_data_path(
            f"{self.settings.DATA_PROCESSED_DIR}/{job_id}/ard_manifest.json"
        )
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    pkg = AnalysisReadyPackage(**data)
                    self._ard_store[job_id] = pkg
                    return pkg
            except Exception as e:
                logger.error("Failed to load manifest %s: %s", manifest_file, e)

        return None


_ard_service_instance: Optional[AnalysisReadyDataService] = None


def get_ard_service() -> AnalysisReadyDataService:
    """Return singleton instance of AnalysisReadyDataService."""
    global _ard_service_instance
    if _ard_service_instance is None:
        _ard_service_instance = AnalysisReadyDataService()
    return _ard_service_instance
