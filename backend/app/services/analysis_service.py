import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.models.analysis import AnalysisRecord, AnalysisStatus
from app.schemas.analysis import AnalysisCreateRequest


# ============================================================================
# Future Module Placeholder Interfaces
# (Intentionally not implemented in Module 1; ready for Modules 2 to 8)
# ============================================================================

def satellite_router(aoi: Dict[str, Any], flood_date: str) -> Dict[str, Any]:
    """
    Module 2: Smart Satellite Router.
    Routes to Sentinel-1 (SAR) and/or Sentinel-2 (Optical) based on weather,
    cloud coverage forecasts, and event timing.
    """
    from app.services.satellite_service import get_satellite_service
    service = get_satellite_service()
    decision = service.route_analysis(aoi=aoi, flood_date=flood_date)
    return decision.model_dump()


def image_selector(aoi: Dict[str, Any], flood_date: str, satellite_meta: Optional[Any] = None) -> Dict[str, Any]:
    """
    Module 3: Before/After Image Selection.
    Selects optimal pre-event baseline and post-event disaster satellite acquisitions
    with compatible geometry and quality metrics.
    """
    from app.services.scene_selector import get_scene_selection_service
    selector = get_scene_selection_service()
    selection = selector.select_scenes(aoi=aoi, flood_date=flood_date)
    return selection.model_dump()


def download_images(selection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module 3: Satellite Product Access & Download.
    Retrieves selected satellite products or reports authentication requirement.
    """
    from app.services.product_downloader import get_product_downloader
    from app.schemas.download import ProductDownloadRequest
    from app.schemas.scene_selection import SelectedScenePair
    downloader = get_product_downloader()
    req = ProductDownloadRequest()
    if selection.get("sentinel_1_selection"):
        req.scene_pair = SelectedScenePair(**selection["sentinel_1_selection"])
    elif selection.get("sentinel_2_selection"):
        req.scene_pair = SelectedScenePair(**selection["sentinel_2_selection"])
    summary = downloader.execute_download_request(req)
    return summary.model_dump()


def clip_images(selection: Dict[str, Any], aoi: Dict[str, Any], job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Module 3: AOI Clipping & Spatial Preparation.
    Subsets satellite scene pair to exact AOI bounds with matching spatial grid.
    """
    from app.services.spatial_clipper import get_spatial_clipper
    from app.schemas.scene_selection import SelectedScenePair
    clipper = get_spatial_clipper()
    pair = None
    if selection.get("sentinel_1_selection"):
        pair = SelectedScenePair(**selection["sentinel_1_selection"])
    elif selection.get("sentinel_2_selection"):
        pair = SelectedScenePair(**selection["sentinel_2_selection"])

    if not pair:
        raise ValueError("No valid scene pair found in selection to clip.")

    res = clipper.prepare_pair_clip(pair, aoi=aoi, job_id=job_id)
    return res.model_dump()


def preprocessing(clipped_images: Any = None, job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Module 3 / Module 4: Preprocessing & Radiometric Calibration.
    Applies dB calibration, speckle filtering, and temporal differencing on SAR channels.
    """
    if clipped_images is None:
        raise NotImplementedError("preprocessing will be executed as part of the full pipeline.")

    from app.services.sar_preprocessor import get_sar_preprocessor
    from app.schemas.clipping import PairClippingResponse
    from app.schemas.preprocessing import Sentinel1PreprocessRequest
    preprocessor = get_sar_preprocessor()
    clipped_pair = PairClippingResponse(**clipped_images)
    req = Sentinel1PreprocessRequest(clipped_pair=clipped_pair, job_id=job_id)
    res = preprocessor.preprocess_sentinel1_pair(req)
    return res.model_dump()


def segmentation(preprocessed_data: Any) -> Dict[str, Any]:
    """
    Placeholder for Module 5: AI Water & Debris Segmentation.
    Runs deep learning models on satellite imagery to segment flood extent and standing water.
    """
    raise NotImplementedError("segmentation will be implemented in Module 5.")


def geospatial_analysis(flood_mask: Any, aoi: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module 5: Geospatial Infrastructure Damage & Exposure Overlay.
    Extracts OpenStreetMap buildings, roads, and bridges intersecting flood extents.
    """
    if flood_mask is None:
        raise NotImplementedError("geospatial_analysis will be executed as part of the full pipeline.")

    from app.services.geospatial_service import get_geospatial_service
    from app.schemas.geospatial import GeospatialAnalysisRequest
    service = get_geospatial_service()
    job_id = flood_mask.get("job_id") if isinstance(flood_mask, dict) else str(flood_mask)
    req = GeospatialAnalysisRequest(job_id=job_id)
    res = service.analyze_job_impact(req)
    return res.model_dump()


def connectivity_analysis(infrastructure_impact: Any) -> Dict[str, Any]:
    """
    Module 6: Settlement & Road Network Accessibility Analysis.
    Calculates network graph reachability, isolated communities, hospital access, and road detours.
    """
    if infrastructure_impact is None:
        raise NotImplementedError("connectivity_analysis requires valid infrastructure impact input.")

    from app.services.connectivity_service import get_connectivity_service
    from app.schemas.connectivity import ConnectivityAnalysisRequest
    service = get_connectivity_service()
    job_id = infrastructure_impact.get("job_id") if isinstance(infrastructure_impact, dict) else str(infrastructure_impact)
    req = ConnectivityAnalysisRequest(job_id=job_id)
    res = service.analyze_connectivity(req)
    return res.model_dump()



def report_generator(analysis_data: Any, language: str = "en") -> Dict[str, Any]:
    """
    Module 7: Multilingual Situation Report Generator.
    Synthesizes structured findings into humanitarian disaster briefs in the requested language.
    """
    if analysis_data is None:
        raise NotImplementedError("report_generator requires valid analysis data.")

    from app.services.report_service import get_report_service
    from app.schemas.results import ReportGenerateRequest
    service = get_report_service()
    job_id = analysis_data.get("job_id") if isinstance(analysis_data, dict) else str(analysis_data)
    req = ReportGenerateRequest(language=language)
    res = service.generate_report(job_id, req)
    return res.model_dump()



# ============================================================================
# Analysis Job Storage & Service
# ============================================================================

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AnalysisRepository:
    """Repository for storing and managing analysis jobs with filesystem persistence."""

    def __init__(self, outputs_dir: Optional[Path] = None) -> None:
        self._jobs: Dict[str, AnalysisRecord] = {}
        try:
            from app.core.config import get_settings
            settings = get_settings()
            self._outputs_dir = outputs_dir or settings.get_data_path(settings.DATA_OUTPUTS_DIR)
        except Exception:
            self._outputs_dir = outputs_dir or Path("backend/data/outputs")
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self._outputs_dir or not self._outputs_dir.exists():
            return
        for child in self._outputs_dir.iterdir():
            if child.is_dir():
                job_id = child.name
                if job_id in self._jobs:
                    continue
                record_file = child / "job_record.json"
                if record_file.exists():
                    try:
                        with open(record_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            self._jobs[job_id] = AnalysisRecord.model_validate(data)
                        continue
                    except Exception as e:
                        logger.warning("Could not load job record for %s: %s", job_id, e)

                # Fallback: check if analysis_result.json exists to reconstruct record
                res_file = child / "analysis_result.json"
                if res_file.exists():
                    try:
                        with open(res_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        meta = data.get("metadata", {})
                        aoi = meta.get("aoi", {})
                        if not aoi or not aoi.get("coordinates"):
                            from app.services.satellite_service import DEFAULT_NEPAL_AOI
                            aoi = DEFAULT_NEPAL_AOI

                        created_str = data.get("created_at") or meta.get("processing_timestamp")
                        try:
                            created_at = datetime.fromisoformat(created_str) if created_str else datetime.now(timezone.utc)
                        except Exception:
                            created_at = datetime.now(timezone.utc)

                        st_val = data.get("overall_status", "completed")
                        status_enum = (
                            AnalysisStatus.COMPLETED
                            if "completed" in st_val
                            else (AnalysisStatus.FAILED if st_val == "failed" else AnalysisStatus.PROCESSING)
                        )

                        rec = AnalysisRecord(
                            job_id=job_id,
                            status=status_enum,
                            aoi=aoi,
                            flood_date=meta.get("flood_date", "2026-08-26"),
                            report_language=meta.get("report_language", "en"),
                            created_at=created_at,
                            updated_at=created_at,
                        )
                        self._jobs[job_id] = rec
                    except Exception as e:
                        logger.warning("Could not reconstruct record for %s: %s", job_id, e)

    def save(self, record: AnalysisRecord) -> AnalysisRecord:
        record.updated_at = datetime.now(timezone.utc)
        self._jobs[record.job_id] = record
        if self._outputs_dir:
            try:
                job_dir = self._outputs_dir / record.job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                with open(job_dir / "job_record.json", "w", encoding="utf-8") as f:
                    f.write(record.model_dump_json(indent=2))
            except Exception as e:
                logger.warning("Could not persist job record for %s: %s", record.job_id, e)
        return record

    def get_by_id(self, job_id: str) -> Optional[AnalysisRecord]:
        rec = self._jobs.get(job_id)
        if rec:
            return rec
        # Attempt to load from disk if not yet in memory
        if self._outputs_dir:
            job_dir = self._outputs_dir / job_id
            rec_file = job_dir / "job_record.json"
            if rec_file.exists():
                try:
                    with open(rec_file, "r", encoding="utf-8") as f:
                        record = AnalysisRecord.model_validate(json.load(f))
                        self._jobs[job_id] = record
                        return record
                except Exception:
                    pass
            res_file = job_dir / "analysis_result.json"
            if res_file.exists():
                try:
                    with open(res_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    meta = data.get("metadata", {})
                    aoi = meta.get("aoi", {})
                    if not aoi or not aoi.get("coordinates"):
                        from app.services.satellite_service import DEFAULT_NEPAL_AOI
                        aoi = DEFAULT_NEPAL_AOI
                    rec = AnalysisRecord(
                        job_id=job_id,
                        status=AnalysisStatus.COMPLETED,
                        aoi=aoi,
                        flood_date=meta.get("flood_date", "2026-08-26"),
                        report_language=meta.get("report_language", "en"),
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                    self._jobs[job_id] = rec
                    return rec
                except Exception:
                    pass
        return None

    def list_all(self) -> List[AnalysisRecord]:
        self._load_from_disk()
        return list(self._jobs.values())

    def update_status(
        self,
        job_id: str,
        status: AnalysisStatus,
        error_message: Optional[str] = None,
        results: Optional[Dict[str, Any]] = None,
    ) -> Optional[AnalysisRecord]:
        record = self.get_by_id(job_id)
        if not record:
            return None
        record.status = status
        record.updated_at = datetime.now(timezone.utc)
        if error_message is not None:
            record.error_message = error_message
        if results is not None:
            record.results = results
        self.save(record)
        return record

    def clear(self) -> None:
        """Utility for test suite reset."""
        self._jobs.clear()


class AnalysisService:
    """Business logic service for managing flood damage analysis jobs."""

    def __init__(self, repository: Optional[AnalysisRepository] = None) -> None:
        self.repository = repository or AnalysisRepository()

    def create_analysis(self, payload: AnalysisCreateRequest) -> AnalysisRecord:
        """
        Create a new analysis record from validated user input.
        Generates a unique job_id and sets initial status to 'created'.
        """
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        record = AnalysisRecord(
            job_id=job_id,
            status=AnalysisStatus.CREATED,
            aoi=payload.aoi,
            flood_date=payload.flood_date,
            report_language=payload.report_language,
            created_at=now,
            updated_at=now,
        )

        return self.repository.save(record)

    def get_analysis(self, job_id: str) -> Optional[AnalysisRecord]:
        """Retrieve an analysis job by its ID."""
        return self.repository.get_by_id(job_id)

    def execute_pipeline(self, job_id: str) -> None:
        """
        Placeholder pipeline orchestration.
        Future modules (2 through 8) will be chained here.
        """
        record = self.get_analysis(job_id)
        if not record:
            raise ValueError(f"Job {job_id} not found")

        # Future pipeline sequence:
        # 1. satellite_meta = satellite_router(record.aoi, record.flood_date)
        # 2. images = image_selector(record.aoi, record.flood_date, satellite_meta)
        # 3. preprocessed = preprocessing(images)
        # 4. flood_mask = segmentation(preprocessed)
        # 5. infrastructure = geospatial_analysis(flood_mask, record.aoi)
        # 6. connectivity = connectivity_analysis(infrastructure)
        # 7. report = report_generator(connectivity, record.report_language)
        raise NotImplementedError(
            "Analysis pipeline execution is intentionally reserved for subsequent modules."
        )


# Global singleton instance for app dependency injection
_analysis_service_instance = AnalysisService()


def get_analysis_service() -> AnalysisService:
    """Dependency provider for FastAPI route handlers."""
    return _analysis_service_instance
