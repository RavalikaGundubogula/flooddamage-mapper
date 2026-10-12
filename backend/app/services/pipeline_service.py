import json
import logging
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import Settings, get_settings
from app.models.analysis import AnalysisRecord, AnalysisStatus
from app.schemas.analysis import AnalysisCreateRequest, normalize_and_validate_aoi
from app.schemas.ard import PrepareArdRequest
from app.schemas.connectivity import ConnectivityAnalysisRequest
from app.schemas.geospatial import GeospatialAnalysisRequest
from app.schemas.pipeline import (
    PipelineRunRequest,
    PipelineStage,
    PipelineStatusResponse,
    StageLogEntry,
)
from app.schemas.results import ReportGenerateRequest
from app.schemas.segmentation import AffectedAreaSummary, SegmentationRequest, SegmentationResponse
from app.services.analysis_service import AnalysisRepository, get_analysis_service
from app.services.ard_service import AnalysisReadyDataService, get_ard_service
from app.services.connectivity_service import ConnectivityAnalysisService, get_connectivity_service
from app.services.geospatial_service import GeospatialDamageAnalysisService, get_geospatial_service
from app.services.product_downloader import ProductDownloaderService, get_product_downloader
from app.services.report_service import SituationReportService, get_report_service
from app.services.results_service import ResultsAggregationService, get_results_service
from app.services.satellite_service import (
    DEFAULT_NEPAL_AOI,
    SatelliteDiscoveryService,
    get_satellite_service,
)
from app.services.scene_selector import SceneSelectionService, get_scene_selection_service
from app.services.segmentation_service import SegmentationInferenceService, get_segmentation_service

logger = logging.getLogger(__name__)


class PipelineExecutionService:
    """
    Central orchestration engine for the Flood Damage Mapper (Module 8).
    
    Coordinates the 13-step disaster response chain:
    1. Validation & Input Normalization
    2. Job Registration
    3. Sentinel-1 SAR Catalog Discovery
    4. Sentinel-2 Optical Catalog Discovery
    5. Smart Resource Router Decision Matrix
    6. Relative-Orbit Scene Selection
    7. CDSE Authentication Check & ARD Preparation
    8. AI Flood & Debris Segmentation (FloodUNet)
    9. Geospatial Infrastructure Overlay (OSM Buildings, Roads, Bridges)
    10. Road Blocking & Settlement/Hospital Reachability Graph
    11. Central Structured Results Aggregation
    12. Multilingual Situation Report Generation (English & Nepali)
    13. Dashboard Presentation & Layer Publishing
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        repo: Optional[AnalysisRepository] = None,
        satellite_service: Optional[SatelliteDiscoveryService] = None,
        scene_selector: Optional[SceneSelectionService] = None,
        downloader: Optional[ProductDownloaderService] = None,
        ard_service: Optional[AnalysisReadyDataService] = None,
        segmentation_service: Optional[SegmentationInferenceService] = None,
        geospatial_service: Optional[GeospatialDamageAnalysisService] = None,
        connectivity_service: Optional[ConnectivityAnalysisService] = None,
        results_service: Optional[ResultsAggregationService] = None,
        report_service: Optional[SituationReportService] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repo = repo or get_analysis_service().repository
        self.sat_service = satellite_service or get_satellite_service()
        self.selector = scene_selector or get_scene_selection_service()
        self.downloader = downloader or get_product_downloader()
        self.ard_service = ard_service or get_ard_service()
        self.seg_service = segmentation_service or get_segmentation_service()
        self.geo_service = geospatial_service or get_geospatial_service()
        self.conn_service = connectivity_service or get_connectivity_service()
        self.res_service = results_service or get_results_service()
        self.rep_service = report_service or get_report_service()

        self.outputs_dir = self.settings.get_data_path(self.settings.DATA_OUTPUTS_DIR)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self._pipeline_states: Dict[str, PipelineStatusResponse] = {}

    def _normalize_date_str(self, date_str: str) -> str:
        """Ensure date string is YYYY-MM-DD format."""
        date_str = date_str.strip()
        # Check DD-MM-YYYY
        m_dmy = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", date_str)
        if m_dmy:
            day, month, year = m_dmy.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
        # Check YYYY-MM-DD
        m_ymd = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", date_str)
        if m_ymd:
            year, month, day = m_ymd.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
        # Fallback default
        return "2026-08-26"

    def _update_stage(
        self,
        job_id: str,
        stage: PipelineStage,
        progress_percent: int,
        message: str,
        status_code: str = "processing",
        is_blocked: bool = False,
        blocker: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> PipelineStatusResponse:
        """Record stage transition in memory and append to history."""
        now = datetime.now(timezone.utc)
        entry = StageLogEntry(
            stage=stage,
            timestamp=now,
            message=message,
            progress_percent=progress_percent,
        )

        state = self._pipeline_states.get(job_id)
        if not state:
            state = PipelineStatusResponse(
                job_id=job_id,
                status=status_code,
                current_stage=stage,
                progress_percent=progress_percent,
                message=message,
                timestamp=now,
                is_blocked=is_blocked,
                blocker=blocker,
                error_message=error_message,
                stages_history=[entry],
            )
        else:
            state.status = status_code
            state.current_stage = stage
            state.progress_percent = progress_percent
            state.message = message
            state.timestamp = now
            if is_blocked:
                state.is_blocked = True
            if blocker:
                state.blocker = blocker
            if error_message:
                state.error_message = error_message
            state.stages_history.append(entry)

        self._pipeline_states[job_id] = state

        # Also update underlying analysis job record
        job_rec = self.repo.get_by_id(job_id)
        if job_rec:
            if status_code == "completed":
                job_rec.status = AnalysisStatus.COMPLETED
            elif status_code == "failed":
                job_rec.status = AnalysisStatus.FAILED
                job_rec.error_message = error_message
            else:
                job_rec.status = AnalysisStatus.PROCESSING
            job_rec.updated_at = now
            self.repo.save(job_rec)

        return state

    def get_status(self, job_id: str) -> Optional[PipelineStatusResponse]:
        """Query current pipeline status for a job."""
        if job_id in self._pipeline_states:
            state = self._pipeline_states[job_id]
            # Refresh available layers
            state.available_layers = self.list_available_layers(job_id)
            return state

        # If not in cache, check if job exists in repo or disk
        job_rec = self.repo.get_by_id(job_id)
        job_dir = self.outputs_dir / job_id
        if not job_rec and not job_dir.exists():
            return None

        # Check for structured result on disk to provide rich state
        structured_res = None
        res_file = job_dir / "analysis_result.json"
        if res_file.exists():
            try:
                from app.schemas.results import StructuredAnalysisResult
                with open(res_file, "r", encoding="utf-8") as f:
                    structured_res = StructuredAnalysisResult(**json.load(f))
            except Exception:
                pass

        if structured_res:
            is_blocked = (structured_res.source_data_status == "satellite_binary_data_blocked")
            status_str = "completed_with_limitations" if is_blocked else "completed"
            stage_enum = PipelineStage.COMPLETED_WITH_LIMITATIONS if is_blocked else PipelineStage.COMPLETED
            blocker = "CDSE binary satellite download credentials unavailable" if is_blocked else None
            bldgs = structured_res.geospatial.affected_buildings
            bldgs_cnt = (
                (bldgs.get("heavily_affected", 0) + bldgs.get("potentially_affected", 0))
                if isinstance(bldgs, dict)
                else bldgs
            )
            roads = structured_res.geospatial.affected_roads
            roads_cnt = (
                (roads.get("potentially_blocked", 0) + roads.get("heavily_affected", 0))
                if isinstance(roads, dict)
                else roads
            )
            roads_km = roads.get("affected_length_km") if isinstance(roads, dict) else None
            bridges = structured_res.geospatial.affected_bridges
            bridges_cnt = (
                bridges.get("heavily_affected", 0)
                if isinstance(bridges, dict)
                else bridges
            )
            summary = {
                "flood_area_km2": structured_res.segmentation.flood_area_km2,
                "affected_buildings": bldgs_cnt,
                "affected_buildings_details": bldgs,
                "affected_roads": roads_cnt,
                "affected_roads_km": roads_km,
                "affected_roads_details": roads,
                "affected_bridges": bridges_cnt,
                "affected_bridges_details": bridges,
                "blocked_roads": structured_res.connectivity.blocked_road_count,
                "isolated_settlements": structured_res.connectivity.flood_isolated_settlement_count,
                "hospital_access_lost": structured_res.connectivity.hospital_access_lost_count,
                "debris_zones": structured_res.geospatial.change_zone_count,
            }
            sat_status = getattr(structured_res.satellite.status, "value", str(structured_res.satellite.status))
            seg_status = getattr(structured_res.segmentation.status, "value", str(structured_res.segmentation.status))
            geo_status = getattr(structured_res.geospatial.status, "value", str(structured_res.geospatial.status))
            conn_status = getattr(structured_res.connectivity.status, "value", str(structured_res.connectivity.status))
        else:
            status_str = job_rec.status.value if job_rec else "completed"
            is_blocked = "blocked" in job_id.lower()
            stage_enum = PipelineStage.COMPLETED_WITH_LIMITATIONS if is_blocked else PipelineStage.COMPLETED
            blocker = "CDSE binary satellite download credentials unavailable" if is_blocked else None
            summary = None
            sat_status = None
            seg_status = None
            geo_status = None
            conn_status = None

        state = PipelineStatusResponse(
            job_id=job_id,
            status=status_str,
            current_stage=stage_enum if status_str in ["completed", "completed_with_limitations"] else PipelineStage.CREATED,
            progress_percent=100 if status_str == "completed" else 0,
            message=f"Analysis pipeline completed for job '{job_id}'." if status_str == "completed" else f"Pipeline status for job '{job_id}'",
            timestamp=datetime.now(timezone.utc),
            is_blocked=is_blocked,
            blocker=blocker,
            available_layers=self.list_available_layers(job_id),
            satellite_status=sat_status,
            segmentation_status=seg_status,
            geospatial_status=geo_status,
            connectivity_status=conn_status,
            summary=summary,
        )
        self._pipeline_states[job_id] = state
        return state

    def list_available_layers(self, job_id: str) -> List[str]:
        """Check which GeoJSON layers are generated and available on disk."""
        layers = []
        job_dir = self.outputs_dir / job_id
        if not job_dir.exists():
            return layers

        mapping = {
            "flood_regions.geojson": "flood",
            "buildings.geojson": "buildings",
            "roads.geojson": "roads",
            "bridges.geojson": "bridges",
            "change_zones.geojson": "change_zones",
            "blocked_roads.geojson": "blocked_roads",
            "partially_affected_roads.geojson": "partially_affected_roads",
            "settlements_connectivity.geojson": "settlements",
            "isolated_settlements.geojson": "isolated_settlements",
            "hospital_access_lost_settlements.geojson": "hospitals",
        }
        for fname, lname in mapping.items():
            if (job_dir / fname).exists():
                layers.append(lname)

        if (self.outputs_dir / f"{job_id}_segmentation.json").exists():
            if "flood" not in layers:
                layers.append("flood")

        # AOI layer is available if job record, ARD manifest, or result exists
        if self.repo.get_by_id(job_id) or (job_dir / "analysis_result.json").exists():
            layers.insert(0, "aoi")

        return layers

    def get_layer_geojson(self, job_id: str, layer_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve GeoJSON FeatureCollection for a specific layer."""
        job_dir = self.outputs_dir / job_id

        # Direct file mapping
        mapping = {
            "flood": "flood_regions.geojson",
            "buildings": "buildings.geojson",
            "roads": "roads.geojson",
            "bridges": "bridges.geojson",
            "change_zones": "change_zones.geojson",
            "blocked_roads": "blocked_roads.geojson",
            "partially_affected_roads": "partially_affected_roads.geojson",
            "settlements": "settlements_connectivity.geojson",
            "isolated_settlements": "isolated_settlements.geojson",
            "hospitals": "hospital_access_lost_settlements.geojson",
        }

        # AOI special handling
        if layer_name == "aoi":
            job = self.repo.get_by_id(job_id)
            aoi_poly = None
            if job and job.aoi and job.aoi.get("coordinates"):
                aoi_poly = job.aoi
            else:
                job_dir = self.outputs_dir / job_id
                ard_path = self.settings.get_data_path(self.settings.DATA_PROCESSED_DIR) / job_id / "ard_manifest.json"
                if ard_path.exists():
                    try:
                        with open(ard_path, "r", encoding="utf-8") as f:
                            ard_data = json.load(f)
                            if ard_data.get("aoi", {}).get("coordinates"):
                                aoi_poly = ard_data["aoi"]
                    except Exception:
                        pass
                if not aoi_poly:
                    res_path = job_dir / "analysis_result.json"
                    if res_path.exists():
                        try:
                            with open(res_path, "r", encoding="utf-8") as f:
                                res_data = json.load(f)
                                if res_data.get("metadata", {}).get("aoi", {}).get("coordinates"):
                                    aoi_poly = res_data["metadata"]["aoi"]
                        except Exception:
                            pass
            if not aoi_poly:
                aoi_poly = DEFAULT_NEPAL_AOI
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": aoi_poly,
                        "properties": {
                            "name": "Area of Interest (AOI)",
                            "job_id": job_id,
                        },
                    }
                ],
            }

        filename = mapping.get(layer_name)
        if not filename:
            return None

        file_path = job_dir / filename
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Could not read layer %s for job %s: %s", layer_name, job_id, e)

        # Fallback for flood from segmentation output
        if layer_name == "flood":
            seg_json = self.outputs_dir / f"{job_id}_segmentation.json"
            if seg_json.exists():
                try:
                    with open(seg_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return data.get("geojson")
                except Exception:
                    pass

        # Return empty FeatureCollection rather than 404 if layer expected
        return {"type": "FeatureCollection", "features": []}

    def execute_pipeline(
        self,
        request: PipelineRunRequest,
        job_id: Optional[str] = None,
    ) -> PipelineStatusResponse:
        """
        Execute full end-to-end disaster analysis pipeline.
        
        Handles:
        - Real pipeline execution with automatic sensor selection.
        - CDSE credential verification and blocked data path.
        - Demo mode execution with full layer synthesis.
        """
        job_id = job_id or f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # -----------------------------------------------------------------
        # STEP 1: VALIDATING
        # -----------------------------------------------------------------
        self._update_stage(
            job_id,
            PipelineStage.VALIDATING,
            progress_percent=5,
            message="Validating Area of Interest, event date, and language settings...",
        )

        try:
            if request.aoi:
                resolved_aoi = normalize_and_validate_aoi(request.aoi)
            else:
                resolved_aoi = DEFAULT_NEPAL_AOI
            normalized_date = self._normalize_date_str(request.flood_date)
            lang = request.report_language.lower() if request.report_language in ["en", "ne"] else "en"
        except Exception as e:
            logger.error("Validation failed: %s", e)
            return self._update_stage(
                job_id,
                PipelineStage.FAILED,
                progress_percent=5,
                message=f"Validation failed: {str(e)}",
                status_code="failed",
                error_message=str(e),
            )

        # -----------------------------------------------------------------
        # STEP 2: JOB REGISTRATION
        # -----------------------------------------------------------------
        job_rec = AnalysisRecord(
            job_id=job_id,
            status=AnalysisStatus.PROCESSING,
            aoi=resolved_aoi,
            flood_date=normalized_date,
            report_language=lang,
            created_at=now,
            updated_at=now,
        )
        self.repo.save(job_rec)

        # -----------------------------------------------------------------
        # STEP 3: DISCOVERING SATELLITES (Sentinel-1 & Sentinel-2)
        # -----------------------------------------------------------------
        self._update_stage(
            job_id,
            PipelineStage.DISCOVERING_SATELLITES,
            progress_percent=15,
            message="Querying Copernicus CDSE for Sentinel-1 SAR and Sentinel-2 optical constellations...",
        )

        days_before = getattr(request, "pre_event_window_days", 15) or 15
        days_after = getattr(request, "post_event_window_days", 7) or 7

        # Dynamically widen search windows if explicit reference/comparison dates are specified
        try:
            ev_dt = datetime.strptime(normalized_date, "%Y-%m-%d").date()
            if getattr(request, "reference_date", None):
                ref_dt = datetime.strptime(self._normalize_date_str(request.reference_date), "%Y-%m-%d").date()
                diff_ref = (ev_dt - ref_dt).days
                if diff_ref > 0:
                    days_before = max(days_before, diff_ref + 2)
            if getattr(request, "comparison_date", None):
                comp_dt = datetime.strptime(self._normalize_date_str(request.comparison_date), "%Y-%m-%d").date()
                diff_comp = (comp_dt - ev_dt).days
                if diff_comp > 0:
                    days_after = max(days_after, diff_comp + 2)
        except Exception as e:
            logger.debug("Date offset resolution note: %s", e)

        try:
            s1_discovery = self.sat_service.discover_sentinel_1(
                aoi=resolved_aoi,
                flood_date=normalized_date,
                days_before=days_before,
                days_after=days_after,
            )
            s2_discovery = self.sat_service.discover_sentinel_2(
                aoi=resolved_aoi,
                flood_date=normalized_date,
                days_before=days_before,
                days_after=days_after,
            )
        except Exception as e:
            logger.warning("CDSE discovery catalog query warning: %s", e)
            s1_discovery = None
            s2_discovery = None

        # -----------------------------------------------------------------
        # STEP 4: SELECTING SCENES & SMART ROUTER
        # -----------------------------------------------------------------
        self._update_stage(
            job_id,
            PipelineStage.SELECTING_SCENES,
            progress_percent=25,
            message="Evaluating Smart Satellite Router decision matrix and matching relative orbit pairs...",
        )

        router_decision = self.sat_service.route_analysis(
            aoi=resolved_aoi,
            flood_date=normalized_date,
        )

        scene_selection = self.selector.select_scenes(
            aoi=resolved_aoi,
            flood_date=normalized_date,
            days_before=days_before,
            days_after=days_after,
        )

        # -----------------------------------------------------------------
        # STEP 5: PREPARING ARD & CDSE AUTHENTICATION CHECK
        # -----------------------------------------------------------------
        self._update_stage(
            job_id,
            PipelineStage.PREPARING_ARD,
            progress_percent=40,
            message="Checking CDSE Keycloak OAuth credentials and generating Analysis-Ready Data package...",
        )

        is_auth_configured = self.downloader.is_auth_configured()
        is_blocked = False
        blocker_msg = None

        if not is_auth_configured and not request.demo_mode:
            is_blocked = True
            blocker_msg = "CDSE binary satellite download credentials unavailable"
            logger.info("CDSE credentials unavailable for job %s. Gating execution to blocked-data path.", job_id)

        # Prepare ARD manifest
        ard_pkg = self.ard_service.prepare_ard(
            PrepareArdRequest(
                aoi=resolved_aoi,
                flood_date=normalized_date,
                job_id=job_id,
                pre_event_window_days=days_before,
                post_event_window_days=days_after,
            )
        )

        job_out_dir = self.outputs_dir / job_id
        job_out_dir.mkdir(parents=True, exist_ok=True)

        if request.demo_mode:
            logger.info("Running pipeline in DEMO MODE for job %s", job_id)
            if request.aoi is None:
                ref_dir = self.outputs_dir / "job_nepal_2026_case"
                if ref_dir.exists():
                    for ref_file in ref_dir.glob("*.geojson"):
                        dest_file = job_out_dir / ref_file.name
                        if not dest_file.exists():
                            shutil.copy(ref_file, dest_file)

        # -----------------------------------------------------------------
        # STEP 6: RUNNING SEGMENTATION (AI FloodUNet)
        # -----------------------------------------------------------------
        self._update_stage(
            job_id,
            PipelineStage.RUNNING_SEGMENTATION,
            progress_percent=55,
            message="Executing Kuro Siwo FloodUNet deep learning inference...",
            is_blocked=is_blocked,
            blocker=blocker_msg,
        )

        try:
            if request.demo_mode:
                seg_res = self._create_demo_segmentation(job_id, ard_pkg)
            elif is_blocked:
                # Do NOT fake flood segmentation. Preserve null/blocked state.
                seg_res = self._create_blocked_segmentation(job_id, ard_pkg)
            else:
                seg_res = self.seg_service.segment_ard(
                    SegmentationRequest(
                        job_id=job_id,
                        confidence_threshold=0.5,
                    )
                )
        except Exception as e:
            logger.warning("Segmentation execution note: %s", e)
            seg_res = self._create_blocked_segmentation(job_id, ard_pkg)

        # -----------------------------------------------------------------
        # STEP 7: RUNNING GEOSPATIAL ANALYSIS
        # -----------------------------------------------------------------
        self._update_stage(
            job_id,
            PipelineStage.RUNNING_GEOSPATIAL,
            progress_percent=70,
            message="Overlaying OpenStreetMap buildings, roads, and bridges with flood polygons...",
            is_blocked=is_blocked,
            blocker=blocker_msg,
        )

        try:
            geo_res = self.geo_service.analyze_job_impact(
                GeospatialAnalysisRequest(
                    job_id=job_id,
                )
            )
        except Exception as e:
            logger.error("Geospatial analysis failed: %s", e)
            return self._update_stage(
                job_id,
                PipelineStage.FAILED,
                progress_percent=70,
                message=f"Geospatial analysis failed: {str(e)}",
                status_code="failed",
                error_message=str(e),
            )

        # -----------------------------------------------------------------
        # STEP 8: RUNNING CONNECTIVITY ANALYSIS
        # -----------------------------------------------------------------
        self._update_stage(
            job_id,
            PipelineStage.RUNNING_CONNECTIVITY,
            progress_percent=85,
            message="Building road network graph and evaluating settlement isolation and hospital reachability...",
            is_blocked=is_blocked,
            blocker=blocker_msg,
        )

        try:
            conn_res = self.conn_service.analyze_connectivity(
                ConnectivityAnalysisRequest(
                    job_id=job_id,
                )
            )
        except Exception as e:
            logger.error("Connectivity analysis failed: %s", e)
            return self._update_stage(
                job_id,
                PipelineStage.FAILED,
                progress_percent=85,
                message=f"Connectivity analysis failed: {str(e)}",
                status_code="failed",
                error_message=str(e),
            )

        # -----------------------------------------------------------------
        # STEP 9: BUILDING STRUCTURED RESULTS
        # -----------------------------------------------------------------
        self._update_stage(
            job_id,
            PipelineStage.BUILDING_RESULTS,
            progress_percent=92,
            message="Synthesizing central machine-readable results with provenance and fact registry...",
            is_blocked=is_blocked,
            blocker=blocker_msg,
        )

        try:
            structured_res = self.res_service.build_analysis_result(job_id)
        except Exception as e:
            logger.error("Results aggregation failed: %s", e)
            return self._update_stage(
                job_id,
                PipelineStage.FAILED,
                progress_percent=92,
                message=f"Results aggregation failed: {str(e)}",
                status_code="failed",
                error_message=str(e),
            )

        # -----------------------------------------------------------------
        # STEP 10: GENERATING SITUATION REPORT
        # -----------------------------------------------------------------
        self._update_stage(
            job_id,
            PipelineStage.GENERATING_REPORT,
            progress_percent=98,
            message=f"Generating verified humanitarian disaster situation report in {lang.upper()}...",
            is_blocked=is_blocked,
            blocker=blocker_msg,
        )

        try:
            situation_rep = self.rep_service.generate_report(
                job_id,
                ReportGenerateRequest(language=lang),
            )
        except Exception as e:
            logger.error("Report generation failed: %s", e)
            return self._update_stage(
                job_id,
                PipelineStage.FAILED,
                progress_percent=98,
                message=f"Report generation failed: {str(e)}",
                status_code="failed",
                error_message=str(e),
            )

        # -----------------------------------------------------------------
        # FINAL STAGE COMPLETION
        # -----------------------------------------------------------------
        if is_blocked:
            final_stage = PipelineStage.COMPLETED_WITH_LIMITATIONS
            status_str = "completed_with_limitations"
            completion_msg = (
                f"Analysis pipeline completed for job '{job_id}' with satellite data blockers. "
                "Copernicus CDSE binary download credentials unavailable; physical flood rasters withheld. "
                "Infrastructure assets and road network connectivity baseline evaluated."
            )
        else:
            final_stage = PipelineStage.COMPLETED
            status_str = "completed"
            completion_msg = f"Analysis pipeline completed successfully for job '{job_id}'."

        final_state = self._update_stage(
            job_id,
            final_stage,
            progress_percent=100,
            message=completion_msg,
            status_code=status_str,
            is_blocked=is_blocked,
            blocker=blocker_msg,
        )
        final_state.flood_date = normalized_date
        final_state.report_language = lang
        final_state.demo_mode = request.demo_mode
        final_state.available_layers = self.list_available_layers(job_id)
        final_state.satellite_status = getattr(structured_res.satellite.status, "value", str(structured_res.satellite.status))
        final_state.segmentation_status = getattr(structured_res.segmentation.status, "value", str(structured_res.segmentation.status))
        final_state.geospatial_status = getattr(structured_res.geospatial.status, "value", str(structured_res.geospatial.status))
        final_state.connectivity_status = getattr(structured_res.connectivity.status, "value", str(structured_res.connectivity.status))
        final_state.report_status = situation_rep.metadata.report_status

        # Brief summary dictionary for quick cards
        bldgs = structured_res.geospatial.affected_buildings
        bldgs_cnt = (
            (bldgs.get("heavily_affected", 0) + bldgs.get("potentially_affected", 0))
            if isinstance(bldgs, dict)
            else bldgs
        )
        roads = structured_res.geospatial.affected_roads
        roads_cnt = (
            (roads.get("potentially_blocked", 0) + roads.get("heavily_affected", 0))
            if isinstance(roads, dict)
            else roads
        )
        roads_km = roads.get("affected_length_km") if isinstance(roads, dict) else None
        bridges = structured_res.geospatial.affected_bridges
        bridges_cnt = (
            bridges.get("heavily_affected", 0)
            if isinstance(bridges, dict)
            else bridges
        )
        final_state.summary = {
            "flood_area_km2": structured_res.segmentation.flood_area_km2,
            "affected_buildings": bldgs_cnt,
            "affected_buildings_details": bldgs,
            "affected_roads": roads_cnt,
            "affected_roads_km": roads_km,
            "affected_roads_details": roads,
            "affected_bridges": bridges_cnt,
            "affected_bridges_details": bridges,
            "blocked_roads": structured_res.connectivity.blocked_road_count,
            "isolated_settlements": structured_res.connectivity.flood_isolated_settlement_count,
            "hospital_access_lost": structured_res.connectivity.hospital_access_lost_count,
            "debris_zones": structured_res.geospatial.change_zone_count,
        }

        return final_state

    def _create_demo_segmentation(self, job_id: str, ard_pkg: Any) -> SegmentationResponse:
        """Create synthetic demo segmentation with valid layers for UI preview."""
        now = datetime.now(timezone.utc)
        bounds = (
            ard_pkg.spatial_reference.bounding_box
            if (ard_pkg and ard_pkg.spatial_reference and ard_pkg.spatial_reference.bounding_box)
            else [84.95, 27.75, 85.50, 28.25]
        )
        crs = (
            ard_pkg.spatial_reference.crs
            if (ard_pkg and ard_pkg.spatial_reference and ard_pkg.spatial_reference.crs)
            else "EPSG:4326"
        )
        mask_path = self.outputs_dir / f"{job_id}_flood_mask.tif"
        if not mask_path.exists():
            import rasterio
            from rasterio.transform import from_bounds
            import numpy as np
            transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], 64, 64)
            demo_mask = np.zeros((64, 64), dtype=np.uint8)
            demo_mask[20:35, 25:40] = 1
            with rasterio.open(
                mask_path, "w", driver="GTiff", height=64, width=64, count=1, dtype="uint8", crs=crs, transform=transform
            ) as dst:
                dst.write(demo_mask, 1)

        min_x, min_y, max_x, max_y = float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
        # Approximate geodesic area in km2 based on geographic bounding box
        dx_km = abs(max_x - min_x) * 111.32 * np.cos(np.radians((min_y + max_y) / 2.0))
        dy_km = abs(max_y - min_y) * 110.57
        total_aoi_km2 = max(0.1, dx_km * dy_km)
        # Synthetic demo flood is approximately 1.8% to 3.5% of total AOI
        flood_fraction = 0.024
        demo_flood_km2 = round(total_aoi_km2 * flood_fraction, 2)
        demo_flood_m2 = round(demo_flood_km2 * 1_000_000, 1)

        summary = AffectedAreaSummary(
            total_valid_pixels=100000,
            flood_pixel_count=int(100000 * flood_fraction),
            flood_percentage=round(flood_fraction * 100, 2),
            flood_area_m2=demo_flood_m2,
            flood_area_sqkm=demo_flood_km2,
        )
        poly_min_x = round(min_x + 0.3 * (max_x - min_x), 6)
        poly_max_x = round(min_x + 0.6 * (max_x - min_x), 6)
        poly_min_y = round(min_y + 0.3 * (max_y - min_y), 6)
        poly_max_y = round(min_y + 0.6 * (max_y - min_y), 6)
        demo_poly = {
            "type": "Feature",
            "properties": {"name": "DEMO_FLOOD_EXTENT", "area_sqkm": demo_flood_km2},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [poly_min_x, poly_min_y],
                        [poly_max_x, poly_min_y],
                        [poly_max_x, poly_max_y],
                        [poly_min_x, poly_max_y],
                        [poly_min_x, poly_min_y],
                    ]
                ],
            },
        }
        res = SegmentationResponse(
            job_id=job_id,
            status="completed",
            model_name="flood_unet_s1 (DEMO_MODE)",
            model_architecture="FloodUNet (Encoder-Decoder with Skip Connections)",
            confidence_threshold=0.5,
            summary=summary,
            metrics=None,
            mask_raster_path=str(mask_path.resolve()),
            geojson_features_count=1,
            geojson={"type": "FeatureCollection", "features": [demo_poly]},
            crs=crs,
            spatial_bounds=list(bounds),
            created_at=now,
            message="Demo mode segmentation generated with synthetic flood polygon.",
        )
        res_json = self.outputs_dir / f"{job_id}_segmentation.json"
        with open(res_json, "w", encoding="utf-8") as f:
            json.dump(res.model_dump(mode="json"), f, indent=2)
        return res

    def _create_blocked_segmentation(self, job_id: str, ard_pkg: Any) -> SegmentationResponse:
        """Create a safe blocked segmentation result preserving null metrics."""
        now = datetime.now(timezone.utc)
        bounds = (
            ard_pkg.spatial_reference.bounding_box
            if (ard_pkg and ard_pkg.spatial_reference and ard_pkg.spatial_reference.bounding_box)
            else [84.95, 27.75, 85.50, 28.25]
        )
        crs = (
            ard_pkg.spatial_reference.crs
            if (ard_pkg and ard_pkg.spatial_reference and ard_pkg.spatial_reference.crs)
            else "EPSG:4326"
        )

        # Empty mask raster placeholder
        mask_path = self.outputs_dir / f"{job_id}_flood_mask.tif"
        if not mask_path.exists():
            import rasterio
            from rasterio.transform import from_bounds
            transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], 64, 64)
            with rasterio.open(
                mask_path,
                "w",
                driver="GTiff",
                height=64,
                width=64,
                count=1,
                dtype="uint8",
                crs=crs,
                transform=transform,
            ) as dst:
                import numpy as np
                dst.write(np.zeros((64, 64), dtype=np.uint8), 1)

        empty_fc = {"type": "FeatureCollection", "features": []}
        summary = AffectedAreaSummary(
            total_valid_pixels=100000,
            flood_pixel_count=0,
            flood_percentage=0.0,
            flood_area_m2=0.0,
            flood_area_sqkm=0.0,
        )

        res = SegmentationResponse(
            job_id=job_id,
            status="blocked",
            model_name="flood_unet_s1",
            model_architecture="FloodUNet (Encoder-Decoder with Skip Connections)",
            confidence_threshold=0.5,
            summary=summary,
            metrics=None,
            mask_raster_path=str(mask_path.resolve()),
            geojson_features_count=0,
            geojson=empty_fc,
            crs=crs,
            spatial_bounds=list(bounds),
            created_at=now,
            message="Segmentation blocked: CDSE binary satellite download credentials unavailable.",
        )

        # Save to disk
        res_json = self.outputs_dir / f"{job_id}_segmentation.json"
        with open(res_json, "w", encoding="utf-8") as f:
            json.dump(res.model_dump(mode="json"), f, indent=2)

        return res


_pipeline_service_instance: Optional[PipelineExecutionService] = None


def get_pipeline_service() -> PipelineExecutionService:
    """Dependency provider for PipelineExecutionService."""
    global _pipeline_service_instance
    if _pipeline_service_instance is None:
        _pipeline_service_instance = PipelineExecutionService()
    return _pipeline_service_instance
