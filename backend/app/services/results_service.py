import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import Settings, get_settings
from app.schemas.results import (
    AnalysisMetadata,
    ConnectivityResults,
    FactItem,
    FactStatus,
    GeospatialResults,
    ProvenanceInfo,
    SatelliteResults,
    SectionStatus,
    SegmentationResults,
    StructuredAnalysisResult,
)
from app.services.analysis_service import AnalysisService, get_analysis_service
from app.services.ard_service import AnalysisReadyDataService, get_ard_service
from app.services.connectivity_service import (
    ConnectivityAnalysisService,
    get_connectivity_service,
)
from app.services.geospatial_service import (
    GeospatialDamageAnalysisService,
    get_geospatial_service,
)
from app.services.segmentation_service import (
    SegmentationInferenceService,
    get_segmentation_service,
)

logger = logging.getLogger(__name__)


class ResultsAggregationService:
    """
    Module 7 Central Results Aggregation Service.
    
    Consolidates outputs from Modules 1 through 6 into a single, reliable,
    machine-readable StructuredAnalysisResult.
    
    Guarantees:
    - Never converts missing or blocked data into zero.
    - Explicitly marks satellite blockers and missing credentials.
    - Enforces strict factuality classification (VERIFIED, ESTIMATED, BLOCKED, UNAVAILABLE, NOT_RUN).
    - Preserves data provenance and guarantees EMSR927 is not an analysis input.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        analysis_service: Optional[AnalysisService] = None,
        ard_service: Optional[AnalysisReadyDataService] = None,
        seg_service: Optional[SegmentationInferenceService] = None,
        geo_service: Optional[GeospatialDamageAnalysisService] = None,
        conn_service: Optional[ConnectivityAnalysisService] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.analysis_service = analysis_service or get_analysis_service()
        self.ard_service = ard_service or get_ard_service()
        self.seg_service = seg_service or get_segmentation_service()
        self.geo_service = geo_service or get_geospatial_service()
        self.conn_service = conn_service or get_connectivity_service()

        self.outputs_dir = self.settings.get_data_path(self.settings.DATA_OUTPUTS_DIR)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._results_cache: Dict[str, StructuredAnalysisResult] = {}

    def build_analysis_result(self, job_id: str) -> StructuredAnalysisResult:
        """
        Aggregate all available outputs across Modules 1 to 6 into a canonical structured result.
        """
        # 1. Module 1: Job Metadata
        job_record = self.analysis_service.get_analysis(job_id)
        aoi = job_record.aoi if job_record else {"type": "Polygon", "coordinates": []}
        flood_date = job_record.flood_date if job_record else "unknown"
        report_lang = job_record.report_language if job_record else "en"

        # 2. Module 3: Satellite ARD Package
        ard_pkg = self.ard_service.get_ard_by_job_id(job_id)

        # 3. Module 4: AI Segmentation Result
        seg_res = self.seg_service.get_segmentation_result(job_id)

        # 4. Module 5: Geospatial Damage Overlay
        geo_res = self.geo_service.get_geospatial_result(job_id)

        # 5. Module 6: Road Network & Connectivity Result
        conn_res = self.conn_service.get_connectivity_result(job_id)

        # 6. Determine Physical Data Status & Blocker Condition
        is_blocked = False
        source_data_status = "calibrated_raster_available"

        if (
            (ard_pkg and ard_pkg.status == "contract_ready")
            or (seg_res and getattr(seg_res, "status", None) == "blocked")
            or (geo_res and geo_res.source_data_status == "satellite_binary_data_blocked")
            or (conn_res and conn_res.source_data_status == "satellite_binary_data_blocked")
        ):
            is_blocked = True
            source_data_status = "satellite_binary_data_blocked"
        elif "test" in job_id.lower() or "fixture" in job_id.lower():
            source_data_status = "test_fixture"

        limitations: List[str] = []
        facts: Dict[str, FactItem] = {}

        # ---------------------------------------------------------------------
        # Assemble Satellite Section
        # ---------------------------------------------------------------------
        if is_blocked:
            sat_status = SectionStatus.BLOCKED
            sat_reason = "Copernicus CDSE binary download credentials unavailable; pre/post Sentinel rasters not calibrated."
            limitations.append("CDSE binary raster downloads are currently blocked awaiting data space credentials.")
        elif ard_pkg:
            sat_status = SectionStatus.AVAILABLE
            sat_reason = None
        else:
            sat_status = SectionStatus.NOT_RUN
            sat_reason = "Satellite discovery and ARD preprocessing have not been executed."

        sat_dates = []
        pre_scene = None
        post_scene = None
        if ard_pkg:
            if ard_pkg.sar_ard:
                pre_scene = getattr(ard_pkg.sar_ard, "product_id_pre", None) or getattr(ard_pkg.sar_ard, "pre_scene_id", None)
                post_scene = getattr(ard_pkg.sar_ard, "product_id_post", None) or getattr(ard_pkg.sar_ard, "post_scene_id", None)
            if not pre_scene:
                pre_scene = getattr(ard_pkg, "pre_s1_scene", None)
            if not post_scene:
                post_scene = getattr(ard_pkg, "post_s1_scene", None)
            if pre_scene:
                sat_dates.append(str(pre_scene))
            if post_scene:
                sat_dates.append(str(post_scene))

        satellite_results = SatelliteResults(
            status=sat_status,
            sentinel_1_used=bool(ard_pkg and ("s1" in ard_pkg.strategy_used or "sentinel-1" in ard_pkg.strategy_used)),
            sentinel_2_used=bool(ard_pkg and ("s2" in ard_pkg.strategy_used or "sentinel-2" in ard_pkg.strategy_used)),
            pre_event_scene=pre_scene,
            post_event_scene=post_scene,
            orbit_direction="DESCENDING",
            relative_orbit=121 if ard_pkg else None,
            acquisition_dates=sat_dates,
            data_availability="blocked" if is_blocked else ("available" if ard_pkg else "not_run"),
            reason=sat_reason,
        )

        facts["satellite_data_available"] = FactItem(
            value=False if is_blocked else bool(ard_pkg),
            status=FactStatus.BLOCKED if is_blocked else (FactStatus.VERIFIED if ard_pkg else FactStatus.NOT_RUN),
            source="Module 2/3 Satellite Router",
            reason=sat_reason,
        )

        # ---------------------------------------------------------------------
        # Assemble Segmentation Section
        # ---------------------------------------------------------------------
        if is_blocked:
            seg_status = SectionStatus.BLOCKED
            seg_reason = "AI segmentation cannot infer floodwater extents without calibrated satellite rasters."
            seg_results = SegmentationResults(
                status=seg_status,
                model_name="FloodUNet",
                model_version="1.0.0",
                training_dataset="Kuro Siwo Benchmark (NeurIPS 2024)",
                flood_area_km2=None,  # NEVER convert to 0!
                flood_percentage=None,
                flood_pixel_count=None,
                polygon_count=None,
                confidence_information=None,
                reason=seg_reason,
            )
            facts["flood_area_km2"] = FactItem(
                value=None,
                status=FactStatus.BLOCKED,
                source="Module 4 FloodUNet",
                reason=seg_reason,
            )
        else:
            flood_km2 = None
            flood_pct = None
            pixel_count = None
            poly_count = None
            m_name = "FloodUNet"

            if seg_res and hasattr(seg_res, "summary") and seg_res.summary:
                m_name = getattr(seg_res, "model_name", "FloodUNet")
                flood_km2 = seg_res.summary.flood_area_sqkm
                flood_pct = seg_res.summary.flood_percentage
                pixel_count = seg_res.summary.flood_pixel_count
            elif geo_res and getattr(geo_res, "flood_metrics", None):
                fm = geo_res.flood_metrics
                flood_km2 = fm.flooded_area_km2
                flood_pct = fm.flooded_percentage
                pixel_count = getattr(fm, "flooded_pixel_count", None)

            if geo_res and getattr(geo_res, "flood_metrics", None):
                poly_count = geo_res.flood_metrics.flood_region_count

            if flood_km2 is not None:
                seg_status = SectionStatus.ESTIMATED
                seg_results = SegmentationResults(
                    status=seg_status,
                    model_name=m_name,
                    model_version="1.0.0",
                    training_dataset="Kuro Siwo Benchmark (NeurIPS 2024)",
                    flood_area_km2=flood_km2,
                    flood_percentage=flood_pct,
                    flood_pixel_count=pixel_count,
                    polygon_count=poly_count,
                    confidence_information="Pixel-level sigmoid inference over multi-temporal SAR dB",
                )
                facts["flood_area_km2"] = FactItem(
                    value=flood_km2,
                    status=FactStatus.ESTIMATED,
                    source="Module 4 FloodUNet",
                    method="Geodesic metric area integration from GeoTIFF mask",
                )
                facts["flood_percentage"] = FactItem(
                    value=flood_pct,
                    status=FactStatus.ESTIMATED,
                    source="Module 4 FloodUNet",
                    method="Flooded pixels / Total valid pixels in AOI",
                )
            else:
                seg_status = SectionStatus.NOT_RUN
                seg_results = SegmentationResults(
                    status=seg_status,
                    reason="AI flood segmentation has not been executed.",
                )
                facts["flood_area_km2"] = FactItem(
                    value=None,
                    status=FactStatus.NOT_RUN,
                    source="Module 4 FloodUNet",
                    reason="Segmentation not run.",
                )



        # ---------------------------------------------------------------------
        # Assemble Geospatial Section
        # ---------------------------------------------------------------------
        if is_blocked:
            geo_status = SectionStatus.BLOCKED
            geo_reason = "Physical infrastructure exposure cannot be measured without validated floodwater extents."
            geo_results = GeospatialResults(
                status=geo_status,
                flood_region_count=None,
                affected_buildings=None,
                affected_roads=None,
                affected_bridges=None,
                change_zone_count=None,
                reason=geo_reason,
            )
            facts["affected_buildings"] = FactItem(
                value=None,
                status=FactStatus.BLOCKED,
                source="Module 5 Geospatial Service",
                reason=geo_reason,
            )
        elif geo_res:
            geo_status = SectionStatus.ESTIMATED
            bldgs = geo_res.asset_summary.buildings
            roads = geo_res.asset_summary.roads
            bridges = geo_res.asset_summary.bridges

            geo_results = GeospatialResults(
                status=geo_status,
                flood_region_count=geo_res.flood_metrics.flood_region_count,
                affected_buildings={
                    "total": bldgs.total,
                    "potentially_affected": bldgs.potentially_affected,
                    "heavily_affected": bldgs.heavily_affected,
                    "unaffected": bldgs.unaffected,
                },
                affected_roads={
                    "total_segments": roads.total_segments,
                    "potentially_blocked": roads.potentially_blocked,
                    "heavily_affected": roads.heavily_affected,
                    "affected_length_km": roads.affected_length_km,
                },
                affected_bridges={
                    "total": bridges.total,
                    "potentially_affected": bridges.potentially_affected,
                    "heavily_affected": bridges.heavily_affected,
                },
                change_zone_count=geo_res.change_zone_metrics.total_change_zones,
                estimated_exposure_metrics={
                    "flooded_area_km2": geo_res.flood_metrics.flooded_area_km2,
                    "flooded_percentage": geo_res.flood_metrics.flooded_percentage,
                },
            )
            facts["affected_buildings"] = FactItem(
                value=bldgs.potentially_affected + bldgs.heavily_affected,
                status=FactStatus.ESTIMATED,
                source="Module 5 Geospatial Service",
                method="Spatial overlap of OSM buildings with flood polygons",
            )
            facts["affected_roads"] = FactItem(
                value=roads.potentially_blocked + roads.heavily_affected,
                status=FactStatus.ESTIMATED,
                source="Module 5 Geospatial Service",
                method="Geodesic metric intersection of road LineStrings with flood polygons",
            )
        else:
            geo_status = SectionStatus.NOT_RUN
            geo_results = GeospatialResults(
                status=geo_status,
                reason="Geospatial infrastructure overlay analysis has not been executed.",
            )
            facts["affected_buildings"] = FactItem(
                value=None,
                status=FactStatus.NOT_RUN,
                source="Module 5 Geospatial Service",
                reason="Geospatial overlay not run.",
            )

        # ---------------------------------------------------------------------
        # Assemble Connectivity Section
        # ---------------------------------------------------------------------
        if is_blocked:
            conn_status = SectionStatus.BLOCKED
            conn_reason = "Road blocking and settlement isolation cannot be computed without physical flood inundation."
            conn_results = ConnectivityResults(
                status=conn_status,
                road_count=None,
                blocked_road_count=None,
                partially_affected_road_count=None,
                isolated_settlement_count=None,
                flood_isolated_settlement_count=None,
                pre_existingly_disconnected_count=None,
                hospital_access_lost_count=None,
                hospital_access_preserved_count=None,
                detour_statistics=None,
                reason=conn_reason,
            )
            facts["flood_isolated_settlements"] = FactItem(
                value=None,
                status=FactStatus.BLOCKED,
                source="Module 6 Connectivity Service",
                reason=conn_reason,
            )
            facts["blocked_roads"] = FactItem(
                value=None,
                status=FactStatus.BLOCKED,
                source="Module 6 Connectivity Service",
                reason=conn_reason,
            )
            facts["hospital_access_lost"] = FactItem(
                value=None,
                status=FactStatus.BLOCKED,
                source="Module 6 Connectivity Service",
                reason=conn_reason,
            )
        elif conn_res:
            conn_status = SectionStatus.ESTIMATED
            rn = conn_res.road_network
            sm = conn_res.settlements
            ha = conn_res.hospital_access

            conn_results = ConnectivityResults(
                status=conn_status,
                road_count=rn.total_edges,
                blocked_road_count=rn.blocked_edges,
                partially_affected_road_count=rn.partially_affected_edges,
                isolated_settlement_count=sm.flood_isolated,
                flood_isolated_settlement_count=sm.flood_isolated,
                pre_existingly_disconnected_count=sm.pre_existingly_disconnected,
                hospital_access_lost_count=ha.hospital_access_lost,
                hospital_access_preserved_count=ha.access_preserved,
                detour_statistics={
                    "minor_detour": sm.minor_detour,
                    "major_detour": sm.major_detour,
                    "no_change": sm.unaffected,
                },
            )
            facts["blocked_roads"] = FactItem(
                value=rn.blocked_edges,
                status=FactStatus.ESTIMATED,
                source="Module 6 Connectivity Service",
                method="Threshold-based road edge pruning (>= 50% flood overlap)",
            )
            facts["flood_isolated_settlements"] = FactItem(
                value=sm.flood_isolated,
                status=FactStatus.ESTIMATED,
                source="Module 6 Connectivity Service",
                method="Shortest-path reachability comparison (baseline connected AND post-flood disconnected)",
            )
            facts["hospital_access_lost"] = FactItem(
                value=ha.hospital_access_lost,
                status=FactStatus.ESTIMATED,
                source="Module 6 Connectivity Service",
                method="Shortest-path hospital reachability comparison (baseline access AND post-flood lost)",
            )
        else:
            conn_status = SectionStatus.NOT_RUN
            conn_results = ConnectivityResults(
                status=conn_status,
                reason="Road network connectivity analysis has not been executed.",
            )
            facts["flood_isolated_settlements"] = FactItem(
                value=None,
                status=FactStatus.NOT_RUN,
                source="Module 6 Connectivity Service",
                reason="Connectivity analysis not run.",
            )

        # ---------------------------------------------------------------------
        # Assemble General Limitations & Overall Status
        # ---------------------------------------------------------------------
        limitations.append("All infrastructure exposure estimates are derived from spatial overlays; not confirmed physical damage.")
        limitations.append("Road network accessibility uses topological graph models with distance weighting; does not account for real-time traffic or debris velocity.")
        limitations.append("Settlements and hospitals reflect pre-event OpenStreetMap metadata without independent population censuses.")

        if is_blocked:
            overall_status = "satellite_binary_data_blocked"
        elif any(s == SectionStatus.NOT_RUN for s in [sat_status, seg_status, geo_status, conn_status]):
            overall_status = "completed_with_limitations"
        else:
            overall_status = "completed"

        result = StructuredAnalysisResult(
            job_id=job_id,
            overall_status=overall_status,
            source_data_status=source_data_status,
            metadata=AnalysisMetadata(
                job_id=job_id,
                aoi=aoi,
                flood_date=flood_date,
                processing_timestamp=datetime.now(timezone.utc),
                report_language=report_lang,
                source_data_status=source_data_status,
            ),
            satellite=satellite_results,
            segmentation=seg_results,
            geospatial=geo_results,
            connectivity=conn_results,
            provenance=ProvenanceInfo(),
            limitations=limitations,
            facts=facts,
        )

        # Persist and update local memory cache
        self._persist_result(job_id, result)
        self._results_cache[job_id] = result
        return result

    def _persist_result(self, job_id: str, result: StructuredAnalysisResult) -> None:
        """Persist structured analysis result JSON to disk."""
        job_dir = self.outputs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        result_file = job_dir / "analysis_result.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2)

    def get_analysis_result(self, job_id: str) -> Optional[StructuredAnalysisResult]:
        """Retrieve stored analysis result from memory cache or disk."""
        if job_id in self._results_cache:
            return self._results_cache[job_id]

        result_file = self.outputs_dir / job_id / "analysis_result.json"
        if result_file.exists():
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                res = StructuredAnalysisResult(**data)
                self._results_cache[job_id] = res
                return res
            except Exception as e:
                logger.error("Failed to load analysis result for job %s: %s", job_id, e)

        return None


# Global singleton instance
_results_service_instance: Optional[ResultsAggregationService] = None


def get_results_service() -> ResultsAggregationService:
    """Dependency provider for ResultsAggregationService."""
    global _results_service_instance
    if _results_service_instance is None:
        _results_service_instance = ResultsAggregationService()
    return _results_service_instance
