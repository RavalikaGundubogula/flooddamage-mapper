import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.results import (
    FactStatus,
    SectionStatus,
    StructuredAnalysisResult,
)
from app.services.connectivity_service import get_connectivity_service
from app.services.geospatial_service import get_geospatial_service
from app.services.results_service import get_results_service
from backend.tests.test_connectivity import (  # reuse deterministic fixtures
    deterministic_osm_network,
    mock_module5_geospatial_job,
)

client = TestClient(app)


def test_aggregate_results_with_complete_job(mock_module5_geospatial_job):
    """Verify central result aggregation combines Modules 1 through 6 accurately."""
    job_id, osm_path = mock_module5_geospatial_job
    service = get_results_service()
    conn_service = get_connectivity_service()

    # Pre-run Module 6 connectivity
    from app.schemas.connectivity import ConnectivityAnalysisRequest
    conn_service.analyze_connectivity(ConnectivityAnalysisRequest(job_id=job_id, osm_geojson_path=osm_path))

    # Build central structured result
    res = service.build_analysis_result(job_id)

    assert isinstance(res, StructuredAnalysisResult)
    assert res.job_id == job_id
    assert res.overall_status in ["completed", "completed_with_limitations"]

    # Module 1 metadata
    assert res.metadata.job_id == job_id
    assert res.metadata.report_language == "en"

    # Module 4 segmentation metrics
    assert res.segmentation.status in [SectionStatus.ESTIMATED, SectionStatus.AVAILABLE]
    assert res.segmentation.flood_area_km2 == 0.12
    assert res.segmentation.flood_percentage == 12.0

    # Module 5 geospatial metrics
    assert res.geospatial.status in [SectionStatus.ESTIMATED, SectionStatus.AVAILABLE]
    assert res.geospatial.affected_roads is not None
    assert res.geospatial.affected_roads["total_segments"] > 0

    # Module 6 connectivity metrics
    assert res.connectivity.status == SectionStatus.ESTIMATED
    assert res.connectivity.blocked_road_count == 2
    assert res.connectivity.flood_isolated_settlement_count == 1
    assert res.connectivity.hospital_access_lost_count == 1

    # Factuality registry
    assert "flood_area_km2" in res.facts
    assert res.facts["flood_area_km2"].value == 0.12
    assert res.facts["flood_area_km2"].status == FactStatus.ESTIMATED

    assert "blocked_roads" in res.facts
    assert res.facts["blocked_roads"].value == 2
    assert res.facts["blocked_roads"].status == FactStatus.ESTIMATED

    assert "flood_isolated_settlements" in res.facts
    assert res.facts["flood_isolated_settlements"].value == 1
    assert res.facts["flood_isolated_settlements"].status == FactStatus.ESTIMATED

    # Provenance integrity
    assert "EMSR927 is strictly reserved as an external post-event validation reference" in res.provenance.emsr927_note
    assert "EMSR927" not in str(res.provenance.satellite_sources)


def test_blocked_satellite_data_preserves_null_not_zero():
    """Verify that CDSE satellite binary blocker keeps flood measurements null and NEVER converts to 0."""
    service = get_results_service()
    job_id = "job_test_blocked_case"
    from unittest.mock import MagicMock, patch

    mock_pkg = MagicMock()
    mock_pkg.job_id = job_id
    mock_pkg.status = "contract_ready"
    mock_pkg.strategy_used = "sentinel-1"
    mock_pkg.sar_ard = None
    mock_pkg.optical_ard = None
    mock_pkg.pre_s1_scene = None
    mock_pkg.post_s1_scene = None

    with patch.object(service.ard_service, "get_ard_by_job_id", return_value=mock_pkg):
        res = service.build_analysis_result(job_id)

    assert res.overall_status == "satellite_binary_data_blocked"
    assert res.source_data_status == "satellite_binary_data_blocked"
    assert res.satellite.status == SectionStatus.BLOCKED
    assert res.satellite.data_availability == "blocked"
    assert res.segmentation.status == SectionStatus.BLOCKED
    assert res.segmentation.flood_area_km2 is None
    assert res.segmentation.flood_pixel_count is None
    assert res.segmentation.flood_percentage is None
    assert res.geospatial.status == SectionStatus.BLOCKED
    assert res.geospatial.affected_buildings is None
    assert res.geospatial.affected_roads is None
    assert res.connectivity.status == SectionStatus.BLOCKED
    assert res.connectivity.blocked_road_count is None
    assert res.connectivity.flood_isolated_settlement_count is None
    assert res.facts["flood_area_km2"].status == FactStatus.BLOCKED
    assert res.facts["flood_area_km2"].value is None
    assert any("CDSE binary raster downloads are currently blocked" in lim for lim in res.limitations)


def test_missing_upstream_modules_handled_gracefully():
    """Verify that a job with no upstream processing does not crash and marks sections NOT_RUN."""
    service = get_results_service()
    job_id = "test_empty_unrun_job_xyz"

    res = service.build_analysis_result(job_id)

    assert res.job_id == job_id
    assert res.overall_status in ["completed_with_limitations", "failed"]
    assert res.segmentation.status == SectionStatus.NOT_RUN
    assert res.geospatial.status == SectionStatus.NOT_RUN
    assert res.connectivity.status == SectionStatus.NOT_RUN
    assert res.facts["flood_area_km2"].status == FactStatus.NOT_RUN
    assert res.facts["flood_area_km2"].value is None


def test_results_api_endpoints(mock_module5_geospatial_job):
    """Verify POST and GET /analysis/{job_id}/results endpoints."""
    job_id, _ = mock_module5_geospatial_job

    # 1. POST /analysis/{job_id}/results
    post_res = client.post(f"/analysis/{job_id}/results")
    assert post_res.status_code == 200
    data = post_res.json()
    assert data["job_id"] == job_id
    assert "metadata" in data
    assert "satellite" in data
    assert "segmentation" in data
    assert "geospatial" in data
    assert "connectivity" in data
    assert "facts" in data

    # 2. GET /analysis/{job_id}/results
    get_res = client.get(f"/analysis/{job_id}/results")
    assert get_res.status_code == 200
    assert get_res.json()["job_id"] == job_id
