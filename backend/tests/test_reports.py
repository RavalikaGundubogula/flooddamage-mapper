import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.connectivity import ConnectivityAnalysisRequest
from app.schemas.results import ReportGenerateRequest
from app.services.analysis_service import report_generator
from app.services.connectivity_service import get_connectivity_service
from app.services.report_service import get_report_service
from backend.tests.test_connectivity import (
    deterministic_osm_network,
    mock_module5_geospatial_job,
)

client = TestClient(app)


def test_generate_english_report(mock_module5_geospatial_job):
    """Verify English situation report contains all 11 standardized sections and exact factual metrics."""
    job_id, osm_path = mock_module5_geospatial_job
    conn_service = get_connectivity_service()
    report_service = get_report_service()

    # Pre-run Module 6
    conn_service.analyze_connectivity(ConnectivityAnalysisRequest(job_id=job_id, osm_geojson_path=osm_path))

    # Generate English report
    req = ReportGenerateRequest(language="en", force_fallback=True)
    resp = report_service.generate_report(job_id, req)

    assert resp.job_id == job_id
    assert resp.language == "en"
    report = resp.report
    assert report.metadata.language == "en"
    assert report.metadata.report_status == "fallback"

    # Verify all 11 sections exist
    section_ids = [s.section_id for s in report.sections]
    assert len(section_ids) == 11
    assert "situation_summary" in section_ids
    assert "flood_extent" in section_ids
    assert "infrastructure_exposure" in section_ids
    assert "road_connectivity" in section_ids
    assert "settlement_isolation" in section_ids
    assert "hospital_accessibility" in section_ids
    assert "change_zones" in section_ids
    assert "data_sources" in section_ids
    assert "methodology" in section_ids
    assert "limitations" in section_ids
    assert "validation_notes" in section_ids

    # Numerical accuracy in text
    sec_map = {s.section_id: s for s in report.sections}
    assert "0.12" in sec_map["flood_extent"].content
    assert "12.0" in sec_map["flood_extent"].content
    assert "2" in sec_map["road_connectivity"].content  # 2 blocked edges
    assert "1" in sec_map["settlement_isolation"].content  # 1 flood isolated settlement
    assert "1" in sec_map["hospital_accessibility"].content  # 1 lost hospital access


def test_generate_nepali_report_numerical_consistency(mock_module5_geospatial_job):
    """Verify Nepali situation report contains identical numbers to the English report."""
    job_id, osm_path = mock_module5_geospatial_job
    conn_service = get_connectivity_service()
    report_service = get_report_service()

    # Pre-run Module 6
    conn_service.analyze_connectivity(ConnectivityAnalysisRequest(job_id=job_id, osm_geojson_path=osm_path))

    # Generate English and Nepali reports
    resp_en = report_service.generate_report(job_id, ReportGenerateRequest(language="en", force_fallback=True))
    resp_ne = report_service.generate_report(job_id, ReportGenerateRequest(language="ne", force_fallback=True))

    rep_en = resp_en.report
    rep_ne = resp_ne.report

    assert rep_ne.metadata.language == "ne"
    assert len(rep_ne.sections) == 11

    sec_en = {s.section_id: s for s in rep_en.sections}
    sec_ne = {s.section_id: s for s in rep_ne.sections}

    # Verify flood metrics consistency
    assert "0.12" in sec_ne["flood_extent"].content
    assert "12.0" in sec_ne["flood_extent"].content
    assert sec_en["flood_extent"].key_metrics["flood_area_km2"] == sec_ne["flood_extent"].key_metrics["flood_area_km2"]

    # Verify road connectivity metrics consistency
    assert sec_en["road_connectivity"].key_metrics["blocked_edges"] == sec_ne["road_connectivity"].key_metrics["blocked_edges"]
    assert "2" in sec_ne["road_connectivity"].content

    # Verify settlement isolation consistency
    assert sec_en["settlement_isolation"].key_metrics["flood_isolated"] == sec_ne["settlement_isolation"].key_metrics["flood_isolated"]
    assert "1" in sec_ne["settlement_isolation"].content

    # Verify hospital access consistency
    assert sec_en["hospital_accessibility"].key_metrics["access_lost"] == sec_ne["hospital_accessibility"].key_metrics["access_lost"]
    assert "1" in sec_ne["hospital_accessibility"].content


def test_blocked_satellite_data_handled_in_report():
    """Verify that reports for blocked satellite jobs never claim 0 flood pixels or that no flooding occurred."""
    report_service = get_report_service()
    job_id = "job_test_blocked_report_case"
    from unittest.mock import MagicMock, patch
    from app.schemas.results import SectionStatus

    mock_structured = MagicMock()
    mock_structured.overall_status = "satellite_binary_data_blocked"
    mock_structured.source_data_status = "satellite_binary_data_blocked"
    mock_structured.version = "1.0.0"
    mock_structured.job_id = job_id
    mock_structured.metadata = MagicMock(job_id=job_id, flood_date="2026-08-26", report_language="en")
    mock_structured.satellite = MagicMock(status=SectionStatus.BLOCKED, data_availability="blocked")
    mock_structured.segmentation = MagicMock(status=SectionStatus.BLOCKED, flood_area_km2=None)
    mock_structured.geospatial = MagicMock(status=SectionStatus.BLOCKED, affected_buildings=None)
    mock_structured.connectivity = MagicMock(status=SectionStatus.BLOCKED, blocked_road_count=None)
    mock_structured.limitations = []
    mock_structured.model_dump = lambda **kwargs: {}

    with patch.object(report_service.results_service, "get_analysis_result", return_value=mock_structured):
        with patch.object(report_service.results_service, "build_analysis_result", return_value=mock_structured):
            resp_en = report_service.generate_report(job_id, ReportGenerateRequest(language="en", force_fallback=True))
            rep_en = resp_en.report
            sec_extent = next(s for s in rep_en.sections if s.section_id == "flood_extent")
            assert sec_extent.key_metrics["flood_area_km2"] is None
            assert "BLOCKED" in sec_extent.content
            assert "unmeasured" in sec_extent.content.lower()
            assert "0 km²" not in sec_extent.content

            resp_ne = report_service.generate_report(job_id, ReportGenerateRequest(language="ne", force_fallback=True))
            rep_ne = resp_ne.report
            sec_ne_extent = next(s for s in rep_ne.sections if s.section_id == "flood_extent")
            assert sec_ne_extent.key_metrics["flood_area_km2"] is None
            assert "अवरुद्ध" in sec_ne_extent.content


def test_unsupported_language_raises_error(mock_module5_geospatial_job):
    """Verify requesting an unsupported language code raises ValueError."""
    job_id, _ = mock_module5_geospatial_job
    report_service = get_report_service()

    with pytest.raises(ValueError) as excinfo:
        report_service.generate_report(job_id, ReportGenerateRequest(language="de"))
    assert "not supported" in str(excinfo.value).lower()


def test_report_api_endpoints(mock_module5_geospatial_job):
    """Verify POST and GET /analysis/{job_id}/report endpoints in both English and Nepali."""
    job_id, osm_path = mock_module5_geospatial_job

    # 1. POST /analysis/{job_id}/report (English)
    post_en = client.post(f"/analysis/{job_id}/report", json={"language": "en"})
    assert post_en.status_code == 200
    data_en = post_en.json()
    assert data_en["job_id"] == job_id
    assert data_en["language"] == "en"
    assert len(data_en["report"]["sections"]) == 11

    # 2. POST /analysis/{job_id}/report (Nepali)
    post_ne = client.post(f"/analysis/{job_id}/report", json={"language": "ne"})
    assert post_ne.status_code == 200
    data_ne = post_ne.json()
    assert data_ne["language"] == "ne"

    # 3. GET /analysis/{job_id}/report
    get_res = client.get(f"/analysis/{job_id}/report?language=en")
    assert get_res.status_code == 200
    assert get_res.json()["language"] == "en"

    # 4. GET /analysis/{job_id}/report/{language}
    get_lang = client.get(f"/analysis/{job_id}/report/ne")
    assert get_lang.status_code == 200
    assert get_lang.json()["language"] == "ne"

    # 5. Unsupported language gives 400
    bad_res = client.get(f"/analysis/{job_id}/report/fr")
    assert bad_res.status_code == 400


def test_report_generator_interface_behavior(mock_module5_geospatial_job):
    """Verify report_generator(None) raises NotImplementedError and dict invocation succeeds."""
    with pytest.raises(NotImplementedError):
        report_generator(None, "en")

    job_id, _ = mock_module5_geospatial_job
    res = report_generator({"job_id": job_id}, "en")
    assert isinstance(res, dict)
    assert res["job_id"] == job_id
    assert "report" in res
