import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.models.analysis import AnalysisStatus
from app.schemas.pipeline import PipelineRunRequest, PipelineStage
from app.services.pipeline_service import get_pipeline_service
from app.services.product_downloader import get_product_downloader

client = TestClient(app)


def test_pipeline_run_sync_blocked_cdse_case():
    """
    Test full end-to-end pipeline execution for the 2026 Nepal case study
    when CDSE binary download credentials are unavailable.
    """
    pipeline = get_pipeline_service()
    downloader = get_product_downloader()

    req = PipelineRunRequest(
        flood_date="2026-08-26",
        report_language="en",
        demo_mode=False,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("app.services.product_downloader.ProductDownloaderService.is_auth_configured", lambda self: False)
        state = pipeline.execute_pipeline(req)

    assert state.job_id is not None
    assert state.status == "completed_with_limitations"
    assert state.current_stage == PipelineStage.COMPLETED_WITH_LIMITATIONS
    assert state.is_blocked is True
    assert "CDSE binary satellite download credentials unavailable" in (state.blocker or "")
    assert state.summary is not None
    # Must preserve None/null for flood area, never zero
    assert state.summary["flood_area_km2"] is None

    # Check stage progression history
    stage_names = [e.stage for e in state.stages_history]
    assert PipelineStage.VALIDATING in stage_names
    assert PipelineStage.DISCOVERING_SATELLITES in stage_names
    assert PipelineStage.SELECTING_SCENES in stage_names
    assert PipelineStage.PREPARING_ARD in stage_names
    assert PipelineStage.RUNNING_SEGMENTATION in stage_names
    assert PipelineStage.RUNNING_GEOSPATIAL in stage_names
    assert PipelineStage.RUNNING_CONNECTIVITY in stage_names
    assert PipelineStage.BUILDING_RESULTS in stage_names
    assert PipelineStage.GENERATING_REPORT in stage_names


def test_pipeline_run_sync_demo_mode():
    """
    Test pipeline execution in DEMO MODE with synthetic/pre-staged layers.
    """
    pipeline = get_pipeline_service()
    req = PipelineRunRequest(
        flood_date="2026-08-26",
        report_language="en",
        demo_mode=True,
    )

    state = pipeline.execute_pipeline(req)
    assert state.job_id is not None
    assert state.progress_percent == 100
    assert len(state.stages_history) >= 9


def test_pipeline_date_normalization():
    """Verify robust date parsing for both DD-MM-YYYY and YYYY-MM-DD."""
    pipeline = get_pipeline_service()
    assert pipeline._normalize_date_str("26-08-2026") == "2026-08-26"
    assert pipeline._normalize_date_str("2026-08-26") == "2026-08-26"
    assert pipeline._normalize_date_str("01-09-2026") == "2026-09-01"


def test_pipeline_validation_error_invalid_coordinates():
    """Verify invalid polygon coordinates gracefully transition to FAILED stage."""
    pipeline = get_pipeline_service()
    bad_req = PipelineRunRequest(
        aoi={"type": "Polygon", "coordinates": [[[10, 10], [10, 20]]]}, # invalid polygon < 4 coords
        flood_date="2026-08-26",
    )
    state = pipeline.execute_pipeline(bad_req)
    assert state.status == "failed"
    assert state.current_stage == PipelineStage.FAILED
    assert state.error_message is not None


def test_api_analysis_run_async_and_status_polling():
    """Test POST /analysis/run and subsequent GET /analysis/{job_id}/status polling."""
    payload = {
        "flood_date": "2026-08-26",
        "report_language": "en",
        "demo_mode": False,
    }
    with patch("app.services.product_downloader.ProductDownloaderService.is_auth_configured", return_value=False):
        resp = client.post("/analysis/run", json=payload)
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    job_id = data["job_id"]
    assert data["current_stage"] in ["created", "completed_with_limitations", "blocked", "completed"]

    # Poll status endpoint
    poll_resp = client.get(f"/analysis/{job_id}/status")
    assert poll_resp.status_code == 200
    poll_data = poll_resp.json()
    assert poll_data["job_id"] == job_id


def test_api_analysis_run_sync():
    """Test POST /analysis/run?sync=true synchronous execution."""
    payload = {
        "flood_date": "2026-08-26",
        "report_language": "ne",
        "demo_mode": True,
    }
    resp = client.post("/analysis/run?sync=true", json=payload)
    assert resp.status_code == 202
    data = resp.json()
    assert data["job_id"] is not None
    assert data["progress_percent"] == 100
    assert data["report_language"] == "ne"


def test_api_get_analysis_layers():
    """Test GeoJSON map layer retrieval endpoints."""
    # Use reference Nepal job which already has populated GeoJSON outputs
    job_id = "job_nepal_2026_case"
    
    # Layer: aoi
    aoi_resp = client.get(f"/analysis/{job_id}/layers/aoi")
    assert aoi_resp.status_code == 200
    aoi_data = aoi_resp.json()
    assert aoi_data["type"] == "FeatureCollection"

    # Layer: roads
    roads_resp = client.get(f"/analysis/{job_id}/layers/roads")
    assert roads_resp.status_code == 200
    roads_data = roads_resp.json()
    assert roads_data["type"] == "FeatureCollection"

    # Layer: buildings
    bldgs_resp = client.get(f"/analysis/{job_id}/layers/buildings")
    assert bldgs_resp.status_code == 200
    assert bldgs_resp.json()["type"] == "FeatureCollection"


def test_dashboard_html_view_and_root_content_negotiation():
    """Verify /dashboard returns HTML and / handles content negotiation."""
    # Direct /dashboard
    dash_resp = client.get("/dashboard")
    assert dash_resp.status_code == 200
    assert "FLOOD DAMAGE MAPPER" in dash_resp.text
    assert "<!DOCTYPE html>" in dash_resp.text

    # / with HTML accept header
    browser_resp = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert browser_resp.status_code == 200
    assert "FLOOD DAMAGE MAPPER" in browser_resp.text

    # / with JSON accept header (API client)
    api_resp = client.get("/", headers={"Accept": "application/json"})
    assert api_resp.status_code == 200
    assert "endpoints" in api_resp.json()
    assert "dashboard_url" in api_resp.json()


def test_emsr927_remains_validation_only_in_pipeline():
    """Verify EMSR927 is never used as input in pipeline results or reports."""
    job_id = "job_nepal_2026_case"
    
    # Check structured results
    res_resp = client.get(f"/analysis/{job_id}/results")
    assert res_resp.status_code == 200
    res_data = res_resp.json()
    
    prov = res_data.get("provenance", {})
    assert prov.get("validation_sources") is not None
    assert "EMSR927" in prov["validation_sources"][0]
    assert "validation only" in prov["validation_sources"][0].lower() or "not an input" in prov["validation_sources"][0].lower()

    # Check report
    rep_resp = client.get(f"/analysis/{job_id}/report/en")
    assert rep_resp.status_code == 200
    rep_data = rep_resp.json()
    val_secs = [s for s in rep_data["report"]["sections"] if "Validation" in s["title"]]
    assert len(val_secs) > 0
    assert "EMSR927" in val_secs[0]["content"]
    assert "NOT used as an" in val_secs[0]["content"]


def test_custom_aoi_parameter_propagation_and_isolation():
    """
    Verify that custom AOIs and dates:
    1. Are correctly received by the pipeline.
    2. Register separate jobs with their own IDs.
    3. Retain the submitted AOI coordinates in job record and /layers/aoi.
    4. Do not reuse stale/static results from the Nepal reference case.
    """
    custom_aoi = {
        "type": "Polygon",
        "coordinates": [
            [
                [86.95, 26.50],
                [87.25, 26.50],
                [87.25, 26.85],
                [86.95, 26.85],
                [86.95, 26.50],
            ]
        ],
    }
    req_payload = {
        "aoi": custom_aoi,
        "flood_date": "2026-08-15",
        "comparison_date": "2026-08-22",
        "report_language": "ne",
        "demo_mode": True,
        "pre_event_window_days": 10,
        "post_event_window_days": 10,
    }

    # Run custom pipeline synchronously
    resp = client.post("/analysis/run?sync=true", json=req_payload)
    assert resp.status_code in [200, 202]
    data = resp.json()

    job_id = data["job_id"]
    assert job_id.startswith("job_")
    assert job_id != "job_nepal_2026_case"
    assert data["flood_date"] == "2026-08-15"
    assert data["report_language"] == "ne"

    # Query AOI layer for this custom job
    aoi_resp = client.get(f"/analysis/{job_id}/layers/aoi")
    assert aoi_resp.status_code == 200
    aoi_data = aoi_resp.json()
    assert aoi_data["type"] == "FeatureCollection"
    assert len(aoi_data["features"]) == 1
    # Check that coordinates match the custom AOI, NOT default Nepal
    coords = aoi_data["features"][0]["geometry"]["coordinates"][0]
    assert coords[0][0] == 86.95
    assert coords[0][1] == 26.50

    # Verify that available layers include aoi
    assert "aoi" in data["available_layers"]
