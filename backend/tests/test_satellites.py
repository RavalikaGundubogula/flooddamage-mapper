import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.clients.copernicus import CopernicusCatalogClient, geojson_to_wkt_polygon
from app.schemas.satellite import SceneMetadata
from app.services.satellite_cache import get_satellite_cache

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear discovery cache before each test."""
    cache = get_satellite_cache()
    cache.clear()
    yield
    cache.clear()


def test_satellite_status_endpoint():
    """Verify GET /satellites/status returns operational status and supported satellites."""
    with patch.object(CopernicusCatalogClient, "health_check", return_value=True):
        response = client.get("/satellites/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert "SENTINEL-1" in data["supported_satellites"]
        assert "SENTINEL-2" in data["supported_satellites"]
        assert data["catalog_name"] == "Copernicus Data Space Ecosystem (CDSE)"


def test_geojson_to_wkt_polygon():
    """Verify GeoJSON polygon is correctly transformed into WKT string."""
    geojson = {
        "type": "Polygon",
        "coordinates": [
            [
                [85.2, 27.8],
                [85.5, 27.8],
                [85.5, 28.1],
                [85.2, 28.1],
                [85.2, 27.8],
            ]
        ],
    }
    wkt = geojson_to_wkt_polygon(geojson)
    assert wkt == "POLYGON((85.2 27.8, 85.5 27.8, 85.5 28.1, 85.2 28.1, 85.2 27.8))"


def test_geojson_to_wkt_invalid():
    """Verify error on invalid geometry."""
    with pytest.raises(ValueError):
        geojson_to_wkt_polygon({"type": "Point", "coordinates": [85.2, 27.8]})


def test_scene_metadata_normalization():
    """Verify SceneMetadata schema handles both radar and optical fields."""
    item = SceneMetadata(
        product_id="S1A_IW_GRDH_1SDV_20240831T122235_TEST",
        collection="SENTINEL-1",
        platform="S1A",
        acquisition_datetime="2024-08-31T12:22:35Z",
        orbit_direction="ASCENDING",
        relative_orbit=85,
        polarization="VV&VH",
        product_type="IW_GRDH_1S",
    )
    assert item.platform == "S1A"
    assert item.cloud_cover is None
    assert item.relative_orbit == 85
    assert item.polarization == "VV&VH"


def test_sentinel_1_discovery_get_endpoint():
    """Verify GET /satellites/sentinel-1 returns candidates with timing and SAR metadata."""
    mock_scenes = [
        SceneMetadata(
            product_id="S1A_IW_GRDH_1SDV_20240819T122235_PRE",
            collection="SENTINEL-1",
            platform="S1A",
            acquisition_datetime=datetime(2024, 8, 19, 12, 22, 35, tzinfo=timezone.utc),
            orbit_direction="ASCENDING",
            relative_orbit=85,
            polarization="VV&VH",
            product_type="IW_GRDH_1S",
        ),
        SceneMetadata(
            product_id="S1A_IW_GRDH_1SDV_20240831T122235_POST",
            collection="SENTINEL-1",
            platform="S1A",
            acquisition_datetime=datetime(2024, 8, 31, 12, 22, 35, tzinfo=timezone.utc),
            orbit_direction="ASCENDING",
            relative_orbit=85,
            polarization="VV&VH",
            product_type="IW_GRDH_1S",
        ),
    ]

    with patch.object(CopernicusCatalogClient, "search", return_value=mock_scenes):
        response = client.get("/satellites/sentinel-1?flood_date=2024-08-26")
        assert response.status_code == 200
        data = response.json()
        assert data["collection"] == "SENTINEL-1"
        assert data["total_candidates"] == 2
        assert data["pre_event_count"] == 1
        assert data["post_event_count"] == 1
        assert data["grd_candidate_count"] == 2

        candidates = data["candidates"]
        assert len(candidates) == 2
        assert candidates[0]["timing"] == "pre_event"
        assert candidates[0]["days_from_event"] < 0
        assert candidates[0]["relative_orbit"] == 85
        assert candidates[0]["orbit_direction"] == "ASCENDING"
        assert candidates[0]["polarization"] == "VV&VH"

        assert candidates[1]["timing"] == "post_event"
        assert candidates[1]["days_from_event"] > 0


def test_sentinel_1_discovery_post_endpoint():
    """Verify POST /satellites/sentinel-1 accepts custom AOI and returns candidates."""
    mock_scenes = [
        SceneMetadata(
            product_id="S1A_IW_GRDH_1SDV_20240831T122235_POST",
            collection="SENTINEL-1",
            platform="S1A",
            acquisition_datetime=datetime(2024, 8, 31, 12, 22, 35, tzinfo=timezone.utc),
            orbit_direction="ASCENDING",
            relative_orbit=85,
            polarization="VV&VH",
            product_type="IW_GRDH_1S",
        )
    ]

    with patch.object(CopernicusCatalogClient, "search", return_value=mock_scenes):
        payload = {
            "aoi": [85.2, 27.8, 85.5, 28.1],
            "flood_date": "2024-08-26",
            "days_before": 10,
            "days_after": 5,
        }
        response = client.post("/satellites/sentinel-1", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_candidates"] == 1
        assert data["candidates"][0]["is_grd"] is True


def test_sentinel_2_discovery_get_endpoint():
    """Verify GET /satellites/sentinel-2 evaluates cloud conditions and usability."""
    mock_scenes = [
        SceneMetadata(
            product_id="S2A_MSIL2A_20240815T050000_CLEAR",
            collection="SENTINEL-2",
            platform="S2A",
            acquisition_datetime=datetime(2024, 8, 15, 5, 0, 0, tzinfo=timezone.utc),
            cloud_cover=12.5,
            product_type="S2MSI2A",
        ),
        SceneMetadata(
            product_id="S2B_MSIL2A_20240820T050000_CLOUDY",
            collection="SENTINEL-2",
            platform="S2B",
            acquisition_datetime=datetime(2024, 8, 20, 5, 0, 0, tzinfo=timezone.utc),
            cloud_cover=88.0,
            product_type="S2MSI2A",
        ),
        SceneMetadata(
            product_id="S2A_MSIL2A_20240829T050000_MODERATE",
            collection="SENTINEL-2",
            platform="S2A",
            acquisition_datetime=datetime(2024, 8, 29, 5, 0, 0, tzinfo=timezone.utc),
            cloud_cover=35.0,
            product_type="S2MSI2A",
        ),
    ]

    with patch.object(CopernicusCatalogClient, "search", return_value=mock_scenes):
        response = client.get("/satellites/sentinel-2?flood_date=2024-08-26")
        assert response.status_code == 200
        data = response.json()
        assert data["collection"] == "SENTINEL-2"
        assert data["total_candidates"] == 3
        assert data["pre_event_count"] == 2
        assert data["post_event_count"] == 1
        assert data["usable_candidate_count"] == 2  # 12.5% and 35.0% are <= 50%
        assert data["average_cloud_cover"] == 45.17

        candidates = data["candidates"]
        assert len(candidates) == 3
        assert candidates[0]["optical_usability"] == "clear"
        assert candidates[0]["is_l2a"] is True
        assert candidates[1]["optical_usability"] == "cloudy"
        assert candidates[2]["optical_usability"] == "moderate"


def test_sentinel_2_discovery_post_endpoint():
    """Verify POST /satellites/sentinel-2 accepts custom AOI."""
    mock_scenes = [
        SceneMetadata(
            product_id="S2A_MSIL2A_20240829T050000_MODERATE",
            collection="SENTINEL-2",
            platform="S2A",
            acquisition_datetime=datetime(2024, 8, 29, 5, 0, 0, tzinfo=timezone.utc),
            cloud_cover=25.0,
            product_type="S2MSI2A",
        )
    ]

    with patch.object(CopernicusCatalogClient, "search", return_value=mock_scenes):
        payload = {
            "aoi": [85.2, 27.8, 85.5, 28.1],
            "flood_date": "2024-08-26",
        }
        response = client.post("/satellites/sentinel-2", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_candidates"] == 1
        assert data["candidates"][0]["optical_usability"] == "moderate"


# ============================================================================
# Smart Satellite Router Tests (4 Required Strategy Scenarios)
# ============================================================================

def test_router_scenario_1_good_s1_poor_s2():
    """Scenario 1: Good Sentinel-1 + Poor Sentinel-2 -> Strategy: 'sentinel-1'."""
    s1_scenes = [
        SceneMetadata(
            product_id="S1A_IW_GRDH_1SDV_20240819_PRE",
            collection="SENTINEL-1",
            acquisition_datetime=datetime(2024, 8, 19, 12, 0, 0, tzinfo=timezone.utc),
            orbit_direction="ASCENDING",
            relative_orbit=85,
            product_type="IW_GRDH_1S",
        ),
        SceneMetadata(
            product_id="S1A_IW_GRDH_1SDV_20240831_POST",
            collection="SENTINEL-1",
            acquisition_datetime=datetime(2024, 8, 31, 12, 0, 0, tzinfo=timezone.utc),
            orbit_direction="ASCENDING",
            relative_orbit=85,
            product_type="IW_GRDH_1S",
        ),
    ]
    s2_scenes = [
        SceneMetadata(
            product_id="S2A_MSIL2A_20240829_CLOUDY",
            collection="SENTINEL-2",
            acquisition_datetime=datetime(2024, 8, 29, 5, 0, 0, tzinfo=timezone.utc),
            cloud_cover=92.0,  # > 50% unsuitable
            product_type="S2MSI2A",
        )
    ]

    def mock_search_dispatch(**kwargs):
        if kwargs.get("collection") == "SENTINEL-1":
            return s1_scenes
        return s2_scenes

    with patch.object(CopernicusCatalogClient, "search", side_effect=mock_search_dispatch):
        response = client.get("/satellites/route?flood_date=2024-08-26")
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == "sentinel-1"
        assert "radar" in data["reason"].lower()
        assert data["sentinel_1"]["is_suitable"] is True
        assert data["sentinel_2"]["is_suitable"] is False


def test_router_scenario_2_good_s1_good_s2():
    """Scenario 2: Good Sentinel-1 + Useful Sentinel-2 -> Strategy: 'sentinel-1-and-sentinel-2'."""
    s1_scenes = [
        SceneMetadata(
            product_id="S1A_IW_GRDH_1SDV_20240819_PRE",
            collection="SENTINEL-1",
            acquisition_datetime=datetime(2024, 8, 19, 12, 0, 0, tzinfo=timezone.utc),
            orbit_direction="ASCENDING",
            relative_orbit=85,
            product_type="IW_GRDH_1S",
        ),
        SceneMetadata(
            product_id="S1A_IW_GRDH_1SDV_20240831_POST",
            collection="SENTINEL-1",
            acquisition_datetime=datetime(2024, 8, 31, 12, 0, 0, tzinfo=timezone.utc),
            orbit_direction="ASCENDING",
            relative_orbit=85,
            product_type="IW_GRDH_1S",
        ),
    ]
    s2_scenes = [
        SceneMetadata(
            product_id="S2A_MSIL2A_20240828_CLEAR",
            collection="SENTINEL-2",
            acquisition_datetime=datetime(2024, 8, 28, 5, 0, 0, tzinfo=timezone.utc),
            cloud_cover=14.0,  # <= 25% clear
            product_type="S2MSI2A",
        )
    ]

    def mock_search_dispatch(**kwargs):
        if kwargs.get("collection") == "SENTINEL-1":
            return s1_scenes
        return s2_scenes

    with patch.object(CopernicusCatalogClient, "search", side_effect=mock_search_dispatch):
        response = client.get("/satellites/route?flood_date=2024-08-26")
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == "sentinel-1-and-sentinel-2"
        assert data["sentinel_1"]["is_suitable"] is True
        assert data["sentinel_2"]["is_suitable"] is True


def test_router_scenario_3_s1_unavailable_good_s2():
    """Scenario 3: Sentinel-1 unavailable + Useful Sentinel-2 -> Strategy: 'sentinel-2'."""
    s1_scenes = []  # No S1
    s2_scenes = [
        SceneMetadata(
            product_id="S2A_MSIL2A_20240828_CLEAR",
            collection="SENTINEL-2",
            acquisition_datetime=datetime(2024, 8, 28, 5, 0, 0, tzinfo=timezone.utc),
            cloud_cover=18.0,
            product_type="S2MSI2A",
        )
    ]

    def mock_search_dispatch(**kwargs):
        if kwargs.get("collection") == "SENTINEL-1":
            return s1_scenes
        return s2_scenes

    with patch.object(CopernicusCatalogClient, "search", side_effect=mock_search_dispatch):
        response = client.get("/satellites/route?flood_date=2024-08-26")
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == "sentinel-2"
        assert data["sentinel_1"]["is_suitable"] is False
        assert data["sentinel_2"]["is_suitable"] is True


def test_router_scenario_4_neither_suitable():
    """Scenario 4: Neither suitable -> Strategy: 'no-suitable-data'."""
    s1_scenes = []
    s2_scenes = [
        SceneMetadata(
            product_id="S2A_MSIL2A_20240828_CLOUDY",
            collection="SENTINEL-2",
            acquisition_datetime=datetime(2024, 8, 28, 5, 0, 0, tzinfo=timezone.utc),
            cloud_cover=98.0,
            product_type="S2MSI2A",
        )
    ]

    def mock_search_dispatch(**kwargs):
        if kwargs.get("collection") == "SENTINEL-1":
            return s1_scenes
        return s2_scenes

    with patch.object(CopernicusCatalogClient, "search", side_effect=mock_search_dispatch):
        response = client.get("/satellites/route?flood_date=2024-08-26")
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == "no-suitable-data"
        assert data["sentinel_1"]["is_suitable"] is False
        assert data["sentinel_2"]["is_suitable"] is False


def test_router_post_endpoint_custom_aoi():
    """Verify POST /satellites/route accepts custom AOI."""
    with patch.object(CopernicusCatalogClient, "search", return_value=[]):
        payload = {
            "aoi": [85.2, 27.8, 85.5, 28.1],
            "flood_date": "2024-08-26",
        }
        response = client.post("/satellites/route", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == "no-suitable-data"


# ============================================================================
# Cache & Error Handling Tests (Milestone 5)
# ============================================================================

def test_cache_hit_and_miss(tmp_path):
    """Verify cache miss queries the catalog and cache hit avoids redundant catalog calls."""
    from app.services.satellite_cache import SatelliteDiscoveryCache
    from app.services.satellite_service import SatelliteDiscoveryService

    mock_catalog = MagicMock()
    mock_catalog.search.return_value = [
        SceneMetadata(
            product_id="S1A_TEST_CACHE",
            collection="SENTINEL-1",
            acquisition_datetime=datetime(2024, 8, 20, 12, 0, 0, tzinfo=timezone.utc),
            product_type="IW_GRDH_1S",
        )
    ]

    temp_cache = SatelliteDiscoveryCache(cache_dir=tmp_path)
    service = SatelliteDiscoveryService(catalog_client=mock_catalog, cache=temp_cache)

    # 1. First call -> Cache MISS (should call catalog.search)
    res1 = service.discover_sentinel_1(flood_date="2024-08-26", max_results=5)
    assert res1.total_candidates == 1
    assert mock_catalog.search.call_count == 1

    # 2. Second call -> Cache HIT (should NOT call catalog.search again)
    res2 = service.discover_sentinel_1(flood_date="2024-08-26", max_results=5)
    assert res2.total_candidates == 1
    assert res2.candidates[0].scene.product_id == "S1A_TEST_CACHE"
    assert mock_catalog.search.call_count == 1  # Unchanged!


def test_cache_stats_and_clear_endpoint():
    """Verify /satellites/cache/stats and /satellites/cache/clear endpoints."""
    stats_res = client.get("/satellites/cache/stats")
    assert stats_res.status_code == 200
    assert "cached_queries_count" in stats_res.json()

    clear_res = client.post("/satellites/cache/clear")
    assert clear_res.status_code == 200
    assert clear_res.json()["status"] == "cleared"


def test_catalog_failure_handling():
    """Verify graceful handling when catalog raises network error or timeout."""
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        catalog = CopernicusCatalogClient()
        assert catalog.health_check() is False
        results = catalog.search(
            aoi={"type": "Polygon", "coordinates": [[[85.2, 27.8], [85.5, 27.8], [85.5, 28.1], [85.2, 28.1], [85.2, 27.8]]]},
            start_date="2024-08-01",
            end_date="2024-08-15",
            collection="SENTINEL-1",
        )
        assert results == []


def test_malformed_catalog_response():
    """Verify catalog client handles unexpected / malformed JSON payloads gracefully."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"value": [{"Name": "MALFORMED_NO_ATTRIBUTES"}]}'
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_cm):
        catalog = CopernicusCatalogClient()
        results = catalog.search(
            aoi={"type": "Polygon", "coordinates": [[[85.2, 27.8], [85.5, 27.8], [85.5, 28.1], [85.2, 28.1], [85.2, 27.8]]]},
            start_date="2024-08-01",
            end_date="2024-08-15",
            collection="SENTINEL-1",
        )
        assert len(results) == 1
        assert results[0].product_id == "MALFORMED_NO_ATTRIBUTES"
        assert results[0].cloud_cover is None


def test_no_results_found():
    """Verify catalog client handles zero products returned."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"value": []}'
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_cm):
        catalog = CopernicusCatalogClient()
        results = catalog.search(
            aoi={"type": "Polygon", "coordinates": [[[85.2, 27.8], [85.5, 27.8], [85.5, 28.1], [85.2, 28.1], [85.2, 27.8]]]},
            start_date="2024-08-01",
            end_date="2024-08-15",
            collection="SENTINEL-1",
        )
        assert results == []


# ============================================================================
# Scene Selection Tests (Module 3 Milestone 1)
# ============================================================================

def test_select_sentinel_1_matching_relative_orbit():
    """Verify S1 selection prioritizes matching relative orbit track over non-matching tracks."""
    from app.services.scene_selector import SceneSelectionService
    from app.schemas.satellite import Sentinel1Candidate

    selector = SceneSelectionService()

    # Pre-event: Orbit 121 (at -2 days) and Orbit 85 (at -4 days)
    pre_orbit_121 = Sentinel1Candidate(
        scene=SceneMetadata(
            product_id="S1A_IW_GRDH_1SDV_20240824_ORBIT121",
            collection="SENTINEL-1",
            acquisition_datetime=datetime(2024, 8, 24, 12, 0, 0, tzinfo=timezone.utc),
            orbit_direction="ASCENDING",
            relative_orbit=121,
            product_type="IW_GRDH_1S",
        ),
        timing="pre_event",
        days_from_event=-2.0,
        is_grd=True,
        relative_orbit=121,
        orbit_direction="ASCENDING",
    )
    pre_orbit_85 = Sentinel1Candidate(
        scene=SceneMetadata(
            product_id="S1A_IW_GRDH_1SDV_20240822_ORBIT85",
            collection="SENTINEL-1",
            acquisition_datetime=datetime(2024, 8, 22, 12, 0, 0, tzinfo=timezone.utc),
            orbit_direction="ASCENDING",
            relative_orbit=85,
            product_type="IW_GRDH_1S",
        ),
        timing="pre_event",
        days_from_event=-4.0,
        is_grd=True,
        relative_orbit=85,
        orbit_direction="ASCENDING",
    )

    # Post-event: Only Orbit 85 exists (at +3 days)
    post_orbit_85 = Sentinel1Candidate(
        scene=SceneMetadata(
            product_id="S1A_IW_GRDH_1SDV_20240829_ORBIT85",
            collection="SENTINEL-1",
            acquisition_datetime=datetime(2024, 8, 29, 12, 0, 0, tzinfo=timezone.utc),
            orbit_direction="ASCENDING",
            relative_orbit=85,
            product_type="IW_GRDH_1S",
        ),
        timing="post_event",
        days_from_event=3.0,
        is_grd=True,
        relative_orbit=85,
        orbit_direction="ASCENDING",
    )

    candidates = [pre_orbit_121, pre_orbit_85, post_orbit_85]
    pair, warnings = selector.select_sentinel_1_pair(candidates)

    assert pair is not None
    assert pair.orbit_compatibility is True
    assert pair.relative_orbit == 85
    # Must pick Orbit 85 for baseline, NOT orbit 121
    assert pair.before_scene.product_id == "S1A_IW_GRDH_1SDV_20240822_ORBIT85"
    assert pair.after_scene.product_id == "S1A_IW_GRDH_1SDV_20240829_ORBIT85"


def test_select_sentinel_2_lowest_cloud_and_l2a():
    """Verify S2 selection prioritizes lowest cloud cover and Level-2A surface reflectance."""
    from app.services.scene_selector import SceneSelectionService
    from app.schemas.satellite import Sentinel2Candidate

    selector = SceneSelectionService()

    pre_scene = Sentinel2Candidate(
        scene=SceneMetadata(
            product_id="S2A_MSIL2A_20240815_PRE",
            collection="SENTINEL-2",
            acquisition_datetime=datetime(2024, 8, 15, 5, 0, 0, tzinfo=timezone.utc),
            cloud_cover=10.0,
            product_type="S2MSI2A",
        ),
        timing="pre_event",
        days_from_event=-11.0,
        cloud_cover=10.0,
        is_l2a=True,
        optical_usability="clear",
    )

    post_cloudy = Sentinel2Candidate(
        scene=SceneMetadata(
            product_id="S2B_MSIL2A_20240828_CLOUDY",
            collection="SENTINEL-2",
            acquisition_datetime=datetime(2024, 8, 28, 5, 0, 0, tzinfo=timezone.utc),
            cloud_cover=85.0,  # Unusable
            product_type="S2MSI2A",
        ),
        timing="post_event",
        days_from_event=2.0,
        cloud_cover=85.0,
        is_l2a=True,
        optical_usability="cloudy",
    )

    post_clear = Sentinel2Candidate(
        scene=SceneMetadata(
            product_id="S2A_MSIL2A_20240830_CLEAR",
            collection="SENTINEL-2",
            acquisition_datetime=datetime(2024, 8, 30, 5, 0, 0, tzinfo=timezone.utc),
            cloud_cover=15.0,  # Clear
            product_type="S2MSI2A",
        ),
        timing="post_event",
        days_from_event=4.0,
        cloud_cover=15.0,
        is_l2a=True,
        optical_usability="clear",
    )

    candidates = [pre_scene, post_cloudy, post_clear]
    pair, warnings = selector.select_sentinel_2_pair(candidates)

    assert pair is not None
    assert pair.before_scene.product_id == "S2A_MSIL2A_20240815_PRE"
    assert pair.after_scene.product_id == "S2A_MSIL2A_20240830_CLEAR"
    assert pair.after_scene.cloud_cover == 15.0


def test_scene_selection_endpoint_success():
    """Verify GET /satellites/select returns selected status and pair metadata."""
    s1_scenes = [
        SceneMetadata(
            product_id="S1A_IW_GRDH_1SDV_20240819_PRE",
            collection="SENTINEL-1",
            acquisition_datetime=datetime(2024, 8, 19, 12, 0, 0, tzinfo=timezone.utc),
            orbit_direction="ASCENDING",
            relative_orbit=85,
            product_type="IW_GRDH_1S",
        ),
        SceneMetadata(
            product_id="S1A_IW_GRDH_1SDV_20240831_POST",
            collection="SENTINEL-1",
            acquisition_datetime=datetime(2024, 8, 31, 12, 0, 0, tzinfo=timezone.utc),
            orbit_direction="ASCENDING",
            relative_orbit=85,
            product_type="IW_GRDH_1S",
        ),
    ]

    with patch.object(CopernicusCatalogClient, "search", return_value=s1_scenes):
        response = client.get("/satellites/select?flood_date=2024-08-26")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "selected"
        assert data["sentinel_1_selection"] is not None
        assert data["sentinel_1_selection"]["orbit_compatibility"] is True
        assert data["sentinel_1_selection"]["relative_orbit"] == 85


def test_scene_selection_insufficient_data():
    """Verify GET /satellites/select returns insufficient_data when no valid pairs exist."""
    with patch.object(CopernicusCatalogClient, "search", return_value=[]):
        response = client.get("/satellites/select?flood_date=2026-08-26")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "insufficient_data"
        assert data["sentinel_1_selection"] is None
        assert data["sentinel_2_selection"] is None


# ============================================================================
# Product Download & Storage Tests (Module 3 Milestone 2)
# ============================================================================

def test_download_endpoint_auth_required():
    """Verify POST /satellites/download informs user when credentials are required."""
    payload = {
        "product_id": "S1A_IW_GRDH_1SDV_20240827T001717_TEST",
        "collection": "SENTINEL-1",
    }
    with patch("app.services.product_downloader.ProductDownloaderService.is_auth_configured", return_value=False):
        response = client.post("/satellites/download", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_requested"] == 1
        assert data["auth_required_count"] == 1
        assert data["auth_configured"] is False
        res = data["results"][0]
        assert res["status"] == "authentication_required"
        assert "credentials are required" in res["message"]


def test_download_single_product_status():
    """Verify GET /satellites/downloads/{product_id} returns status."""
    with patch("app.services.product_downloader.ProductDownloaderService.is_auth_configured", return_value=False):
        response = client.get("/satellites/downloads/S1A_IW_GRDH_1SDV_20240827_UNAUTH")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "authentication_required"
        assert data["product_id"] == "S1A_IW_GRDH_1SDV_20240827_UNAUTH"


def test_download_list_stored_products(tmp_path):
    """Verify GET /satellites/downloads lists local files in raw directories."""
    response = client.get("/satellites/downloads")
    assert response.status_code == 200
    data = response.json()
    assert "total_products" in data
    assert "sentinel1_count" in data
    assert "sentinel2_count" in data
    assert isinstance(data["products"], list)


def test_download_cached_product(tmp_path):
    """Verify cached products are recognized immediately without re-downloading."""
    from app.services.product_downloader import get_product_downloader
    downloader = get_product_downloader()
    
    # Create fake cached product
    raw_s1 = downloader.settings.get_data_path(f"{downloader.settings.DATA_RAW_DIR}/sentinel1")
    raw_s1.mkdir(parents=True, exist_ok=True)
    test_file = raw_s1 / "S1A_IW_GRDH_TEST_CACHED.zip"
    test_file.write_bytes(b"dummy satellite raster payload")
    
    try:
        response = client.post(
            "/satellites/download",
            json={"product_id": "S1A_IW_GRDH_TEST_CACHED", "collection": "SENTINEL-1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cached_count"] == 1
        res = data["results"][0]
        assert res["status"] == "cached"
        assert res["file_size_bytes"] == len(b"dummy satellite raster payload")
    finally:
        if test_file.exists():
            test_file.unlink()


def test_download_mocked_success():
    """Verify successful streaming download when credentials and stream succeed."""
    from app.services.product_downloader import ProductDownloaderService
    import io
    
    mock_catalog = MagicMock()
    mock_catalog.get_product.return_value = SceneMetadata(
        product_id="S1A_IW_GRDH_1SDV_20240827_STREAM",
        collection="SENTINEL-1",
        acquisition_datetime=datetime(2024, 8, 27, tzinfo=timezone.utc),
        product_type="IW_GRDH_1S",
        extra_attributes={"cdse_id": "test-uuid-1234", "content_length": 100},
    )

    downloader = ProductDownloaderService(catalog_client=mock_catalog)
    downloader.is_auth_configured = MagicMock(return_value=True)
    downloader.obtain_access_token = MagicMock(return_value="mock-token-abc")

    # Mock urllib urlopen stream
    mock_resp = MagicMock()
    mock_resp.read.side_effect = [b"streamed_satellite_bytes_123", b""]
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    # Ensure clean slate before test
    target_clean = downloader._clean_product_name("S1A_IW_GRDH_1SDV_20240827_STREAM")
    target_dir = downloader._resolve_storage_directory("SENTINEL-1")
    for f in [target_dir / f"{target_clean}.zip", target_dir / f"{target_clean}.meta.json"]:
        if f.exists():
            f.unlink()

    res = None
    try:
        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = downloader.download_product("S1A_IW_GRDH_1SDV_20240827_STREAM", "SENTINEL-1")
            assert res.status == "completed"
            assert res.file_size_bytes == len(b"streamed_satellite_bytes_123")
            assert res.checksum_sha256 is not None
            assert res.cdse_id == "test-uuid-1234"
    finally:
        # Cleanup created files
        if res and res.file_path and Path(res.file_path).exists():
            Path(res.file_path).unlink()
        meta = Path(res.file_path).with_suffix(".meta.json") if (res and res.file_path) else None
        if meta and meta.exists():
            meta.unlink()


def test_analysis_service_download_images_integration():
    """Verify download_images in analysis_service handles selection pairs."""
    from app.services.analysis_service import download_images
    selection = {
        "sentinel_1_selection": {
            "satellite": "SENTINEL-1",
            "before_scene": {
                "product_id": "S1A_IW_GRDH_PRE",
                "collection": "SENTINEL-1",
                "acquisition_datetime": "2024-08-19T00:00:00Z",
                "product_type": "IW_GRDH_1S",
            },
            "after_scene": {
                "product_id": "S1A_IW_GRDH_POST",
                "collection": "SENTINEL-1",
                "acquisition_datetime": "2024-08-31T00:00:00Z",
                "product_type": "IW_GRDH_1S",
            },
            "orbit_compatibility": True,
            "temporal_separation_days": 12.0,
            "before_days_from_event": -7.0,
            "after_days_from_event": 5.0,
            "quality_notes": "Compatible pair",
        }
    }
    with patch("app.services.product_downloader.ProductDownloaderService.is_auth_configured", return_value=False):
        result = download_images(selection)
        assert result["total_requested"] == 2
        assert result["auth_required_count"] == 2
        assert len(result["results"]) == 2


# ============================================================================
# AOI Clipping & Spatial Preparation Tests (Module 3 Milestone 3)
# ============================================================================

def test_calculate_clip_specification():
    """Verify SpatialClippingService calculates correct dimensions and Affine transform."""
    from app.services.spatial_clipper import get_spatial_clipper
    clipper = get_spatial_clipper()

    aoi = {
        "type": "Polygon",
        "coordinates": [
            [
                [85.2, 27.8],
                [85.5, 27.8],
                [85.5, 28.1],
                [85.2, 28.1],
                [85.2, 27.8],
            ]
        ],
    }
    spec = clipper.calculate_clip_specification(aoi, resolution_meters=10.0)
    assert spec.target_crs == "EPSG:4326"
    assert spec.dimensions[0] > 0
    assert spec.dimensions[1] > 0
    assert len(spec.affine_transform) == 6
    assert spec.bounding_box == [85.2, 27.8, 85.5, 28.1]
    assert spec.aoi_area_sqkm > 0


def test_get_clip_specification_endpoint():
    """Verify GET /satellites/clip/specification returns valid spatial specifications."""
    response = client.get("/satellites/clip/specification?min_lon=85.0&min_lat=27.5&max_lon=85.2&max_lat=27.7&resolution_meters=10")
    assert response.status_code == 200
    data = response.json()
    assert data["target_crs"] == "EPSG:4326"
    assert data["dimensions"][0] > 0
    assert data["dimensions"][1] > 0
    assert data["bounding_box"] == [85.0, 27.5, 85.2, 27.7]
    assert "affine_transform" in data


def test_post_clip_pair_endpoint():
    """Verify POST /satellites/clip creates identical co-registered grids for before and after scenes."""
    pair_payload = {
        "satellite": "SENTINEL-1",
        "before_scene": {
            "product_id": "S1A_IW_GRDH_PRE",
            "collection": "SENTINEL-1",
            "acquisition_datetime": "2024-08-19T00:00:00Z",
            "product_type": "IW_GRDH_1S",
        },
        "after_scene": {
            "product_id": "S1A_IW_GRDH_POST",
            "collection": "SENTINEL-1",
            "acquisition_datetime": "2024-08-31T00:00:00Z",
            "product_type": "IW_GRDH_1S",
        },
        "orbit_compatibility": True,
        "temporal_separation_days": 12.0,
        "before_days_from_event": -7.0,
        "after_days_from_event": 5.0,
        "quality_notes": "Identical track 19",
    }
    
    response = client.post(
        "/satellites/clip",
        json={"scene_pair": pair_payload, "resolution_meters": 10.0},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["identical_grid_alignment"] is True
    assert data["before_scene_clip"]["clip_specification"]["dimensions"] == data["after_scene_clip"]["clip_specification"]["dimensions"]
    assert len(data["before_scene_clip"]["bands"]) == 2  # VV & VH
    assert data["grid_shape"] == data["before_scene_clip"]["clip_specification"]["dimensions"]


def test_clip_raster_file_with_synthetic_geotiff(tmp_path):
    """Verify clip_raster_file cuts real GeoTIFF and computes band statistics."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    from app.services.spatial_clipper import get_spatial_clipper

    clipper = get_spatial_clipper()
    
    # 1. Create a synthetic GeoTIFF covering a larger bbox [85.0, 27.0, 86.0, 28.0]
    large_bounds = (85.0, 27.0, 86.0, 28.0)
    transform = from_bounds(*large_bounds, width=100, height=100)
    data = np.arange(10000, dtype=np.float32).reshape((1, 100, 100))

    src_tif = tmp_path / "synthetic_scene_vv.tif"
    with rasterio.open(
        src_tif,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    # 2. Clip to smaller sub-polygon [85.2, 27.2, 85.5, 27.5]
    sub_aoi = {
        "type": "Polygon",
        "coordinates": [
            [
                [85.2, 27.2],
                [85.5, 27.2],
                [85.5, 27.5],
                [85.2, 27.5],
                [85.2, 27.2],
            ]
        ],
    }

    out_tif = tmp_path / "clipped_vv.tif"
    meta = clipper.clip_raster_file(src_tif, sub_aoi, out_tif, band_name="VV")

    assert out_tif.exists()
    assert meta.band_name == "VV"
    assert meta.width < 100
    assert meta.height < 100
    assert meta.statistics is not None
    assert meta.statistics["min"] >= 0
    assert meta.statistics["max"] > 0


def test_clip_images_analysis_service_integration():
    """Verify clip_images in analysis_service handles scene pair selection."""
    from app.services.analysis_service import clip_images
    selection = {
        "sentinel_1_selection": {
            "satellite": "SENTINEL-1",
            "before_scene": {
                "product_id": "S1A_IW_GRDH_PRE",
                "collection": "SENTINEL-1",
                "acquisition_datetime": "2024-08-19T00:00:00Z",
                "product_type": "IW_GRDH_1S",
            },
            "after_scene": {
                "product_id": "S1A_IW_GRDH_POST",
                "collection": "SENTINEL-1",
                "acquisition_datetime": "2024-08-31T00:00:00Z",
                "product_type": "IW_GRDH_1S",
            },
            "orbit_compatibility": True,
            "temporal_separation_days": 12.0,
            "before_days_from_event": -7.0,
            "after_days_from_event": 5.0,
            "quality_notes": "Nepal track",
        }
    }
    aoi = {
        "type": "Polygon",
        "coordinates": [
            [[85.2, 27.8], [85.4, 27.8], [85.4, 28.0], [85.2, 28.0], [85.2, 27.8]]
        ],
    }
    result = clip_images(selection, aoi=aoi)
    assert result["identical_grid_alignment"] is True
    assert result["status"] == "specification_prepared"
    assert len(result["grid_shape"]) == 2


# ============================================================================
# SAR Preprocessing & Radiometric Calibration Tests (Module 3 Milestone 4)
# ============================================================================

def test_calibrate_dn_to_db():
    """Verify conversion of linear DN to decibels in valid SAR backscatter range."""
    import numpy as np
    from app.services.sar_preprocessor import get_sar_preprocessor
    preprocessor = get_sar_preprocessor()

    # Raw digital numbers (e.g. 50 to 500)
    raw_dn = np.array([[50.0, 100.0], [200.0, 500.0]], dtype=np.float32)
    db = preprocessor.calibrate_dn_to_db(raw_dn)
    assert db.shape == (2, 2)
    assert np.all(db >= -35.0)
    assert np.all(db <= 5.0)
    # Higher DN should result in higher dB (less negative)
    assert db[1, 1] > db[0, 0]


def test_apply_speckle_filter():
    """Verify 2D boxcar smoothing suppresses noise while maintaining dimensions."""
    import numpy as np
    from app.services.sar_preprocessor import get_sar_preprocessor
    preprocessor = get_sar_preprocessor()

    arr = np.zeros((10, 10), dtype=np.float32)
    arr[5, 5] = 100.0  # isolated speckle spike
    filtered = preprocessor.apply_speckle_filter(arr, window=3)
    assert filtered.shape == (10, 10)
    assert filtered[5, 5] < 100.0  # spike was smoothed
    assert filtered[5, 5] > 0.0    # energy distributed to window


def test_compute_temporal_difference():
    """Verify calculation of backscatter change and water drop identification."""
    import numpy as np
    from app.services.sar_preprocessor import get_sar_preprocessor
    preprocessor = get_sar_preprocessor()

    # Baseline land (-10 dB), flooded area drops to -22 dB
    pre = np.full((10, 10), -10.0, dtype=np.float32)
    post = np.full((10, 10), -10.0, dtype=np.float32)
    post[2:5, 2:5] = -22.0  # 9 pixels of new water

    diff, stats = preprocessor.compute_temporal_difference(pre, post)
    assert diff.shape == (10, 10)
    assert stats["potential_flood_pixels"] == 9
    assert stats["potential_flood_area_pct"] == 9.0
    assert stats["mean_difference_db"] < 0.0


def test_post_preprocess_sentinel_1_endpoint():
    """Verify POST /satellites/preprocess/sentinel-1 produces 6-channel SAR package."""
    pair_payload = {
        "satellite": "SENTINEL-1",
        "before_scene": {
            "product_id": "S1A_IW_GRDH_PRE",
            "collection": "SENTINEL-1",
            "acquisition_datetime": "2024-08-19T00:00:00Z",
            "product_type": "IW_GRDH_1S",
        },
        "after_scene": {
            "product_id": "S1A_IW_GRDH_POST",
            "collection": "SENTINEL-1",
            "acquisition_datetime": "2024-08-31T00:00:00Z",
            "product_type": "IW_GRDH_1S",
        },
        "orbit_compatibility": True,
        "temporal_separation_days": 12.0,
        "before_days_from_event": -7.0,
        "after_days_from_event": 5.0,
        "quality_notes": "Track 19",
    }
    
    response = client.post(
        "/satellites/preprocess/sentinel-1",
        json={"scene_pair": pair_payload, "speckle_filter_window": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "contract_prepared"
    assert len(data["channels"]) == 6  # pre_vv, pre_vh, post_vv, post_vh, diff_vv, diff_vh
    assert any(c["band_name"] == "diff_vv" for c in data["channels"])
    assert "change_summary" in data
    assert "3x3" in data["filter_applied"]


def test_preprocessing_pipeline_integration():
    """Verify preprocessing in analysis_service handles clipped pair input."""
    from app.services.analysis_service import preprocessing
    clipped_data = {
        "satellite": "SENTINEL-1",
        "aoi": {"type": "Polygon", "coordinates": [[[85.2, 27.8], [85.5, 27.8], [85.5, 28.1], [85.2, 28.1], [85.2, 27.8]]]},
        "status": "specification_prepared",
        "identical_grid_alignment": True,
        "grid_shape": [3340, 2951],
        "message": "co-registered",
        "before_scene_clip": {
            "product_id": "S1A_PRE",
            "scene_type": "before",
            "collection": "SENTINEL-1",
            "status": "prepared_contract",
            "clip_specification": {
                "aoi_geojson": {},
                "bounding_box": [85.2, 27.8, 85.5, 28.1],
                "target_crs": "EPSG:4326",
                "pixel_resolution_meters": 10.0,
                "pixel_resolution_degrees": 0.0001,
                "dimensions": [3340, 2951],
                "affine_transform": [0.0001, 0, 85.2, 0, -0.0001, 28.1],
                "aoi_area_sqkm": 985.2,
            },
            "bands": [],
            "message": "prepared",
        },
        "after_scene_clip": {
            "product_id": "S1A_POST",
            "scene_type": "after",
            "collection": "SENTINEL-1",
            "status": "prepared_contract",
            "clip_specification": {
                "aoi_geojson": {},
                "bounding_box": [85.2, 27.8, 85.5, 28.1],
                "target_crs": "EPSG:4326",
                "pixel_resolution_meters": 10.0,
                "pixel_resolution_degrees": 0.0001,
                "dimensions": [3340, 2951],
                "affine_transform": [0.0001, 0, 85.2, 0, -0.0001, 28.1],
                "aoi_area_sqkm": 985.2,
            },
            "bands": [],
            "message": "prepared",
        },
    }
    result = preprocessing(clipped_data)
    assert result["status"] == "contract_prepared"
    assert len(result["channels"]) == 6


# ============================================================================
# Sentinel-2 Optical Preprocessing & NDWI Tests (Module 3 Milestone 5)
# ============================================================================

def test_calculate_ndwi_physics():
    """Verify NDWI formula: water produces positive NDWI, vegetation produces negative."""
    import numpy as np
    from app.services.optical_preprocessor import get_optical_preprocessor
    optical_service = get_optical_preprocessor()

    # Water pixel: high green (0.15), low NIR (0.02)
    # Vegetation pixel: moderate green (0.05), very high NIR (0.45)
    green = np.array([[0.15, 0.05]], dtype=np.float32)
    nir = np.array([[0.02, 0.45]], dtype=np.float32)

    ndwi = optical_service.calculate_ndwi(green, nir)
    assert ndwi.shape == (1, 2)
    # Water pixel must be strongly positive (> 0.5)
    assert ndwi[0, 0] > 0.5
    # Vegetation pixel must be strongly negative (< -0.5)
    assert ndwi[0, 1] < -0.5


def test_calculate_mndwi_physics():
    """Verify MNDWI formula using SWIR channel."""
    import numpy as np
    from app.services.optical_preprocessor import get_optical_preprocessor
    optical_service = get_optical_preprocessor()

    green = np.array([[0.20, 0.10]], dtype=np.float32)
    swir = np.array([[0.03, 0.25]], dtype=np.float32)

    mndwi = optical_service.calculate_mndwi(green, swir)
    assert mndwi[0, 0] > 0.5
    assert mndwi[0, 1] < 0.0


def test_sentinel_2_cloud_screening_skips_cloudy():
    """Verify POST /satellites/preprocess/sentinel-2 skips optical when cloud cover > 50%."""
    cloudy_pair = {
        "satellite": "SENTINEL-2",
        "before_scene": {
            "product_id": "S2A_MSIL2A_PRE",
            "collection": "SENTINEL-2",
            "acquisition_datetime": "2024-08-15T00:00:00Z",
            "product_type": "S2MSI2A",
            "cloud_cover": 25.0,
        },
        "after_scene": {
            "product_id": "S2B_MSIL2A_POST_CLOUDY",
            "collection": "SENTINEL-2",
            "acquisition_datetime": "2024-08-28T00:00:00Z",
            "product_type": "S2MSI2A",
            "cloud_cover": 85.0,  # heavy cloud cover
        },
        "orbit_compatibility": True,
        "temporal_separation_days": 13.0,
        "before_days_from_event": -11.0,
        "after_days_from_event": 2.0,
        "quality_notes": "Cloudy post scene",
    }

    response = client.post(
        "/satellites/preprocess/sentinel-2",
        json={"scene_pair": cloudy_pair, "cloud_threshold_pct": 50.0},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "skipped_cloudy"
    assert data["is_optical_useful"] is False
    assert "exceeds" in data["reason"]


def test_sentinel_2_preprocess_usable():
    """Verify POST /satellites/preprocess/sentinel-2 produces NDWI when clear."""
    clear_pair = {
        "satellite": "SENTINEL-2",
        "before_scene": {
            "product_id": "S2A_MSIL2A_PRE_CLEAR",
            "collection": "SENTINEL-2",
            "acquisition_datetime": "2024-08-15T00:00:00Z",
            "product_type": "S2MSI2A",
            "cloud_cover": 10.0,
        },
        "after_scene": {
            "product_id": "S2B_MSIL2A_POST_CLEAR",
            "collection": "SENTINEL-2",
            "acquisition_datetime": "2024-08-28T00:00:00Z",
            "product_type": "S2MSI2A",
            "cloud_cover": 12.0,  # clear optical imagery
        },
        "orbit_compatibility": True,
        "temporal_separation_days": 13.0,
        "before_days_from_event": -11.0,
        "after_days_from_event": 2.0,
        "quality_notes": "Clear skies",
    }

    response = client.post(
        "/satellites/preprocess/sentinel-2",
        json={"scene_pair": clear_pair, "cloud_threshold_pct": 50.0},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "contract_prepared"
    assert data["is_optical_useful"] is True
    assert len(data["channels"]) == 4  # pre_ndwi, post_ndwi, diff_ndwi, post_mndwi
    assert any(c["band_name"] == "post_ndwi" for c in data["channels"])
    assert "water_detection_summary" in data


# ============================================================================
# Analysis-Ready Data (ARD) Integration Tests (Module 3 Milestone 6)
# ============================================================================

def test_post_prepare_ard_endpoint_default_nepal():
    """Verify POST /satellites/prepare-ard creates an end-to-end AnalysisReadyPackage."""
    payload = {
        "flood_date": "2024-08-26",
        "resolution_meters": 10.0,
        "speckle_filter_window": 3,
    }
    response = client.post("/satellites/prepare-ard", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["spatial_reference"]["grid_shape"] == [3340, 2951]
    assert data["spatial_reference"]["crs"] == "EPSG:4326"
    assert "sar_ard" in data
    assert len(data["channels_available"]) >= 6
    assert data["recommended_segmentation_model"] in ["sar_unet_flood", "multimodal_flood_debris"]
    assert data["manifest_file"] is not None

    # Verify retrieval by job_id
    job_id = data["job_id"]
    get_res = client.get(f"/satellites/ard/{job_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["job_id"] == job_id
    assert get_data["spatial_reference"]["grid_shape"] == [3340, 2951]


def test_get_ard_by_job_id_not_found():
    """Verify GET /satellites/ard/{job_id} returns 404 for unknown job."""
    response = client.get("/satellites/ard/job_non_existent_9999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()




