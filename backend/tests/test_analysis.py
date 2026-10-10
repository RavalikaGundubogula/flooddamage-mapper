import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.services.analysis_service import (
    get_analysis_service,
    satellite_router,
    image_selector,
    preprocessing,
    segmentation,
    geospatial_analysis,
    connectivity_analysis,
    report_generator,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_service():
    """Reset the in-memory repository before each test."""
    service = get_analysis_service()
    service.repository.clear()
    yield


# ============================================================================
# 1. Valid Polygon Tests
# ============================================================================

def test_create_analysis_valid_polygon():
    """Verify analysis creation with a standard GeoJSON Polygon."""
    payload = {
        "aoi": {
            "type": "Polygon",
            "coordinates": [
                [
                    [85.25, 27.65],
                    [85.40, 27.65],
                    [85.40, 27.80],
                    [85.25, 27.80],
                    [85.25, 27.65],
                ]
            ],
        },
        "flood_date": "2024-08-15",
    }
    response = client.post("/analysis", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "created"
    assert data["flood_date"] == "2024-08-15"
    assert data["report_language"] == "en"
    assert data["aoi"]["type"] == "Polygon"
    assert len(data["aoi"]["coordinates"][0]) == 5
    assert data["aoi"]["coordinates"][0][0] == [85.25, 27.65]


# ============================================================================
# 2. Valid Bounding Box Tests (List, bbox dict, named keys)
# ============================================================================

def test_create_analysis_valid_bounding_box_list():
    """Verify Bounding Box supplied as [min_lon, min_lat, max_lon, max_lat] is converted to Polygon."""
    payload = {
        "aoi": [85.25, 27.65, 85.40, 27.80],
        "flood_date": "2024-08-15",
        "report_language": "en",
    }
    response = client.post("/analysis", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["aoi"]["type"] == "Polygon"
    coords = data["aoi"]["coordinates"][0]
    assert len(coords) == 5
    # First and last must match to close the polygon
    assert coords[0] == coords[4]
    assert coords[0] == [85.25, 27.65]
    assert coords[2] == [85.40, 27.80]


def test_create_analysis_valid_bounding_box_dict():
    """Verify Bounding Box supplied as {'bbox': [min_lon, min_lat, max_lon, max_lat]}."""
    payload = {
        "aoi": {"bbox": [85.25, 27.65, 85.40, 27.80]},
        "flood_date": "2024-08-15",
    }
    response = client.post("/analysis", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["aoi"]["type"] == "Polygon"
    assert len(data["aoi"]["coordinates"][0]) == 5


def test_create_analysis_valid_bounding_box_named_keys():
    """Verify Bounding Box with explicit min_lon, min_lat, max_lon, max_lat keys."""
    payload = {
        "aoi": {
            "min_lon": 85.25,
            "min_lat": 27.65,
            "max_lon": 85.40,
            "max_lat": 27.80,
        },
        "flood_date": "2024-08-15",
        "report_language": "es",
    }
    response = client.post("/analysis", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["report_language"] == "es"
    assert data["aoi"]["type"] == "Polygon"


# ============================================================================
# 3. Invalid Latitude / Longitude Tests
# ============================================================================

def test_invalid_latitude_out_of_range():
    """Verify error when latitude exceeds +90 degrees."""
    payload = {
        "aoi": [85.25, 95.0, 85.40, 96.0],  # lat > 90
        "flood_date": "2024-08-15",
    }
    response = client.post("/analysis", json=payload)
    assert response.status_code == 422
    assert "Latitude must be between -90.0 and 90.0" in response.text


def test_invalid_longitude_out_of_range():
    """Verify error when longitude exceeds 180 degrees."""
    payload = {
        "aoi": [185.0, 27.0, 190.0, 28.0],  # lon > 180
        "flood_date": "2024-08-15",
    }
    response = client.post("/analysis", json=payload)
    assert response.status_code == 422
    assert "Longitude must be between -180.0 and 180.0" in response.text


def test_invalid_bounding_box_min_greater_than_max():
    """Verify error when min coordinate is greater than max coordinate."""
    payload = {
        "aoi": [85.40, 27.80, 85.25, 27.65],  # min > max
        "flood_date": "2024-08-15",
    }
    response = client.post("/analysis", json=payload)
    assert response.status_code == 422
    assert "min_lon" in response.text or "less than" in response.text


def test_unclosed_polygon_ring():
    """Verify error when polygon exterior ring is not closed."""
    payload = {
        "aoi": {
            "type": "Polygon",
            "coordinates": [
                [
                    [85.25, 27.65],
                    [85.40, 27.65],
                    [85.40, 27.80],
                    [85.25, 27.80],
                    # Missing closing point [85.25, 27.65]
                ]
            ],
        },
        "flood_date": "2024-08-15",
    }
    response = client.post("/analysis", json=payload)
    assert response.status_code == 422
    assert "closed" in response.text


# ============================================================================
# 4. Invalid Date Tests
# ============================================================================

def test_invalid_date_format():
    """Verify error when date does not follow YYYY-MM-DD."""
    payload = {
        "aoi": [85.25, 27.65, 85.40, 27.80],
        "flood_date": "15-08-2024",
    }
    response = client.post("/analysis", json=payload)
    assert response.status_code == 422
    assert "YYYY-MM-DD" in response.text


def test_invalid_calendar_date():
    """Verify error when date is impossible (e.g. Feb 30)."""
    payload = {
        "aoi": [85.25, 27.65, 85.40, 27.80],
        "flood_date": "2024-02-30",
    }
    response = client.post("/analysis", json=payload)
    assert response.status_code == 422


# ============================================================================
# 5. Retrieval Tests (Existing & Nonexistent)
# ============================================================================

def test_get_existing_analysis():
    """Verify retrieving an existing analysis job returns status and metadata."""
    create_payload = {
        "aoi": [85.25, 27.65, 85.40, 27.80],
        "flood_date": "2024-08-15",
        "report_language": "fr",
    }
    post_res = client.post("/analysis", json=create_payload)
    assert post_res.status_code == 201
    job_id = post_res.json()["job_id"]

    get_res = client.get(f"/analysis/{job_id}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["job_id"] == job_id
    assert data["status"] == "created"
    assert data["flood_date"] == "2024-08-15"
    assert data["report_language"] == "fr"
    assert "created_at" in data
    assert "updated_at" in data


def test_get_nonexistent_analysis():
    """Verify 404 is returned when requesting an unknown job_id."""
    response = client.get("/analysis/nonexistent-uuid-99999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_invalid_report_language():
    """Verify error when language code is invalid."""
    payload = {
        "aoi": [85.25, 27.65, 85.40, 27.80],
        "flood_date": "2024-08-15",
        "report_language": "invalid_too_long_12345",
    }
    response = client.post("/analysis", json=payload)
    assert response.status_code == 422


def test_empty_aoi_rejected():
    """Verify error when AOI is empty."""
    payload = {
        "aoi": {},
        "flood_date": "2024-08-15",
    }
    response = client.post("/analysis", json=payload)
    assert response.status_code == 422


def test_root_and_health_endpoints():
    """Verify root and health endpoints return 200 OK."""
    root_res = client.get("/")
    assert root_res.status_code == 200
    assert "endpoints" in root_res.json()

    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"


# ============================================================================
# 6. Future Module Placeholder Interfaces
# ============================================================================

def test_future_placeholders_raise_not_implemented():
    """Verify Module 2 satellite_router and Module 3 image_selector are implemented; modules 4-8 raise NotImplementedError."""
    with patch("app.clients.copernicus.CopernicusCatalogClient.search", return_value=[]):
        res = satellite_router(None, "2024-08-15")
        assert "strategy" in res

        sel_res = image_selector(None, "2024-08-15")
        assert "status" in sel_res

    with pytest.raises(NotImplementedError):
        preprocessing(None)

    with pytest.raises(NotImplementedError):
        segmentation(None)

    with pytest.raises(NotImplementedError):
        geospatial_analysis(None, {})

    with pytest.raises(NotImplementedError):
        connectivity_analysis(None)

    with pytest.raises(NotImplementedError):
        report_generator(None, "en")
