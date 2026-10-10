import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
import rasterio
from affine import Affine
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.schemas.ard import PrepareArdRequest
from app.schemas.geospatial import (
    BridgeExposureClassification,
    BuildingExposureClassification,
    GeospatialAnalysisRequest,
    RoadExposureClassification,
)
from app.schemas.segmentation import SegmentationRequest
from app.services.ard_service import get_ard_service
from app.services.geospatial_service import (
    calculate_linestring_metric_length,
    calculate_polygon_metric_area,
    compute_geodesic_scales,
    get_geospatial_service,
)
from app.services.osm_service import get_osm_service
from app.services.segmentation_service import get_segmentation_service
import shapely.geometry

client = TestClient(app)


# ============================================================================
# Deterministic Test Fixtures for Module 5
# ============================================================================

@pytest.fixture
def test_osm_dataset():
    """Deterministic, self-contained OSM dataset representing buildings, roads, and bridges."""
    return {
        "buildings": [
            {
                "type": "Feature",
                "id": "osm_bldg_101",
                "properties": {"osm_id": "101", "name": "Community Health Post", "building_type": "hospital"},
                # Sits completely inside the flooded zone [85.24, 27.84] to [85.26, 27.86]
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[85.242, 27.842], [85.248, 27.842], [85.248, 27.848], [85.242, 27.848], [85.242, 27.842]]],
                },
            },
            {
                "type": "Feature",
                "id": "osm_bldg_102",
                "properties": {"osm_id": "102", "name": "Trishuli School", "building_type": "school"},
                # Partially overlaps the boundary of the flooded zone
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[85.258, 27.848], [85.264, 27.848], [85.264, 27.854], [85.258, 27.854], [85.258, 27.848]]],
                },
            },
            {
                "type": "Feature",
                "id": "osm_bldg_103",
                "properties": {"osm_id": "103", "name": "Upland Temple", "building_type": "shrine"},
                # Completely outside flood zone (upland safe terrain)
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[85.400, 28.000], [85.410, 28.000], [85.410, 28.010], [85.400, 28.010], [85.400, 28.000]]],
                },
            },
        ],
        "roads": [
            {
                "type": "Feature",
                "id": "osm_road_201",
                "properties": {"osm_id": "201", "name": "Trishuli Highway", "highway_type": "primary"},
                # Passes straight through the flood zone
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[85.230, 27.850], [85.270, 27.850]],
                },
            },
            {
                "type": "Feature",
                "id": "osm_road_202",
                "properties": {"osm_id": "202", "name": "Mountain Ridge Trail", "highway_type": "path"},
                # Safe ridge road, zero overlap
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[85.380, 27.950], [85.450, 27.950]],
                },
            },
        ],
        "bridges": [
            {
                "type": "Feature",
                "id": "osm_bridge_301",
                "properties": {"osm_id": "301", "name": "Trishuli River Suspension Bridge", "bridge_type": "suspension"},
                # Intersects the flood zone
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[85.245, 27.845], [85.255, 27.855]],
                },
            },
            {
                "type": "Feature",
                "id": "osm_bridge_302",
                "properties": {"osm_id": "302", "name": "High Mountain Footbridge", "bridge_type": "footbridge"},
                # Unaffected high mountain bridge
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[85.420, 28.020], [85.430, 28.020]],
                },
            },
        ],
    }


@pytest.fixture
def synthesized_flood_job(tmp_path):
    """Creates a deterministic test job with a synthetic GeoTIFF flood mask in data/outputs."""
    settings = get_settings()
    job_id = "test_geo_job_fixture_001"
    outputs_dir = settings.get_data_path(settings.DATA_OUTPUTS_DIR)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    mask_file = outputs_dir / f"{job_id}_flood_mask.tif"

    # Create a 100x100 raster with a flooded block [40:60, 40:60]
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[40:60, 40:60] = 1  # 400 flooded pixels
    # Add an isolated single-pixel noise artifact at [10, 10]
    arr[10, 10] = 1

    transform = Affine.translation(85.2, 28.1) @ Affine.scale(0.003, -0.003)
    with rasterio.open(
        mask_file,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype=rasterio.uint8,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(arr, 1)

    # Register in segmentation service and save sidecar JSON to outputs_dir
    from app.schemas.segmentation import AffectedAreaSummary, SegmentationResponse
    seg_resp = SegmentationResponse(
        job_id=job_id,
        status="completed",
        model_name="flood_unet_s1",
        confidence_threshold=0.5,
        summary=AffectedAreaSummary(
            flood_pixel_count=401,
            total_valid_pixels=10000,
            flood_percentage=4.01,
            flood_area_m2=4010000.0,
            flood_area_sqkm=4.01,
        ),
        mask_raster_path=str(mask_file.resolve()),
        geojson_features_count=2,
        geojson={"type": "FeatureCollection", "features": []},
        spatial_bounds=[85.2, 27.8, 85.5, 28.1],
        crs="EPSG:4326",
        created_at="2026-10-06T12:00:00Z",
        message="Test segmentation fixture",
    )
    seg_json = outputs_dir / f"{job_id}_segmentation.json"
    with open(seg_json, "w", encoding="utf-8") as f:
        json.dump(seg_resp.model_dump(mode="json"), f, indent=2)

    yield job_id, mask_file

    # Cleanup
    if mask_file.exists():
        mask_file.unlink()
    if seg_json.exists():
        seg_json.unlink()
    shutil.rmtree(outputs_dir / job_id, ignore_errors=True)


# ============================================================================
# 1. Geodesic Calculations & Metric Area Tests
# ============================================================================

def test_compute_geodesic_scales():
    m_lat, m_lon_equator = compute_geodesic_scales(0.0)
    assert round(m_lat, 0) == 111320.0
    assert round(m_lon_equator, 0) == 111320.0

    m_lat, m_lon_nepal = compute_geodesic_scales(28.0)
    assert round(m_lat, 0) == 111320.0
    # At 28 deg N, cos(28) ~ 0.88295 -> ~98,290 m
    assert 95000.0 < m_lon_nepal < 100000.0


def test_polygon_metric_area_calculation():
    # 0.01 deg by 0.01 deg square at 28 deg N (~983m x ~1113m ~ 1,094,000 m2)
    poly = shapely.geometry.box(85.2, 27.9, 85.21, 27.91)
    area_m2 = calculate_polygon_metric_area(poly, mean_lat=28.0)
    assert 1_000_000.0 < area_m2 < 1_200_000.0


def test_linestring_metric_length_calculation():
    # Line of 0.01 deg longitude at 28 deg N (~983 m)
    line = shapely.geometry.LineString([[85.2, 28.0], [85.21, 28.0]])
    length_m = calculate_linestring_metric_length(line, mean_lat=28.0)
    assert 950.0 < length_m < 1020.0


# ============================================================================
# 2. Flood Mask Polygon Extraction & Filtering Tests
# ============================================================================

def test_flood_polygon_extraction_and_noise_filtering(synthesized_flood_job):
    job_id, mask_file = synthesized_flood_job
    service = get_geospatial_service()

    metrics, polys, geojson = service._extract_flood_polygons(
        mask_raster_path=mask_file,
        job_id=job_id,
        min_area_m2=200000.0,  # Filter out single-pixel noise (~98,520 m2)
    )

    assert metrics.flooded_pixel_count == 401
    assert metrics.total_valid_pixels == 10000
    assert metrics.flooded_percentage == 4.01
    assert metrics.flooded_area_km2 > 0.0

    # Only the main connected flood region passes min_area_m2 filter (1 polygon)
    assert metrics.flood_region_count == 1
    assert len(polys) == 1
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1

    feature = geojson["features"][0]
    assert feature["id"] == "flood_region_001"
    assert "area_m2" in feature["properties"]
    assert "centroid" in feature["properties"]
    assert "bounding_box" in feature["properties"]


# ============================================================================
# 3. OSM Pre-event Asset Extraction & Cache Tests
# ============================================================================

def test_osm_service_parse_local_dataset(test_osm_dataset):
    osm_service = get_osm_service()
    aoi = {"type": "Polygon", "coordinates": [[[85.2, 27.8], [85.5, 27.8], [85.5, 28.1], [85.2, 28.1], [85.2, 27.8]]]}

    status, bldgs, roads, bridges = osm_service._parse_local_osm_dataset(test_osm_dataset, aoi)
    assert status == "available"
    assert len(bldgs) == 3
    assert len(roads) == 2
    assert len(bridges) == 2


def test_osm_service_handles_unreachable_gracefully():
    osm_service = get_osm_service()
    # Invalid dummy overpass url
    osm_service.overpass_url = "http://localhost:59999/invalid-overpass"
    osm_service.timeout = 1
    aoi = {"type": "Polygon", "coordinates": [[[80.0, 20.0], [80.1, 20.0], [80.1, 20.1], [80.0, 20.1], [80.0, 20.0]]]}

    status, bldgs, roads, bridges = osm_service.fetch_pre_event_assets(aoi=aoi, force_refresh=True)
    assert status == "unavailable"
    assert len(bldgs) == 0
    assert len(roads) == 0
    assert len(bridges) == 0


# ============================================================================
# 4. Building Impact & Classification Tests
# ============================================================================

def test_building_exposure_classification(test_osm_dataset):
    service = get_geospatial_service()
    # Flood polygon covering [85.24, 27.84] to [85.26, 27.86]
    flood_box = shapely.geometry.box(85.24, 27.84, 85.26, 27.86)
    flood_polys = [("flood_region_001", flood_box, 100000.0)]

    impacts, geojson, summary = service._evaluate_buildings(
        buildings=test_osm_dataset["buildings"],
        flood_polygons=flood_polys,
        mean_lat=27.85,
        pot_thresh=0.0,
        heavy_thresh=50.0,
    )

    assert summary.total == 3
    assert summary.heavily_affected == 1  # Building 101 completely inundated
    assert summary.potentially_affected == 1  # Building 102 partially intersected
    assert summary.unaffected == 1  # Building 103 upland

    # Verify building classifications
    b101 = next(i for i in impacts if i.building_id == "osm_bldg_101")
    assert b101.classification == BuildingExposureClassification.HEAVILY_AFFECTED
    assert b101.overlap_percentage > 90.0

    b102 = next(i for i in impacts if i.building_id == "osm_bldg_102")
    assert b102.classification == BuildingExposureClassification.POTENTIALLY_AFFECTED

    b103 = next(i for i in impacts if i.building_id == "osm_bldg_103")
    assert b103.classification == BuildingExposureClassification.UNAFFECTED
    assert b103.distance_to_flood_m > 5000.0


# ============================================================================
# 5. Road & Bridge Impact Tests
# ============================================================================

def test_road_and_bridge_exposure(test_osm_dataset):
    service = get_geospatial_service()
    flood_box = shapely.geometry.box(85.24, 27.84, 85.26, 27.86)
    flood_polys = [("flood_region_001", flood_box, 100000.0)]

    # Road evaluation
    road_impacts, r_geojson, r_summary = service._evaluate_roads(
        roads=test_osm_dataset["roads"],
        flood_polygons=flood_polys,
        mean_lat=27.85,
        pot_thresh=0.0,
        heavy_thresh=50.0,
    )
    assert r_summary.total_segments == 2
    r201 = next(r for r in road_impacts if r.road_id == "osm_road_201")
    assert r201.classification in [RoadExposureClassification.POTENTIALLY_BLOCKED, RoadExposureClassification.HEAVILY_AFFECTED]
    assert r201.flooded_length_m > 0.0

    r202 = next(r for r in road_impacts if r.road_id == "osm_road_202")
    assert r202.classification == RoadExposureClassification.UNAFFECTED

    # Bridge evaluation
    br_impacts, br_geojson, br_summary = service._evaluate_bridges(
        bridges=test_osm_dataset["bridges"],
        flood_polygons=flood_polys,
        mean_lat=27.85,
        pot_thresh=0.0,
        heavy_thresh=50.0,
    )
    assert br_summary.total == 2
    b301 = next(b for b in br_impacts if b.bridge_id == "osm_bridge_301")
    assert b301.classification in [BridgeExposureClassification.POTENTIALLY_AFFECTED, BridgeExposureClassification.HEAVILY_AFFECTED]
    assert b301.flooded_length_m > 0.0


# ============================================================================
# 6. Change Zone & Debris Heuristic Layer Tests
# ============================================================================

def test_change_zone_heuristics_insufficient_data():
    service = get_geospatial_service()
    # When ARD package has no physical difference files
    metrics, geojson = service._analyze_change_zones(
        ard_pkg=None,
        flood_polygons=[],
        job_id="test_job",
    )
    assert metrics.status == "insufficient_data"
    assert metrics.total_change_zones == 0
    assert len(geojson["features"]) == 0


# ============================================================================
# 7. End-to-End Service & API Endpoint Integration Tests
# ============================================================================

def test_geospatial_analysis_with_local_osm_fixture(synthesized_flood_job, test_osm_dataset, tmp_path):
    job_id, _ = synthesized_flood_job
    service = get_geospatial_service()

    # Save local OSM fixture file
    osm_file = tmp_path / "test_osm_data.json"
    osm_file.write_text(json.dumps(test_osm_dataset), encoding="utf-8")

    req = GeospatialAnalysisRequest(
        job_id=job_id,
        min_flood_region_area_m2=100.0,
        osm_geojson_path=str(osm_file),
    )
    resp = service.analyze_job_impact(req)

    assert resp.job_id == job_id
    assert resp.status == "completed"
    assert resp.osm_status == "available"
    assert resp.flood_metrics.flood_region_count >= 1
    assert resp.asset_summary.buildings.total == 3
    assert resp.asset_summary.roads.total_segments == 2
    assert resp.asset_summary.bridges.total == 2
    assert "© OpenStreetMap contributors" in resp.attribution

    # Verify retrieval
    stored = service.get_geospatial_result(job_id)
    assert stored is not None
    assert stored.job_id == job_id


def test_api_geospatial_endpoints(synthesized_flood_job, test_osm_dataset, tmp_path):
    job_id, _ = synthesized_flood_job
    osm_file = tmp_path / "test_osm_data.json"
    osm_file.write_text(json.dumps(test_osm_dataset), encoding="utf-8")

    # 1. POST /analysis/{job_id}/geospatial
    post_res = client.post(
        f"/analysis/{job_id}/geospatial",
        json={"osm_geojson_path": str(osm_file)},
    )
    assert post_res.status_code == 200
    data = post_res.json()
    assert data["job_id"] == job_id
    assert data["status"] == "completed"
    assert "flood_metrics" in data
    assert "asset_summary" in data

    # 2. GET /analysis/{job_id}/geospatial
    get_res = client.get(f"/analysis/{job_id}/geospatial")
    assert get_res.status_code == 200
    assert get_res.json()["job_id"] == job_id

    # 3. GET /analysis/{job_id}/geospatial/flood-regions
    reg_res = client.get(f"/analysis/{job_id}/geospatial/flood-regions")
    assert reg_res.status_code == 200
    assert reg_res.json()["type"] == "FeatureCollection"

    # 4. GET /analysis/{job_id}/geospatial/buildings
    bldg_res = client.get(f"/analysis/{job_id}/geospatial/buildings")
    assert bldg_res.status_code == 200
    assert bldg_res.json()["type"] == "FeatureCollection"

    # 5. GET /analysis/{job_id}/geospatial/roads
    road_res = client.get(f"/analysis/{job_id}/geospatial/roads")
    assert road_res.status_code == 200
    assert road_res.json()["type"] == "FeatureCollection"

    # 6. GET /analysis/{job_id}/geospatial/bridges
    br_res = client.get(f"/analysis/{job_id}/geospatial/bridges")
    assert br_res.status_code == 200
    assert br_res.json()["type"] == "FeatureCollection"

    # 7. GET /analysis/{job_id}/geospatial/change-zones
    cz_res = client.get(f"/analysis/{job_id}/geospatial/change-zones")
    assert cz_res.status_code == 200
    assert cz_res.json()["type"] == "FeatureCollection"

    # 8. 404 for nonexistent job
    err_res = client.get("/analysis/nonexistent_job_12345/geospatial")
    assert err_res.status_code == 404


def test_2026_case_study_blocker_handled_accurately():
    """Verify that job_nepal_2026_case accurately reports its data status."""
    service = get_geospatial_service()
    job_id = "job_nepal_2026_case"

    resp = service.get_geospatial_result(job_id)
    if not resp:
        req = GeospatialAnalysisRequest(job_id=job_id)
        resp = service.analyze_job_impact(req)
    assert resp.source_data_status in ["satellite_binary_data_blocked", "calibrated_raster_available"]
    if resp.source_data_status == "calibrated_raster_available":
        assert resp.status == "completed"
        assert resp.asset_summary.buildings.total > 0
    else:
        assert resp.status == "satellite_binary_data_blocked"
        assert "2026 CASE STUDY: BLOCKED" in resp.message
        assert resp.asset_summary.buildings.heavily_affected == 0
