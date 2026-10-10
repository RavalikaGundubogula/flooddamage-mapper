import json
from pathlib import Path
import networkx as nx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.connectivity import (
    ConnectivityAnalysisRequest,
    DetourClassification,
    HospitalAccessStatus,
    RoadEdgeStatus,
    SettlementIsolationStatus,
)
from app.schemas.geospatial import (
    AssetSummary,
    BridgeAssetSummary,
    BuildingAssetSummary,
    ChangeZoneMetrics,
    FloodExtentMetrics,
    GeospatialAnalysisResponse,
    RoadAssetSummary,
)
from app.services.analysis_service import connectivity_analysis
from app.services.connectivity_service import (
    ConnectivityAnalysisService,
    get_connectivity_service,
)
from app.services.geospatial_service import get_geospatial_service
from app.services.osm_service import get_osm_service

client = TestClient(app)


# ============================================================================
# Deterministic Test Fixtures for Module 6
# ============================================================================

@pytest.fixture
def deterministic_osm_network():
    """
    Synthetic pre-event OSM dataset designed to deterministically test:
    - Town destination (Town A)
    - Flood-isolated settlement (Village B)
    - Major detour settlement (Village C)
    - Minor detour settlement (Village D)
    - Pre-existingly disconnected settlement (Hamlet E)
    - Hospital accessibility preserved vs lost vs no baseline
    - 0%, partial (<50%), and blocked (>=50%) road segments
    - MultiLineString roads and shared intersections
    """
    return {
        "settlements": [
            {
                "type": "Feature",
                "id": "osm_settlement_1",
                "properties": {
                    "settlement_id": "settlement_town_a",
                    "osm_id": "1001",
                    "name": "Central District Town",
                    "place_type": "town",
                    "population": 15000,
                },
                "geometry": {"type": "Point", "coordinates": [85.200, 27.800]},
            },
            {
                "type": "Feature",
                "id": "osm_settlement_2",
                "properties": {
                    "settlement_id": "settlement_village_b",
                    "osm_id": "1002",
                    "name": "Isolated Valley Village",
                    "place_type": "village",
                    "population": 2500,
                },
                "geometry": {"type": "Point", "coordinates": [85.230, 27.800]},
            },
            {
                "type": "Feature",
                "id": "osm_settlement_3",
                "properties": {
                    "settlement_id": "settlement_village_c",
                    "osm_id": "1003",
                    "name": "Major Detour Village",
                    "place_type": "village",
                    "population": 1200,
                },
                "geometry": {"type": "Point", "coordinates": [85.200, 27.830]},
            },
            {
                "type": "Feature",
                "id": "osm_settlement_4",
                "properties": {
                    "settlement_id": "settlement_village_d",
                    "osm_id": "1004",
                    "name": "Minor Detour Village",
                    "place_type": "village",
                    "population": 800,
                },
                "geometry": {"type": "Point", "coordinates": [85.180, 27.800]},
            },
            {
                "type": "Feature",
                "id": "osm_settlement_5",
                "properties": {
                    "settlement_id": "settlement_hamlet_e",
                    "osm_id": "1005",
                    "name": "Remote Mountain Hamlet",
                    "place_type": "hamlet",
                    # Deliberately omit population to verify None handling
                },
                "geometry": {"type": "Point", "coordinates": [85.400, 27.950]},
            },
        ],
        "hospitals": [
            {
                "type": "Feature",
                "id": "osm_hospital_1",
                "properties": {
                    "hospital_id": "hospital_central",
                    "osm_id": "2001",
                    "name": "Central General Hospital",
                    "amenity_type": "hospital",
                },
                "geometry": {"type": "Point", "coordinates": [85.201, 27.801]},
            },
        ],
        "roads": [
            # Road 1: Town A to Village B (Severed by flood: 80% flooded -> BLOCKED)
            {
                "type": "Feature",
                "id": "road_town_to_b",
                "properties": {
                    "road_id": "road_town_to_b",
                    "osm_id": "3001",
                    "name": "Valley Highway",
                    "highway_type": "primary",
                    "total_length_m": 3000.0,
                    "flooded_length_m": 2400.0,
                    "affected_percentage": 80.0,
                    "classification": "heavily_affected",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[85.200, 27.800], [85.230, 27.800]],
                },
            },
            # Road 2: Town A direct to Village C (Direct road: 60% flooded -> BLOCKED)
            {
                "type": "Feature",
                "id": "road_town_to_c_direct",
                "properties": {
                    "road_id": "road_town_to_c_direct",
                    "osm_id": "3002",
                    "name": "North Avenue Direct",
                    "highway_type": "secondary",
                    "total_length_m": 3300.0,
                    "flooded_length_m": 1980.0,
                    "affected_percentage": 60.0,
                    "classification": "heavily_affected",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[85.200, 27.800], [85.200, 27.830]],
                },
            },
            # Road 3A & 3B: Alternate detour route from Town A to Village C via Waypoint (0% affected -> OPEN)
            # Length: ~6000m (detour ~80% > 25% -> MAJOR_DETOUR)
            {
                "type": "Feature",
                "id": "road_detour_c_part1",
                "properties": {
                    "road_id": "road_detour_c_part1",
                    "osm_id": "3003",
                    "name": "West Ridge Bypass 1",
                    "highway_type": "tertiary",
                    "total_length_m": 3000.0,
                    "flooded_length_m": 0.0,
                    "affected_percentage": 0.0,
                    "classification": "unaffected",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[85.200, 27.800], [85.160, 27.815]],
                },
            },
            {
                "type": "Feature",
                "id": "road_detour_c_part2",
                "properties": {
                    "road_id": "road_detour_c_part2",
                    "osm_id": "3004",
                    "name": "West Ridge Bypass 2",
                    "highway_type": "tertiary",
                    "total_length_m": 3000.0,
                    "flooded_length_m": 0.0,
                    "affected_percentage": 0.0,
                    "classification": "unaffected",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[85.160, 27.815], [85.200, 27.830]],
                },
            },
            # Road 4: Town A to Village D (Primary road: 20% affected -> PARTIALLY_AFFECTED < 50%, remains open)
            {
                "type": "Feature",
                "id": "road_town_to_d_primary",
                "properties": {
                    "road_id": "road_town_to_d_primary",
                    "osm_id": "3005",
                    "name": "Southwest Link",
                    "highway_type": "secondary",
                    "total_length_m": 2000.0,
                    "flooded_length_m": 400.0,
                    "affected_percentage": 20.0,
                    "classification": "potentially_blocked",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[85.200, 27.800], [85.180, 27.800]],
                },
            },
            # Road 5: MultiLineString road connecting isolated mountain track (unconnected to Town A)
            {
                "type": "Feature",
                "id": "road_mountain_isolated",
                "properties": {
                    "road_id": "road_mountain_isolated",
                    "osm_id": "3006",
                    "name": "Isolated Ridge Track",
                    "highway_type": "track",
                    "total_length_m": 1500.0,
                    "flooded_length_m": 0.0,
                    "affected_percentage": 0.0,
                    "classification": "unaffected",
                },
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [
                        [[85.400, 27.950], [85.405, 27.952]],
                        [[85.405, 27.952], [85.410, 27.955]],
                    ],
                },
            },
            # Road 6: Redundant duplicate segment (testing robustness against duplicates)
            {
                "type": "Feature",
                "id": "road_duplicate_southwest",
                "properties": {
                    "road_id": "road_duplicate_southwest",
                    "osm_id": "3005",
                    "name": "Southwest Link Duplicate",
                    "highway_type": "secondary",
                    "total_length_m": 2000.0,
                    "flooded_length_m": 400.0,
                    "affected_percentage": 20.0,
                    "classification": "potentially_blocked",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[85.200, 27.800], [85.180, 27.800]],
                },
            },
        ],
        "bridges": [],
    }


@pytest.fixture
def mock_module5_geospatial_job(deterministic_osm_network, tmp_path):
    """Register a deterministic Module 5 geospatial response in GeospatialDamageAnalysisService."""
    geo_service = get_geospatial_service()
    job_id = "test_connectivity_job_001"

    # Save OSM dataset to temp file
    osm_path = tmp_path / "test_osm_network.json"
    osm_path.write_text(json.dumps(deterministic_osm_network), encoding="utf-8")

    # Construct synthetic GeospatialAnalysisResponse
    resp = GeospatialAnalysisResponse(
        job_id=job_id,
        status="completed",
        source_data_status="test_fixture",
        osm_status="available",
        aoi={
            "type": "Polygon",
            "coordinates": [[[85.15, 27.78], [85.42, 27.78], [85.42, 27.98], [85.15, 27.98], [85.15, 27.78]]],
        },
        crs="EPSG:4326",
        flood_metrics=FloodExtentMetrics(
            total_aoi_area_km2=50.0,
            valid_analysis_area_km2=50.0,
            flooded_pixel_count=1200,
            total_valid_pixels=10000,
            flooded_area_m2=120000.0,
            flooded_area_km2=0.12,
            flooded_percentage=12.0,
            flood_region_count=2,
            largest_flood_region_km2=0.08,
            smallest_flood_region_km2=0.04,
            mean_flood_region_km2=0.06,
        ),
        change_zone_metrics=ChangeZoneMetrics(
            status="available",
            total_change_zones=0,
            total_area_km2=0.0,
            largest_zone_km2=0.0,
            message="No candidate change zones.",
        ),
        asset_summary=AssetSummary(
            buildings=BuildingAssetSummary(
                total=0, unaffected=0, potentially_affected=0, heavily_affected=0, total_area_m2=0, affected_area_m2=0
            ),
            roads=RoadAssetSummary(
                total_segments=len(deterministic_osm_network["roads"]),
                unaffected=3,
                potentially_blocked=2,
                heavily_affected=2,
                total_length_km=18.0,
                affected_length_km=4.78,
            ),
            bridges=BridgeAssetSummary(
                total=0, unaffected=0, potentially_affected=0, heavily_affected=0, total_length_km=0, affected_length_km=0
            ),
        ),
        flood_regions={"type": "FeatureCollection", "features": []},
        change_zones={"type": "FeatureCollection", "features": []},
        buildings={"type": "FeatureCollection", "features": []},
        roads={"type": "FeatureCollection", "features": deterministic_osm_network["roads"]},
        bridges={"type": "FeatureCollection", "features": []},
        attribution=["© Copernicus Sentinel data", "© OpenStreetMap contributors"],
        message="Deterministic test Module 5 output.",
    )

    # Store in geospatial service
    geo_service._geospatial_store[job_id] = resp
    return job_id, str(osm_path)


# ============================================================================
# 1. Graph Construction Tests
# ============================================================================

def test_graph_construction_quality(deterministic_osm_network):
    """Verify graph builder handles LineString, MultiLineString, duplicate segments, and shared intersections."""
    service = get_connectivity_service()
    roads = deterministic_osm_network["roads"]
    bridges = deterministic_osm_network["bridges"]

    (
        base_g,
        post_g,
        edge_details,
        blocked_feats,
        partial_feats,
    ) = service._build_road_graphs(
        roads=roads,
        bridges=bridges,
        mean_lat=27.8,
        blocking_threshold_pct=50.0,
    )

    # 1. Nodes exist and have geographic coordinates
    assert base_g.number_of_nodes() > 0
    for node, data in base_g.nodes(data=True):
        assert "lon" in data
        assert "lat" in data
        assert isinstance(data["lon"], float)
        assert isinstance(data["lat"], float)

    # 2. MultiLineString road was parsed into graph edges
    multi_nodes = [n for n, d in base_g.nodes(data=True) if abs(d["lon"] - 85.405) < 0.001]
    assert len(multi_nodes) >= 1

    # 3. Blocked edges classification (>= 50% threshold)
    assert len(blocked_feats) == 2  # road_town_to_b (80%), road_town_to_c_direct (60%)
    for bf in blocked_feats:
        assert bf["properties"]["blockage_status"] == "estimated_blocked"
        assert bf["properties"]["affected_percentage"] >= 50.0

    # 4. Partially affected edges classification (< 50% threshold)
    assert len(partial_feats) >= 1
    for pf in partial_feats:
        assert pf["properties"]["blockage_status"] == "partially_affected"
        assert 0.0 < pf["properties"]["affected_percentage"] < 50.0

    # 5. Post-flood graph had blocked edges removed
    assert post_g.number_of_edges() < base_g.number_of_edges()


# ============================================================================
# 2. Configurable Blocking Threshold Tests
# ============================================================================

def test_configurable_blocking_threshold(deterministic_osm_network):
    """Verify that changing threshold dynamically alters blocked vs open edges."""
    service = get_connectivity_service()
    roads = deterministic_osm_network["roads"]
    bridges = deterministic_osm_network["bridges"]

    # At 15% threshold: road with 20% affected becomes BLOCKED
    _, _, edges_strict, blocked_strict, _ = service._build_road_graphs(
        roads=roads, bridges=bridges, mean_lat=27.8, blocking_threshold_pct=15.0
    )
    # road_town_to_d_primary (20%) is blocked under 15% threshold
    assert len(blocked_strict) >= 3

    # At 90% threshold: road with 80% and 60% remain PARTIALLY_AFFECTED
    _, _, edges_lenient, blocked_lenient, _ = service._build_road_graphs(
        roads=roads, bridges=bridges, mean_lat=27.8, blocking_threshold_pct=90.0
    )
    assert len(blocked_lenient) == 0


# ============================================================================
# 3. Settlement Connectivity & Isolation Classification
# ============================================================================

def test_settlement_connectivity_end_to_end(mock_module5_geospatial_job):
    """
    Verify complete classification taxonomy:
    - Town A: self/connected (unaffected, NO_CHANGE)
    - Village B: baseline connected -> post-flood severed (FLOOD_ISOLATED)
    - Village C: direct severed -> alternate path available (UNAFFECTED, MAJOR_DETOUR)
    - Village D: primary route partially affected (<50%) -> (UNAFFECTED, NO_CHANGE)
    - Hamlet E: never connected before flood -> (PRE_EXISTINGLY_DISCONNECTED)
    """
    job_id, osm_path = mock_module5_geospatial_job
    service = get_connectivity_service()

    req = ConnectivityAnalysisRequest(
        job_id=job_id,
        osm_geojson_path=osm_path,
        road_blocked_threshold_pct=50.0,
        detour_minor_threshold_pct=5.0,
        detour_major_threshold_pct=25.0,
    )
    resp = service.analyze_connectivity(req)

    assert resp.status == "completed"
    assert resp.job_id == job_id
    assert resp.settlements.total == 5

    by_id = {s.settlement_id: s for s in resp.settlement_details}

    # 1. Town A: Central administrative center
    town_a = by_id["settlement_town_a"]
    assert town_a.baseline_connected is True
    assert town_a.post_flood_connected is True
    assert town_a.status == SettlementIsolationStatus.UNAFFECTED
    assert town_a.detour_classification == DetourClassification.NO_CHANGE

    # 2. Village B: Strictly flood-isolated (had route to Town A, severed post-flood)
    village_b = by_id["settlement_village_b"]
    assert village_b.baseline_connected is True
    assert village_b.post_flood_connected is False
    assert village_b.status == SettlementIsolationStatus.FLOOD_ISOLATED
    assert village_b.detour_classification == DetourClassification.DISCONNECTED
    assert resp.settlements.flood_isolated >= 1

    # 3. Village C: Main road severed, but alternate bypass route exists
    village_c = by_id["settlement_village_c"]
    assert village_c.baseline_connected is True
    assert village_c.post_flood_connected is True
    assert village_c.status == SettlementIsolationStatus.UNAFFECTED
    assert village_c.post_flood_distance_km > village_c.baseline_distance_km
    assert village_c.detour_percentage > 25.0
    assert village_c.detour_classification == DetourClassification.MAJOR_DETOUR

    # 4. Village D: Route has partial water (<50%), remains traversable with no detour
    village_d = by_id["settlement_village_d"]
    assert village_d.baseline_connected is True
    assert village_d.post_flood_connected is True
    assert village_d.status == SettlementIsolationStatus.UNAFFECTED
    assert village_d.detour_classification == DetourClassification.NO_CHANGE

    # 5. Hamlet E: Pre-existingly disconnected mountain hamlet (MUST NOT be classified as flood-isolated!)
    hamlet_e = by_id["settlement_hamlet_e"]
    assert hamlet_e.baseline_connected is False
    assert hamlet_e.post_flood_connected is False
    assert hamlet_e.status == SettlementIsolationStatus.PRE_EXISTINGLY_DISCONNECTED
    assert hamlet_e.detour_classification == DetourClassification.DISCONNECTED
    assert resp.settlements.pre_existingly_disconnected >= 1


# ============================================================================
# 4. Hospital Accessibility Tests
# ============================================================================

def test_hospital_accessibility_rules(mock_module5_geospatial_job):
    """
    Verify healthcare accessibility rules:
    - Town A: Preserved access
    - Village B: Lost access (road to town hospital severed)
    - Village C: Preserved access via detour
    - Hamlet E: No baseline access (not flood caused)
    """
    job_id, osm_path = mock_module5_geospatial_job
    service = get_connectivity_service()

    req = ConnectivityAnalysisRequest(job_id=job_id, osm_geojson_path=osm_path)
    resp = service.analyze_connectivity(req)

    by_id = {s.settlement_id: s for s in resp.settlement_details}

    # Town A: Hospital at town, preserved access
    town_a = by_id["settlement_town_a"]
    assert town_a.baseline_hospital_access is True
    assert town_a.post_flood_hospital_access is True
    assert town_a.hospital_access_lost is False
    assert town_a.hospital_access_status == HospitalAccessStatus.HOSPITAL_ACCESS_PRESERVED

    # Village B: Had hospital access pre-event, route severed by flood -> access LOST!
    village_b = by_id["settlement_village_b"]
    assert village_b.baseline_hospital_access is True
    assert village_b.post_flood_hospital_access is False
    assert village_b.hospital_access_lost is True
    assert village_b.hospital_access_status == HospitalAccessStatus.HOSPITAL_ACCESS_LOST

    # Village C: Access preserved via detour
    village_c = by_id["settlement_village_c"]
    assert village_c.baseline_hospital_access is True
    assert village_c.post_flood_hospital_access is True
    assert village_c.hospital_access_lost is False
    assert village_c.hospital_access_status == HospitalAccessStatus.HOSPITAL_ACCESS_PRESERVED

    # Hamlet E: Pre-existing disconnected -> NO_BASELINE_HOSPITAL_ACCESS (access_lost must be False)
    hamlet_e = by_id["settlement_hamlet_e"]
    assert hamlet_e.baseline_hospital_access is False
    assert hamlet_e.post_flood_hospital_access is False
    assert hamlet_e.hospital_access_lost is False
    assert hamlet_e.hospital_access_status == HospitalAccessStatus.NO_BASELINE_HOSPITAL_ACCESS

    # Aggregated hospital access summary
    assert resp.hospital_access.settlements_with_baseline_access == 4
    assert resp.hospital_access.hospital_access_lost == 1
    assert resp.hospital_access.access_preserved == 3
    assert resp.hospital_access.no_baseline_access == 1


# ============================================================================
# 5. Graph Component Analysis Tests
# ============================================================================

def test_graph_component_analysis(mock_module5_geospatial_job):
    """Verify connected components count before and after flood network fragmentation."""
    job_id, osm_path = mock_module5_geospatial_job
    service = get_connectivity_service()

    req = ConnectivityAnalysisRequest(job_id=job_id, osm_geojson_path=osm_path)
    resp = service.analyze_connectivity(req)

    net = resp.road_network
    assert net.baseline_components >= 1
    assert net.post_flood_components >= net.baseline_components
    assert net.newly_disconnected_components == net.post_flood_components - net.baseline_components
    assert net.blocked_edges == 2


# ============================================================================
# 6. GeoJSON Outputs Tests
# ============================================================================

def test_geojson_layers_structure(mock_module5_geospatial_job):
    """Verify all 5 required GeoJSON layers are generated with valid properties and geometries."""
    job_id, osm_path = mock_module5_geospatial_job
    service = get_connectivity_service()

    req = ConnectivityAnalysisRequest(job_id=job_id, osm_geojson_path=osm_path)
    resp = service.analyze_connectivity(req)

    # 1. Blocked roads GeoJSON
    assert resp.blocked_roads["type"] == "FeatureCollection"
    assert len(resp.blocked_roads["features"]) == 2
    for f in resp.blocked_roads["features"]:
        assert f["geometry"]["type"] == "LineString"
        assert f["properties"]["blockage_status"] == "estimated_blocked"
        assert "length_m" in f["properties"]

    # 2. Partially affected roads GeoJSON
    assert resp.partially_affected_roads["type"] == "FeatureCollection"
    for f in resp.partially_affected_roads["features"]:
        assert f["geometry"]["type"] == "LineString"
        assert f["properties"]["blockage_status"] == "partially_affected"

    # 3. Isolated settlements GeoJSON
    assert resp.isolated_settlements["type"] == "FeatureCollection"
    assert len(resp.isolated_settlements["features"]) == 1
    iso_feat = resp.isolated_settlements["features"][0]
    assert iso_feat["geometry"]["type"] == "Point"
    assert iso_feat["properties"]["status"] == "flood_isolated"
    assert iso_feat["properties"]["settlement_id"] == "settlement_village_b"

    # 4. Hospital access lost settlements GeoJSON
    assert resp.hospital_access_lost_settlements["type"] == "FeatureCollection"
    assert len(resp.hospital_access_lost_settlements["features"]) == 1

    # 5. All settlements connectivity GeoJSON
    assert resp.settlements_connectivity["type"] == "FeatureCollection"
    assert len(resp.settlements_connectivity["features"]) == 5


# ============================================================================
# 7. FastAPI Endpoints Tests
# ============================================================================

def test_api_connectivity_endpoints(mock_module5_geospatial_job):
    """Verify all 6 Module 6 FastAPI endpoints return 200 OK and valid schemas."""
    job_id, osm_path = mock_module5_geospatial_job

    # 1. POST /analysis/{job_id}/connectivity
    post_res = client.post(
        f"/analysis/{job_id}/connectivity",
        json={"osm_geojson_path": osm_path},
    )
    assert post_res.status_code == 200
    data = post_res.json()
    assert data["job_id"] == job_id
    assert data["status"] == "completed"
    assert "road_network" in data
    assert "settlements" in data
    assert "hospital_access" in data

    # 2. GET /analysis/{job_id}/connectivity
    get_res = client.get(f"/analysis/{job_id}/connectivity")
    assert get_res.status_code == 200
    assert get_res.json()["job_id"] == job_id

    # 3. GET /analysis/{job_id}/connectivity/roads
    roads_res = client.get(f"/analysis/{job_id}/connectivity/roads")
    assert roads_res.status_code == 200
    assert roads_res.json()["type"] == "FeatureCollection"

    # 4. GET /analysis/{job_id}/connectivity/settlements
    settle_res = client.get(f"/analysis/{job_id}/connectivity/settlements")
    assert settle_res.status_code == 200
    assert settle_res.json()["type"] == "FeatureCollection"

    # 5. GET /analysis/{job_id}/connectivity/isolated
    iso_res = client.get(f"/analysis/{job_id}/connectivity/isolated")
    assert iso_res.status_code == 200
    assert iso_res.json()["type"] == "FeatureCollection"
    assert len(iso_res.json()["features"]) == 1

    # 6. GET /analysis/{job_id}/connectivity/hospitals
    hosp_res = client.get(f"/analysis/{job_id}/connectivity/hospitals")
    assert hosp_res.status_code == 200
    assert hosp_res.json()["type"] == "FeatureCollection"

    # 7. 404 for nonexistent job
    not_found_res = client.get("/analysis/nonexistent_job_99999/connectivity")
    assert not_found_res.status_code == 404


# ============================================================================
# 8. 2026 Case Study Blocker Validation
# ============================================================================

def test_2026_case_study_blocker_handled_accurately():
    """
    Verify that real 2026 Nepal case study job does not fabricate connectivity
    conclusions while physical satellite binary downloads remain blocked.
    """
    service = get_connectivity_service()
    geo_service = get_geospatial_service()
    job_id = "job_test_blocked_connectivity"

    # Set up synthetic blocked geospatial result for Nepal 2026
    blocked_geo_resp = GeospatialAnalysisResponse(
        job_id=job_id,
        status="satellite_binary_data_blocked",
        source_data_status="satellite_binary_data_blocked",
        osm_status="available",
        aoi={"type": "Polygon", "coordinates": [[[85.2, 27.6], [85.4, 27.6], [85.4, 27.8], [85.2, 27.8], [85.2, 27.6]]]},
        crs="EPSG:4326",
        flood_metrics=FloodExtentMetrics(
            total_aoi_area_km2=10.0, valid_analysis_area_km2=10.0, flooded_pixel_count=0,
            total_valid_pixels=1000, flooded_area_m2=0, flooded_area_km2=0, flooded_percentage=0,
            flood_region_count=0, largest_flood_region_km2=0, smallest_flood_region_km2=0, mean_flood_region_km2=0
        ),
        change_zone_metrics=ChangeZoneMetrics(
            status="available", total_change_zones=0, total_area_km2=0, largest_zone_km2=0, message="None"
        ),
        asset_summary=AssetSummary(
            buildings=BuildingAssetSummary(total=0, unaffected=0, potentially_affected=0, heavily_affected=0, total_area_m2=0, affected_area_m2=0),
            roads=RoadAssetSummary(total_segments=0, unaffected=0, potentially_blocked=0, heavily_affected=0, total_length_km=0, affected_length_km=0),
            bridges=BridgeAssetSummary(total=0, unaffected=0, potentially_affected=0, heavily_affected=0, total_length_km=0, affected_length_km=0),
        ),
        flood_regions={"type": "FeatureCollection", "features": []},
        change_zones={"type": "FeatureCollection", "features": []},
        buildings={"type": "FeatureCollection", "features": []},
        roads={"type": "FeatureCollection", "features": []},
        bridges={"type": "FeatureCollection", "features": []},
        attribution=["© Copernicus Sentinel data"],
        message="2026 CASE STUDY: BLOCKED — waiting for actual Sentinel binary data.",
    )
    geo_service._geospatial_store[job_id] = blocked_geo_resp

    req = ConnectivityAnalysisRequest(job_id=job_id)
    resp = service.analyze_connectivity(req)

    assert resp.status == "satellite_binary_data_blocked"
    assert resp.source_data_status == "satellite_binary_data_blocked"
    assert "2026 CASE STUDY: BLOCKED" in resp.message
    assert "waiting for actual Sentinel binary data" in resp.message
    assert resp.settlements.flood_isolated == 0
    assert resp.road_network.blocked_edges == 0
    assert resp.hospital_access.hospital_access_lost == 0


# ============================================================================
# 9. Data Integrity & Negative Tests
# ============================================================================

def test_connectivity_analysis_interface_behavior(mock_module5_geospatial_job):
    """Verify that connectivity_analysis(None) raises NotImplementedError as tested by test_analysis.py."""
    with pytest.raises(NotImplementedError):
        connectivity_analysis(None)

    job_id, _ = mock_module5_geospatial_job
    res_dict = connectivity_analysis({"job_id": job_id})
    assert isinstance(res_dict, dict)
    assert res_dict["job_id"] == job_id
    assert res_dict["status"] == "completed"


def test_missing_geospatial_analysis_raises_value_error():
    """Verify that attempting connectivity analysis on a nonexistent job raises ValueError."""
    service = get_connectivity_service()
    req = ConnectivityAnalysisRequest(job_id="completely_nonexistent_job_abc")
    with pytest.raises(ValueError) as excinfo:
        service.analyze_connectivity(req)
    assert "geospatial" in str(excinfo.value).lower()



def test_no_invented_settlement_population(mock_module5_geospatial_job):
    """Verify that settlement populations are never invented or fabricated."""
    job_id, osm_path = mock_module5_geospatial_job
    service = get_connectivity_service()

    req = ConnectivityAnalysisRequest(job_id=job_id, osm_geojson_path=osm_path)
    resp = service.analyze_connectivity(req)

    by_id = {s.settlement_id: s for s in resp.settlement_details}
    # Hamlet E had no population tag in OSM fixture
    assert by_id["settlement_hamlet_e"].population is None
    # Town A had population 15000 in OSM fixture
    assert by_id["settlement_town_a"].population == 15000


def test_no_invented_hospital_capacities_or_beds(mock_module5_geospatial_job):
    """Verify that hospital details do NOT invent bed counts, emergency capacities, or operating status."""
    job_id, osm_path = mock_module5_geospatial_job
    service = get_connectivity_service()

    req = ConnectivityAnalysisRequest(job_id=job_id, osm_geojson_path=osm_path)
    resp = service.analyze_connectivity(req)

    for h in resp.hospital_details:
        h_dict = h.model_dump()
        assert "beds" not in h_dict
        assert "capacity" not in h_dict
        assert "operating_status" not in h_dict
        assert "emergency_capacity" not in h_dict
        assert h.nearest_node_id is not None

