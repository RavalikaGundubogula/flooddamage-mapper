import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import shapely.geometry
import shapely.ops

from app.core.config import Settings, get_settings
from app.schemas.connectivity import (
    ConnectivityAnalysisRequest,
    ConnectivityAnalysisResponse,
    DetourClassification,
    HospitalAccessStatus,
    HospitalAccessSummary,
    HospitalDetail,
    RoadEdgeDetail,
    RoadEdgeStatus,
    RoadNetworkSummary,
    SettlementConnectivityDetail,
    SettlementIsolationStatus,
    SettlementSummary,
)
from app.schemas.geospatial import GeospatialAnalysisRequest, GeospatialAnalysisResponse
from app.services.analysis_service import AnalysisRepository, get_analysis_service
from app.services.geospatial_service import (
    calculate_linestring_metric_length,
    compute_geodesic_scales,
    get_geospatial_service,
    GeospatialDamageAnalysisService,
)
from app.services.osm_service import OpenStreetMapService, get_osm_service

logger = logging.getLogger(__name__)


class ConnectivityAnalysisService:
    """
    Module 6 Service: Settlement & Road Network Accessibility Analysis.
    
    Transforms Module 5 spatial exposure into a topological road network graph:
    - Builds baseline routable graph from pre-event OSM roads & bridges.
    - Applies configurable blockage thresholds to derive post-flood estimated graph.
    - Evaluates baseline vs post-flood reachability to destinations and hospitals.
    - Classifies settlement isolation (unaffected, flood_isolated, pre_existingly_disconnected).
    - Quantifies road detours and health facility access loss.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        geo_service: Optional[GeospatialDamageAnalysisService] = None,
        osm_service: Optional[OpenStreetMapService] = None,
        repo: Optional[AnalysisRepository] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.geo_service = geo_service or get_geospatial_service()
        self.osm_service = osm_service or get_osm_service()
        self.repo = repo or get_analysis_service().repository

        self.outputs_dir = self.settings.get_data_path(self.settings.DATA_OUTPUTS_DIR)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._connectivity_store: Dict[str, ConnectivityAnalysisResponse] = {}

    def analyze_connectivity(
        self, request: ConnectivityAnalysisRequest
    ) -> ConnectivityAnalysisResponse:
        """
        Execute comprehensive Module 6 road network and settlement connectivity analysis for a job.
        """
        job_id = request.job_id
        if not job_id:
            raise ValueError("job_id must be provided to run connectivity analysis.")

        # 1. Retrieve Module 5 geospatial asset analysis results
        geo_res = self.geo_service.get_geospatial_result(job_id)
        if not geo_res:
            logger.info("Geospatial result not found in store for %s; attempting to compute on the fly...", job_id)
            try:
                geo_req = GeospatialAnalysisRequest(
                    job_id=job_id,
                    osm_geojson_path=request.osm_geojson_path,
                )
                geo_res = self.geo_service.analyze_job_impact(geo_req)
            except Exception as e:
                raise ValueError(
                    f"No geospatial damage analysis results found or could be computed for job '{job_id}': {e}. "
                    f"Please run POST /analysis/{job_id}/geospatial first."
                )

        # 2. Check for physical satellite blocker
        if (
            geo_res.status == "satellite_binary_data_blocked"
            or geo_res.source_data_status == "satellite_binary_data_blocked"
        ):
            logger.warning(
                "Source data status is 'satellite_binary_data_blocked' for job %s. "
                "Returning explicit blocker status; zero settlement isolation claimed.",
                job_id,
            )
            return self._build_blocked_case_study_response(job_id, geo_res, request)

        # 3. Mean latitude for geodesic scale calculations
        aoi = geo_res.aoi
        try:
            sh_aoi = shapely.geometry.shape(aoi)
            mean_lat = (sh_aoi.bounds[1] + sh_aoi.bounds[3]) / 2.0
        except Exception:
            mean_lat = 27.7

        # 4. Construct Baseline & Post-Flood Road Graphs
        roads_features = geo_res.roads.get("features", [])
        bridges_features = geo_res.bridges.get("features", [])

        (
            baseline_graph,
            post_flood_graph,
            road_edges,
            blocked_road_features,
            partially_affected_road_features,
        ) = self._build_road_graphs(
            roads=roads_features,
            bridges=bridges_features,
            mean_lat=mean_lat,
            blocking_threshold_pct=request.road_blocked_threshold_pct,
        )

        # 5. Extract Settlements & Hospitals from Pre-event OSM
        settlements_features, hospitals_features = self._acquire_settlements_and_hospitals(
            aoi=aoi,
            osm_geojson_path=request.osm_geojson_path,
            settlements_path=request.settlements_geojson_path,
            hospitals_path=request.hospitals_geojson_path,
            geo_res=geo_res,
        )

        # 6. Parse and Snap Settlements & Hospitals to Road Network Nodes
        settlement_objs = self._parse_and_snap_settlements(
            features=settlements_features,
            graph=baseline_graph,
            mean_lat=mean_lat,
        )

        hospital_objs = self._parse_and_snap_hospitals(
            features=hospitals_features,
            graph=baseline_graph,
            mean_lat=mean_lat,
        )

        # 7. Destination Assignment, Reachability & Detour Analysis
        settlement_results, isolated_features, all_settlements_features = self._evaluate_settlements_connectivity(
            settlements=settlement_objs,
            hospitals=hospital_objs,
            baseline_graph=baseline_graph,
            post_flood_graph=post_flood_graph,
            detour_minor_pct=request.detour_minor_threshold_pct,
            detour_major_pct=request.detour_major_threshold_pct,
        )

        # 8. Hospital Access Lost GeoJSON layer
        hosp_lost_features = [
            f for f in all_settlements_features
            if f.get("properties", {}).get("hospital_access_lost", False)
        ]

        # 9. Graph Connected Components Analysis
        baseline_components = list(nx.weakly_connected_components(baseline_graph)) if baseline_graph.number_of_nodes() > 0 else []
        post_flood_components = list(nx.weakly_connected_components(post_flood_graph)) if post_flood_graph.number_of_nodes() > 0 else []
        newly_disconnected = max(0, len(post_flood_components) - len(baseline_components))

        # 10. Compute Aggregated Summaries
        open_edges_count = sum(1 for e in road_edges if e.status == RoadEdgeStatus.ESTIMATED_OPEN)
        partially_affected_edges_count = sum(1 for e in road_edges if e.status == RoadEdgeStatus.PARTIALLY_AFFECTED)
        blocked_edges_count = sum(1 for e in road_edges if e.status == RoadEdgeStatus.ESTIMATED_BLOCKED)

        road_network_summary = RoadNetworkSummary(
            total_nodes=baseline_graph.number_of_nodes(),
            total_edges=len(road_edges),
            open_edges=open_edges_count,
            partially_affected_edges=partially_affected_edges_count,
            blocked_edges=blocked_edges_count,
            baseline_components=len(baseline_components),
            post_flood_components=len(post_flood_components),
            newly_disconnected_components=newly_disconnected,
        )

        settlement_summary = SettlementSummary(
            total=len(settlement_results),
            baseline_connected=sum(1 for s in settlement_results if s.baseline_connected),
            flood_isolated=sum(1 for s in settlement_results if s.status == SettlementIsolationStatus.FLOOD_ISOLATED),
            pre_existingly_disconnected=sum(1 for s in settlement_results if s.status == SettlementIsolationStatus.PRE_EXISTINGLY_DISCONNECTED),
            unaffected=sum(1 for s in settlement_results if s.status == SettlementIsolationStatus.UNAFFECTED and s.detour_classification == DetourClassification.NO_CHANGE),
            minor_detour=sum(1 for s in settlement_results if s.detour_classification == DetourClassification.MINOR_DETOUR),
            major_detour=sum(1 for s in settlement_results if s.detour_classification == DetourClassification.MAJOR_DETOUR),
        )

        hospital_summary = HospitalAccessSummary(
            settlements_with_baseline_access=sum(1 for s in settlement_results if s.baseline_hospital_access),
            access_preserved=sum(1 for s in settlement_results if s.hospital_access_status == HospitalAccessStatus.HOSPITAL_ACCESS_PRESERVED),
            hospital_access_lost=sum(1 for s in settlement_results if s.hospital_access_lost),
            no_baseline_access=sum(1 for s in settlement_results if s.hospital_access_status == HospitalAccessStatus.NO_BASELINE_HOSPITAL_ACCESS),
        )

        msg = (
            f"Module 6 connectivity analysis completed for job '{job_id}'. "
            f"Evaluated {len(settlement_results)} settlements ({settlement_summary.flood_isolated} flood-isolated, "
            f"{settlement_summary.minor_detour + settlement_summary.major_detour} detours). "
            f"Evaluated road network with {len(road_edges)} segments ({blocked_edges_count} estimated blocked, "
            f"{partially_affected_edges_count} partially affected). "
            f"Hospital accessibility: {hospital_summary.hospital_access_lost} settlements lost hospital access."
        )

        response = ConnectivityAnalysisResponse(
            job_id=job_id,
            status="completed",
            source_data_status=geo_res.source_data_status,
            crs="EPSG:4326",
            blocking_threshold_pct=request.road_blocked_threshold_pct,
            road_network=road_network_summary,
            settlements=settlement_summary,
            hospital_access=hospital_summary,
            settlement_details=settlement_results,
            hospital_details=hospital_objs,
            road_details=road_edges,
            blocked_roads={"type": "FeatureCollection", "features": blocked_road_features},
            partially_affected_roads={"type": "FeatureCollection", "features": partially_affected_road_features},
            isolated_settlements={"type": "FeatureCollection", "features": isolated_features},
            hospital_access_lost_settlements={"type": "FeatureCollection", "features": hosp_lost_features},
            settlements_connectivity={"type": "FeatureCollection", "features": all_settlements_features},
            attribution=[
                "© Copernicus Sentinel data",
                "© OpenStreetMap contributors",
                "Kuro Siwo dataset / Orion-AI-Lab",
            ],
            created_at=datetime.now(timezone.utc),
            message=msg,
        )

        # 11. Persist to disk and memory store
        self._persist_outputs(job_id, response)
        self._connectivity_store[job_id] = response
        return response

    def _build_road_graphs(
        self,
        roads: List[Dict[str, Any]],
        bridges: List[Dict[str, Any]],
        mean_lat: float,
        blocking_threshold_pct: float = 50.0,
    ) -> Tuple[
        nx.DiGraph,
        nx.DiGraph,
        List[RoadEdgeDetail],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """
        Construct topological baseline and post-flood road network graphs.
        
        Handles LineString, MultiLineString, shared endpoints/intersections,
        and directionality. Applies configurable blockage rules:
        - OPEN: affected_percentage == 0
        - PARTIALLY_AFFECTED: 0 < affected_percentage < blocking_threshold_pct
        - BLOCKED: affected_percentage >= blocking_threshold_pct
        """
        baseline_graph = nx.DiGraph()
        edge_details: List[RoadEdgeDetail] = []
        blocked_features: List[Dict[str, Any]] = []
        partially_affected_features: List[Dict[str, Any]] = []

        # Index bridges by osm_id / road_id for correlation
        bridge_map: Dict[str, Dict[str, Any]] = {}
        for b in bridges:
            b_props = b.get("properties", {})
            b_id = str(b.get("id", ""))
            osm_id = str(b_props.get("osm_id", ""))
            if osm_id:
                bridge_map[osm_id] = b
            bridge_map[b_id] = b

        # 1. Parse and segment each road feature into geometric LineStrings
        raw_segments: List[Dict[str, Any]] = []
        coord_counts: Dict[Tuple[float, float], int] = {}

        for r in roads:
            r_id = r.get("id", f"road_{len(raw_segments)+1}")
            props = r.get("properties", {})
            geom = r.get("geometry", {})
            osm_id = props.get("osm_id")
            highway_type = props.get("highway_type", "road")
            aff_pct = float(props.get("affected_percentage", 0.0))
            flooded_m = float(props.get("flooded_length_m", 0.0))
            is_bridge = bool(props.get("is_bridge", False))
            if str(osm_id) in bridge_map or r_id in bridge_map:
                is_bridge = True

            # Extract list of LineString coordinates
            coord_sequences: List[List[Tuple[float, float]]] = []
            if geom.get("type") == "LineString":
                coords = [(round(p[0], 6), round(p[1], 6)) for p in geom.get("coordinates", [])]
                if len(coords) >= 2:
                    coord_sequences.append(coords)
            elif geom.get("type") == "MultiLineString":
                for line_coords in geom.get("coordinates", []):
                    coords = [(round(p[0], 6), round(p[1], 6)) for p in line_coords]
                    if len(coords) >= 2:
                        coord_sequences.append(coords)

            for c_seq in coord_sequences:
                raw_segments.append({
                    "road_id": r_id,
                    "osm_id": osm_id,
                    "name": props.get("name"),
                    "highway_type": highway_type,
                    "coords": c_seq,
                    "affected_percentage": aff_pct,
                    "flooded_length_m": flooded_m,
                    "is_bridge": is_bridge,
                    "oneway": props.get("oneway"),
                })
                # Register endpoints and vertices
                coord_counts[c_seq[0]] = coord_counts.get(c_seq[0], 0) + 1
                coord_counts[c_seq[-1]] = coord_counts.get(c_seq[-1], 0) + 1
                for pt in c_seq[1:-1]:
                    coord_counts[pt] = coord_counts.get(pt, 0) + 1

        # 2. Break segments into graph edges between intersection nodes
        edge_counter = 0
        for seg in raw_segments:
            coords = seg["coords"]
            r_id = seg["road_id"]
            osm_id = seg["osm_id"]
            name = seg["name"]
            hw_type = seg["highway_type"]
            aff_pct = seg["affected_percentage"]
            flooded_m = seg["flooded_length_m"]
            is_bridge = seg["is_bridge"]
            oneway_tag = str(seg.get("oneway", "")).lower()

            # Find split indices (start, end, and any shared intersection vertex)
            split_indices = [0]
            for i in range(1, len(coords) - 1):
                if coord_counts.get(coords[i], 0) > 1:
                    split_indices.append(i)
            split_indices.append(len(coords) - 1)

            # De-duplicate split indices preserving order
            split_indices = sorted(list(set(split_indices)))

            for i in range(len(split_indices) - 1):
                sub_coords = coords[split_indices[i] : split_indices[i + 1] + 1]
                if len(sub_coords) < 2:
                    continue

                u_pt = sub_coords[0]
                v_pt = sub_coords[-1]
                u_id = f"node_{u_pt[0]:.6f}_{u_pt[1]:.6f}"
                v_id = f"node_{v_pt[0]:.6f}_{v_pt[1]:.6f}"

                # Calculate metric subsegment length
                line_sh = shapely.geometry.LineString(sub_coords)
                seg_len_m = calculate_linestring_metric_length(line_sh, mean_lat)
                if seg_len_m <= 0.1:
                    seg_len_m = 1.0  # minimal safe metric length

                # Edge classification & blockage determination
                if aff_pct <= 0.0:
                    status = RoadEdgeStatus.ESTIMATED_OPEN
                elif aff_pct < blocking_threshold_pct:
                    status = RoadEdgeStatus.PARTIALLY_AFFECTED
                else:
                    status = RoadEdgeStatus.ESTIMATED_BLOCKED

                edge_counter += 1
                edge_id = f"{r_id}_e{edge_counter}"

                edge_detail = RoadEdgeDetail(
                    edge_id=edge_id,
                    road_id=r_id,
                    osm_id=osm_id,
                    name=name,
                    highway_type=hw_type,
                    from_node=u_id,
                    to_node=v_id,
                    length_m=round(seg_len_m, 1),
                    flooded_length_m=round(flooded_m * (len(sub_coords) / max(2, len(coords))), 1),
                    affected_percentage=aff_pct,
                    is_bridge=is_bridge,
                    status=status,
                    geometry={"type": "LineString", "coordinates": [[pt[0], pt[1]] for pt in sub_coords]},
                )
                edge_details.append(edge_detail)

                # Add nodes to graph with geographic coordinate attributes
                if not baseline_graph.has_node(u_id):
                    baseline_graph.add_node(u_id, lon=u_pt[0], lat=u_pt[1])
                if not baseline_graph.has_node(v_id):
                    baseline_graph.add_node(v_id, lon=v_pt[0], lat=v_pt[1])

                # Edge attributes dictionary for NetworkX
                attrs = {
                    "edge_id": edge_id,
                    "road_id": r_id,
                    "osm_id": osm_id,
                    "name": name,
                    "highway_type": hw_type,
                    "length_m": seg_len_m,
                    "affected_percentage": aff_pct,
                    "status": status.value,
                    "is_bridge": is_bridge,
                }

                # Handle directionality: default undirected (bidirectional)
                if oneway_tag in ["yes", "true", "1"]:
                    baseline_graph.add_edge(u_id, v_id, **attrs)
                elif oneway_tag == "-1":
                    baseline_graph.add_edge(v_id, u_id, **attrs)
                else:
                    baseline_graph.add_edge(u_id, v_id, **attrs)
                    baseline_graph.add_edge(v_id, u_id, **attrs)

                # GeoJSON features for blocked and partially affected segments
                feature_dict = {
                    "type": "Feature",
                    "id": edge_id,
                    "properties": {
                        "road_id": r_id,
                        "edge_id": edge_id,
                        "osm_id": osm_id,
                        "name": name,
                        "road_class": hw_type,
                        "length_m": round(seg_len_m, 1),
                        "affected_length_m": round(flooded_m, 1),
                        "affected_percentage": aff_pct,
                        "blockage_status": status.value,
                        "is_bridge": is_bridge,
                    },
                    "geometry": edge_detail.geometry,
                }

                if status == RoadEdgeStatus.ESTIMATED_BLOCKED:
                    blocked_features.append(feature_dict)
                elif status == RoadEdgeStatus.PARTIALLY_AFFECTED:
                    partially_affected_features.append(feature_dict)

        # 3. Construct post-flood graph by copying baseline and removing blocked edges
        post_flood_graph = baseline_graph.copy()
        edges_to_remove = []
        for u, v, data in post_flood_graph.edges(data=True):
            if data.get("status") == RoadEdgeStatus.ESTIMATED_BLOCKED.value:
                edges_to_remove.append((u, v))

        for u, v in edges_to_remove:
            if post_flood_graph.has_edge(u, v):
                post_flood_graph.remove_edge(u, v)

        logger.info(
            "Road graph constructed: %d nodes, %d directed edges (%d estimated blocked removed).",
            baseline_graph.number_of_nodes(),
            baseline_graph.number_of_edges(),
            len(edges_to_remove),
        )

        return (
            baseline_graph,
            post_flood_graph,
            edge_details,
            blocked_features,
            partially_affected_features,
        )

    def _acquire_settlements_and_hospitals(
        self,
        aoi: Dict[str, Any],
        osm_geojson_path: Optional[str] = None,
        settlements_path: Optional[str] = None,
        hospitals_path: Optional[str] = None,
        geo_res: Optional[GeospatialAnalysisResponse] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Acquire pre-event settlements and hospitals from local files, cache, or Overpass API."""
        settlements: List[Dict[str, Any]] = []
        hospitals: List[Dict[str, Any]] = []

        # 1. Check if separate settlements file provided
        if settlements_path:
            _, s_list, _ = self.osm_service.fetch_pre_event_settlements_and_hospitals(
                aoi=aoi, custom_file_path=settlements_path
            )
            settlements.extend(s_list)

        # 2. Check if separate hospitals file provided
        if hospitals_path:
            _, _, h_list = self.osm_service.fetch_pre_event_settlements_and_hospitals(
                aoi=aoi, custom_file_path=hospitals_path
            )
            hospitals.extend(h_list)

        # 3. Main pre-event dataset (osm_geojson_path) or query
        if not settlements or not hospitals:
            _, s_main, h_main = self.osm_service.fetch_pre_event_settlements_and_hospitals(
                aoi=aoi, custom_file_path=osm_geojson_path
            )
            if not settlements:
                settlements.extend(s_main)
            if not hospitals:
                hospitals.extend(h_main)

        # 4. Fallback for hospitals from Module 5 buildings if not yet extracted
        if not hospitals and geo_res and geo_res.buildings:
            for b in geo_res.buildings.get("features", []):
                props = b.get("properties", {})
                b_type = str(props.get("building_type", "")).lower()
                amenity = str(props.get("amenity", "")).lower()
                if b_type == "hospital" or amenity == "hospital":
                    geom = b.get("geometry", {})
                    try:
                        sh_geom = shapely.geometry.shape(geom)
                        c = sh_geom.centroid
                        hospitals.append({
                            "type": "Feature",
                            "id": f"osm_hospital_{b.get('id', len(hospitals)+1)}",
                            "properties": {
                                "hospital_id": f"osm_hospital_{b.get('id', len(hospitals)+1)}",
                                "osm_id": props.get("osm_id"),
                                "name": props.get("name"),
                                "amenity_type": "hospital",
                                "osm_tags": props,
                            },
                            "geometry": {"type": "Point", "coordinates": [round(c.x, 6), round(c.y, 6)]},
                        })
                    except Exception:
                        pass

        return settlements, hospitals

    def _snap_point_to_nearest_node(
        self,
        lon: float,
        lat: float,
        graph: nx.DiGraph,
        mean_lat: float,
    ) -> Tuple[Optional[str], Optional[float]]:
        """Snap a geographic coordinate to the closest road network node in meters."""
        if graph.number_of_nodes() == 0:
            return None, None

        m_lat, m_lon = compute_geodesic_scales(mean_lat)
        best_node = None
        best_dist = float("inf")

        for n, data in graph.nodes(data=True):
            n_lon = data.get("lon")
            n_lat = data.get("lat")
            if n_lon is None or n_lat is None:
                continue
            dx = (n_lon - lon) * m_lon
            dy = (n_lat - lat) * m_lat
            dist_m = math.hypot(dx, dy)
            if dist_m < best_dist:
                best_dist = dist_m
                best_node = n

        return best_node, round(best_dist, 1) if best_node else None

    def _parse_and_snap_settlements(
        self,
        features: List[Dict[str, Any]],
        graph: nx.DiGraph,
        mean_lat: float,
    ) -> List[Dict[str, Any]]:
        """Parse settlement features and snap to the nearest road network node."""
        settlements = []
        for idx, f in enumerate(features):
            props = f.get("properties", {})
            geom = f.get("geometry", {})
            coords = geom.get("coordinates", [])
            if len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            s_id = str(props.get("settlement_id") or f.get("id") or f"settlement_{idx+1}")
            nearest_n, snap_dist_m = self._snap_point_to_nearest_node(lon, lat, graph, mean_lat)

            settlements.append({
                "settlement_id": s_id,
                "name": props.get("name"),
                "latitude": lat,
                "longitude": lon,
                "place_type": str(props.get("place_type", "settlement")).lower(),
                "population": props.get("population"),
                "nearest_node_id": nearest_n,
                "distance_to_nearest_node_m": snap_dist_m,
                "geometry": geom,
            })
        return settlements

    def _parse_and_snap_hospitals(
        self,
        features: List[Dict[str, Any]],
        graph: nx.DiGraph,
        mean_lat: float,
    ) -> List[HospitalDetail]:
        """Parse hospital facilities and snap to the nearest road network node."""
        hospitals = []
        for idx, f in enumerate(features):
            props = f.get("properties", {})
            geom = f.get("geometry", {})
            coords = geom.get("coordinates", [])
            if len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            h_id = str(props.get("hospital_id") or f.get("id") or f"hospital_{idx+1}")
            nearest_n, snap_dist_m = self._snap_point_to_nearest_node(lon, lat, graph, mean_lat)

            hospitals.append(
                HospitalDetail(
                    hospital_id=h_id,
                    name=props.get("name"),
                    latitude=lat,
                    longitude=lon,
                    amenity_type=str(props.get("amenity_type", "hospital")),
                    nearest_node_id=nearest_n,
                    distance_to_nearest_node_m=snap_dist_m,
                    osm_tags=props.get("osm_tags", {}),
                )
            )
        return hospitals

    def _evaluate_settlements_connectivity(
        self,
        settlements: List[Dict[str, Any]],
        hospitals: List[HospitalDetail],
        baseline_graph: nx.DiGraph,
        post_flood_graph: nx.DiGraph,
        detour_minor_pct: float = 5.0,
        detour_major_pct: float = 25.0,
    ) -> Tuple[
        List[SettlementConnectivityDetail],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """
        Determine destination, baseline vs post-flood route connectivity,
        detour metrics, and hospital access for each settlement.
        """
        details: List[SettlementConnectivityDetail] = []
        isolated_features: List[Dict[str, Any]] = []
        all_features: List[Dict[str, Any]] = []

        for s in settlements:
            s_id = s["settlement_id"]
            s_name = s.get("name")
            s_lat = s["latitude"]
            s_lon = s["longitude"]
            s_node = s["nearest_node_id"]

            # 1. Select destination (nearest town/city or major connected settlement)
            dest, reason = self._select_destination(s, settlements, baseline_graph)
            d_id = dest["settlement_id"] if dest else None
            d_name = dest.get("name") if dest else None
            d_node = dest["nearest_node_id"] if dest else None

            # 2. Baseline Connectivity Analysis
            baseline_connected = False
            baseline_distance_km = None

            if s_node and d_node:
                if s_node == d_node:
                    baseline_connected = True
                    baseline_distance_km = 0.0
                elif baseline_graph.has_node(s_node) and baseline_graph.has_node(d_node):
                    if nx.has_path(baseline_graph, s_node, d_node):
                        dist_m = nx.shortest_path_length(baseline_graph, s_node, d_node, weight="length_m")
                        baseline_connected = True
                        baseline_distance_km = round(dist_m / 1000.0, 3)

            # 3. Post-Flood Connectivity & Isolation Classification
            post_flood_connected = False
            post_flood_distance_km = None
            additional_dist_km = None
            detour_pct = None

            if not baseline_connected:
                # Pre-existing disconnect must NEVER be blamed on the flood
                status = SettlementIsolationStatus.PRE_EXISTINGLY_DISCONNECTED
                detour_class = DetourClassification.DISCONNECTED
            else:
                if s_node == d_node:
                    post_flood_connected = True
                    post_flood_distance_km = 0.0
                    additional_dist_km = 0.0
                    detour_pct = 0.0
                    status = SettlementIsolationStatus.UNAFFECTED
                    detour_class = DetourClassification.NO_CHANGE
                elif post_flood_graph.has_node(s_node) and post_flood_graph.has_node(d_node) and nx.has_path(post_flood_graph, s_node, d_node):
                    post_dist_m = nx.shortest_path_length(post_flood_graph, s_node, d_node, weight="length_m")
                    post_flood_connected = True
                    post_flood_distance_km = round(post_dist_m / 1000.0, 3)
                    status = SettlementIsolationStatus.UNAFFECTED

                    # Road detour calculation
                    additional_dist_km = round(max(0.0, post_flood_distance_km - (baseline_distance_km or 0.0)), 3)
                    if baseline_distance_km and baseline_distance_km > 0.0:
                        detour_pct = round((additional_dist_km / baseline_distance_km) * 100.0, 2)
                    else:
                        detour_pct = 0.0

                    if detour_pct <= detour_minor_pct:
                        detour_class = DetourClassification.NO_CHANGE
                    elif detour_pct <= detour_major_pct:
                        detour_class = DetourClassification.MINOR_DETOUR
                    else:
                        detour_class = DetourClassification.MAJOR_DETOUR
                else:
                    post_flood_connected = False
                    status = SettlementIsolationStatus.FLOOD_ISOLATED
                    detour_class = DetourClassification.DISCONNECTED

            # 4. Hospital Accessibility Analysis
            hosp_eval = self._evaluate_hospital_access(s_node, hospitals, baseline_graph, post_flood_graph)

            # 5. Build Structured Settlement Detail
            detail = SettlementConnectivityDetail(
                settlement_id=s_id,
                name=s_name,
                latitude=s_lat,
                longitude=s_lon,
                place_type=s["place_type"],
                population=s["population"],
                nearest_node_id=s_node,
                distance_to_nearest_node_m=s["distance_to_nearest_node_m"],
                destination_id=d_id,
                destination_name=d_name,
                destination_selection_reason=reason,
                baseline_connected=baseline_connected,
                post_flood_connected=post_flood_connected,
                status=status,
                baseline_distance_km=baseline_distance_km,
                post_flood_distance_km=post_flood_distance_km,
                additional_distance_km=additional_dist_km,
                detour_percentage=detour_pct,
                detour_classification=detour_class,
                nearest_hospital_id=hosp_eval["hospital_id"],
                nearest_hospital_name=hosp_eval["hospital_name"],
                baseline_hospital_access=hosp_eval["baseline_access"],
                post_flood_hospital_access=hosp_eval["post_flood_access"],
                hospital_access_lost=hosp_eval["access_lost"],
                hospital_access_status=hosp_eval["status"],
                baseline_hospital_distance_km=hosp_eval["baseline_distance_km"],
                post_flood_hospital_distance_km=hosp_eval["post_flood_distance_km"],
            )
            details.append(detail)

            # Feature representation
            feat = {
                "type": "Feature",
                "id": s_id,
                "properties": detail.model_dump(),
                "geometry": {"type": "Point", "coordinates": [s_lon, s_lat]},
            }
            all_features.append(feat)

            if status == SettlementIsolationStatus.FLOOD_ISOLATED:
                isolated_features.append(feat)

        return details, isolated_features, all_features

    def _select_destination(
        self,
        current: Dict[str, Any],
        all_settlements: List[Dict[str, Any]],
        baseline_graph: nx.DiGraph,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Determine the primary destination town/city for a settlement.
        Prioritizes connected towns/cities, then major settlements.
        """
        c_id = current["settlement_id"]
        c_node = current["nearest_node_id"]

        # If current is the only settlement
        if len(all_settlements) <= 1:
            return current, "Single settlement in AOI (administrative center)"

        other_settlements = [s for s in all_settlements if s["settlement_id"] != c_id]
        if not other_settlements:
            return current, "Single settlement in AOI"

        # Separate into towns/cities vs other settlements
        urban_candidates = [
            s for s in other_settlements
            if s.get("place_type") in ["city", "town"]
        ]
        candidates = urban_candidates if urban_candidates else other_settlements

        # 1. Search for reachable candidate with shortest baseline road distance
        reachable_candidates: List[Tuple[Dict[str, Any], float]] = []
        if c_node and baseline_graph.has_node(c_node):
            for cand in candidates:
                cand_node = cand["nearest_node_id"]
                if cand_node and baseline_graph.has_node(cand_node):
                    if nx.has_path(baseline_graph, c_node, cand_node):
                        dist_m = nx.shortest_path_length(baseline_graph, c_node, cand_node, weight="length_m")
                        reachable_candidates.append((cand, dist_m))

        if reachable_candidates:
            reachable_candidates.sort(key=lambda x: x[1])
            best_cand = reachable_candidates[0][0]
            reason = (
                "Nearest connected town/city in baseline road network"
                if best_cand.get("place_type") in ["city", "town"]
                else "Nearest connected major settlement in baseline road network"
            )
            return best_cand, reason

        # 2. If no road path exists, choose geographically nearest candidate
        c_lon, c_lat = current["longitude"], current["latitude"]
        geo_sorted = sorted(
            candidates,
            key=lambda s: math.hypot(s["longitude"] - c_lon, s["latitude"] - c_lat),
        )
        nearest_geo = geo_sorted[0]
        reason = (
            "Nearest town/city (unconnected in baseline road network)"
            if nearest_geo.get("place_type") in ["city", "town"]
            else "Nearest settlement (unconnected in baseline road network)"
        )
        return nearest_geo, reason

    def _evaluate_hospital_access(
        self,
        s_node: Optional[str],
        hospitals: List[HospitalDetail],
        baseline_graph: nx.DiGraph,
        post_flood_graph: nx.DiGraph,
    ) -> Dict[str, Any]:
        """
        Evaluate reachability to healthcare facilities before and after flood.
        Guarantees that pre-existing hospital disconnect is not falsely attributed to the flood.
        """
        if not hospitals or not s_node or not baseline_graph.has_node(s_node):
            return {
                "hospital_id": None,
                "hospital_name": None,
                "baseline_access": False,
                "post_flood_access": False,
                "access_lost": False,
                "status": HospitalAccessStatus.NO_BASELINE_HOSPITAL_ACCESS,
                "baseline_distance_km": None,
                "post_flood_distance_km": None,
            }

        # 1. Search for baseline reachable hospitals
        baseline_reachable: List[Tuple[HospitalDetail, float]] = []
        for h in hospitals:
            h_node = h.nearest_node_id
            if h_node and baseline_graph.has_node(h_node) and nx.has_path(baseline_graph, s_node, h_node):
                dist_m = nx.shortest_path_length(baseline_graph, s_node, h_node, weight="length_m")
                baseline_reachable.append((h, dist_m))

        if not baseline_reachable:
            # Settlement had no road route to any hospital before the flood
            # Reference the closest hospital by straight-line distance
            return {
                "hospital_id": hospitals[0].hospital_id if hospitals else None,
                "hospital_name": hospitals[0].name if hospitals else None,
                "baseline_access": False,
                "post_flood_access": False,
                "access_lost": False,
                "status": HospitalAccessStatus.NO_BASELINE_HOSPITAL_ACCESS,
                "baseline_distance_km": None,
                "post_flood_distance_km": None,
            }

        # Settlement has baseline access! Find closest hospital
        baseline_reachable.sort(key=lambda x: x[1])
        best_hosp, best_dist_m = baseline_reachable[0]
        best_dist_km = round(best_dist_m / 1000.0, 3)

        # 2. Check post-flood reachability to any healthcare facility
        post_reachable: List[Tuple[HospitalDetail, float]] = []
        if post_flood_graph.has_node(s_node):
            for h in hospitals:
                h_node = h.nearest_node_id
                if h_node and post_flood_graph.has_node(h_node) and nx.has_path(post_flood_graph, s_node, h_node):
                    p_dist_m = nx.shortest_path_length(post_flood_graph, s_node, h_node, weight="length_m")
                    post_reachable.append((h, p_dist_m))

        if post_reachable:
            post_reachable.sort(key=lambda x: x[1])
            p_hosp, p_dist_m = post_reachable[0]
            return {
                "hospital_id": p_hosp.hospital_id,
                "hospital_name": p_hosp.name,
                "baseline_access": True,
                "post_flood_access": True,
                "access_lost": False,
                "status": HospitalAccessStatus.HOSPITAL_ACCESS_PRESERVED,
                "baseline_distance_km": best_dist_km,
                "post_flood_distance_km": round(p_dist_m / 1000.0, 3),
            }
        else:
            return {
                "hospital_id": best_hosp.hospital_id,
                "hospital_name": best_hosp.name,
                "baseline_access": True,
                "post_flood_access": False,
                "access_lost": True,
                "status": HospitalAccessStatus.HOSPITAL_ACCESS_LOST,
                "baseline_distance_km": best_dist_km,
                "post_flood_distance_km": None,
            }

    def _build_blocked_case_study_response(
        self,
        job_id: str,
        geo_res: GeospatialAnalysisResponse,
        request: ConnectivityAnalysisRequest,
    ) -> ConnectivityAnalysisResponse:
        """Construct deterministic blocker response for Nepal 2026 without fabricating claims."""
        empty_fc = {"type": "FeatureCollection", "features": []}
        msg = (
            f"Module 6 road network connectivity analysis evaluated for job '{job_id}'. "
            "2026 CASE STUDY: BLOCKED — waiting for actual Sentinel binary data. "
            "Road network accessibility and settlement isolation cannot be computed without physical "
            "satellite inundation observations. Zero road segments blocked; zero settlements isolated."
        )

        response = ConnectivityAnalysisResponse(
            job_id=job_id,
            status="satellite_binary_data_blocked",
            source_data_status="satellite_binary_data_blocked",
            crs="EPSG:4326",
            blocking_threshold_pct=request.road_blocked_threshold_pct,
            road_network=RoadNetworkSummary(
                total_nodes=0,
                total_edges=0,
                open_edges=0,
                partially_affected_edges=0,
                blocked_edges=0,
                baseline_components=0,
                post_flood_components=0,
                newly_disconnected_components=0,
            ),
            settlements=SettlementSummary(
                total=0,
                baseline_connected=0,
                flood_isolated=0,
                pre_existingly_disconnected=0,
                unaffected=0,
                minor_detour=0,
                major_detour=0,
            ),
            hospital_access=HospitalAccessSummary(
                settlements_with_baseline_access=0,
                access_preserved=0,
                hospital_access_lost=0,
                no_baseline_access=0,
            ),
            settlement_details=[],
            hospital_details=[],
            road_details=[],
            blocked_roads=empty_fc,
            partially_affected_roads=empty_fc,
            isolated_settlements=empty_fc,
            hospital_access_lost_settlements=empty_fc,
            settlements_connectivity=empty_fc,
            attribution=[
                "© Copernicus Sentinel data (2026)",
                "© OpenStreetMap contributors",
                "Kuro Siwo dataset / Orion-AI-Lab",
            ],
            created_at=datetime.now(timezone.utc),
            message=msg,
        )

        self._persist_outputs(job_id, response)
        self._connectivity_store[job_id] = response
        return response

    def _persist_outputs(self, job_id: str, res: ConnectivityAnalysisResponse) -> None:
        """Persist structured JSON summary and GeoJSON feature layers to disk."""
        job_dir = self.outputs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        with open(job_dir / "connectivity_summary.json", "w", encoding="utf-8") as f:
            json.dump(res.model_dump(mode="json"), f, indent=2)

        layers = [
            ("blocked_roads.geojson", res.blocked_roads),
            ("partially_affected_roads.geojson", res.partially_affected_roads),
            ("isolated_settlements.geojson", res.isolated_settlements),
            ("hospital_access_lost_settlements.geojson", res.hospital_access_lost_settlements),
            ("settlements_connectivity.geojson", res.settlements_connectivity),
        ]
        for filename, geojson_data in layers:
            with open(job_dir / filename, "w", encoding="utf-8") as f:
                json.dump(geojson_data, f, indent=2)

    def get_connectivity_result(self, job_id: str) -> Optional[ConnectivityAnalysisResponse]:
        """Retrieve stored connectivity analysis result from memory or disk."""
        if job_id in self._connectivity_store:
            return self._connectivity_store[job_id]

        summary_file = self.outputs_dir / job_id / "connectivity_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                res = ConnectivityAnalysisResponse(**data)
                self._connectivity_store[job_id] = res
                return res
            except Exception as e:
                logger.error("Failed to load connectivity summary %s: %s", summary_file, e)

        return None

    def get_layer_geojson(self, job_id: str, layer_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific GeoJSON layer for a job."""
        layer_file = self.outputs_dir / job_id / f"{layer_name}.geojson"
        if layer_file.exists():
            try:
                with open(layer_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to read layer %s: %s", layer_file, e)
        return None


# Global singleton instance
_connectivity_service_instance: Optional[ConnectivityAnalysisService] = None


def get_connectivity_service() -> ConnectivityAnalysisService:
    """Dependency provider for ConnectivityAnalysisService."""
    global _connectivity_service_instance
    if _connectivity_service_instance is None:
        _connectivity_service_instance = ConnectivityAnalysisService()
    return _connectivity_service_instance
