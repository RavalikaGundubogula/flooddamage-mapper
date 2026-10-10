import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import shapes
import shapely.geometry
import shapely.ops

from app.core.config import Settings, get_settings
from app.schemas.geospatial import (
    AssetSummary,
    BridgeAssetSummary,
    BridgeExposureClassification,
    BridgeImpact,
    BuildingAssetSummary,
    BuildingExposureClassification,
    BuildingImpact,
    ChangeZoneMetrics,
    ChangeZoneProperties,
    FloodExtentMetrics,
    FloodRegionProperties,
    GeospatialAnalysisRequest,
    GeospatialAnalysisResponse,
    RoadAssetSummary,
    RoadExposureClassification,
    RoadImpact,
)
from app.services.analysis_service import AnalysisRepository, get_analysis_service
from app.services.ard_service import AnalysisReadyDataService, get_ard_service
from app.services.osm_service import OpenStreetMapService, get_osm_service
from app.services.segmentation_service import SegmentationInferenceService, get_segmentation_service

logger = logging.getLogger(__name__)


def compute_geodesic_scales(mean_lat: float) -> Tuple[float, float]:
    """
    Compute metric conversion scales (meters per degree) at a given latitude.
    m_lat: ~111,320 m/deg
    m_lon: ~111,320 * cos(lat) m/deg
    """
    rad = math.radians(mean_lat)
    m_lat = 111320.0
    m_lon = max(100.0, 111320.0 * math.cos(rad))
    return m_lat, m_lon


def calculate_polygon_metric_area(geom: shapely.geometry.base.BaseGeometry, mean_lat: float) -> float:
    """Calculate geographic polygon area in square meters using spherical latitude scaling."""
    if geom.is_empty:
        return 0.0
    m_lat, m_lon = compute_geodesic_scales(mean_lat)
    # Scale polygon coordinates to planar meters centered at centroid
    c_x, c_y = geom.centroid.x, geom.centroid.y
    scaled_geom = shapely.affinity.scale(geom, xfact=m_lon, yfact=m_lat, origin=(c_x, c_y))
    return float(max(0.0, scaled_geom.area))


def calculate_linestring_metric_length(geom: shapely.geometry.base.BaseGeometry, mean_lat: float) -> float:
    """Calculate line length in meters using spherical latitude scaling."""
    if geom.is_empty:
        return 0.0
    m_lat, m_lon = compute_geodesic_scales(mean_lat)
    total_len = 0.0
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
        for i in range(len(coords) - 1):
            dx = (coords[i + 1][0] - coords[i][0]) * m_lon
            dy = (coords[i + 1][1] - coords[i][1]) * m_lat
            total_len += math.hypot(dx, dy)
    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            total_len += calculate_linestring_metric_length(part, mean_lat)
    return float(total_len)


class GeospatialDamageAnalysisService:
    """
    Module 5: Geospatial Infrastructure Damage & Exposure Overlay.
    
    Transforms AI segmentation predictions into structured geospatial impact metrics:
    1. Polygonizes connected flood extents and computes geodesic surface areas.
    2. Identifies possible terrain change / debris candidate zones from SAR differencing.
    3. Acquires and caches pre-event OpenStreetMap infrastructure (buildings, roads, bridges).
    4. Evaluates spatial exposure of buildings (overlap %), roads (flooded length), and bridges.
    5. Delivers structured GeoJSON layers and damage summary tables.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        seg_service: Optional[SegmentationInferenceService] = None,
        ard_service: Optional[AnalysisReadyDataService] = None,
        osm_service: Optional[OpenStreetMapService] = None,
        repo: Optional[AnalysisRepository] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.seg_service = seg_service or get_segmentation_service()
        self.ard_service = ard_service or get_ard_service()
        self.osm_service = osm_service or get_osm_service()
        self.repo = repo or get_analysis_service().repository

        self.outputs_dir = self.settings.get_data_path(self.settings.DATA_OUTPUTS_DIR)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._geospatial_store: Dict[str, GeospatialAnalysisResponse] = {}

    def analyze_job_impact(self, request: GeospatialAnalysisRequest) -> GeospatialAnalysisResponse:
        """
        Execute comprehensive Module 5 geospatial asset damage analysis for a job.
        """
        job_id = request.job_id
        if not job_id:
            raise ValueError("job_id must be provided to run geospatial analysis.")

        # 1. Retrieve Module 4 AI segmentation results
        seg_res = self.seg_service.get_segmentation_result(job_id)
        if not seg_res:
            raise ValueError(
                f"No AI segmentation output found for job '{job_id}'. "
                f"Please run POST /analysis/{job_id}/segment first."
            )

        # 2. Retrieve ARD package to check physical satellite data status
        ard_pkg = self.ard_service.get_ard_by_job_id(job_id)
        source_data_status = "calibrated_raster_available"
        if (ard_pkg and ard_pkg.status == "contract_ready") or (seg_res and getattr(seg_res, "status", None) == "blocked"):
            source_data_status = "satellite_binary_data_blocked"
        elif "test" in job_id.lower() or "fixture" in job_id.lower():
            source_data_status = "test_fixture"

        mask_raster_path = Path(seg_res.mask_raster_path)
        if not mask_raster_path.exists():
            raise ValueError(f"Flood mask raster file missing from disk: {mask_raster_path}")

        # 3. Read raster mask and calculate precise geodesic flood extents
        flood_metrics, flood_polygons, flood_geojson = self._extract_flood_polygons(
            mask_raster_path=mask_raster_path,
            job_id=job_id,
            min_area_m2=request.min_flood_region_area_m2,
        )

        # 4. Compute heuristic terrain change / debris candidate zones
        change_metrics, change_geojson = self._analyze_change_zones(
            ard_pkg=ard_pkg,
            flood_polygons=flood_polygons,
            job_id=job_id,
        )

        # 5. Acquire Pre-event OpenStreetMap infrastructure assets
        aoi = ard_pkg.aoi if ard_pkg else {"type": "Polygon", "coordinates": [[[seg_res.spatial_bounds[0], seg_res.spatial_bounds[1]], [seg_res.spatial_bounds[2], seg_res.spatial_bounds[1]], [seg_res.spatial_bounds[2], seg_res.spatial_bounds[3]], [seg_res.spatial_bounds[0], seg_res.spatial_bounds[3]], [seg_res.spatial_bounds[0], seg_res.spatial_bounds[1]]]]}
        osm_status, osm_buildings, osm_roads, osm_bridges = self.osm_service.fetch_pre_event_assets(
            aoi=aoi,
            custom_file_path=request.osm_geojson_path,
        )

        # 6. Evaluate infrastructure impact
        mean_lat = (seg_res.spatial_bounds[1] + seg_res.spatial_bounds[3]) / 2.0
        buildings_impact, bldg_geojson, bldg_summary = self._evaluate_buildings(
            buildings=osm_buildings,
            flood_polygons=flood_polygons,
            mean_lat=mean_lat,
            pot_thresh=request.building_affected_threshold_pct,
            heavy_thresh=request.building_heavy_threshold_pct,
        )

        roads_impact, roads_geojson, road_summary = self._evaluate_roads(
            roads=osm_roads,
            flood_polygons=flood_polygons,
            mean_lat=mean_lat,
            pot_thresh=request.road_blocked_threshold_pct,
            heavy_thresh=request.road_heavy_threshold_pct,
        )

        bridges_impact, bridge_geojson, bridge_summary = self._evaluate_bridges(
            bridges=osm_bridges,
            flood_polygons=flood_polygons,
            mean_lat=mean_lat,
            pot_thresh=request.bridge_affected_threshold_pct,
            heavy_thresh=request.bridge_heavy_threshold_pct,
        )

        asset_summary = AssetSummary(
            buildings=bldg_summary,
            roads=road_summary,
            bridges=bridge_summary,
        )

        # 7. Formulate operational status message
        overall_status = "completed"
        if source_data_status == "satellite_binary_data_blocked":
            overall_status = "satellite_binary_data_blocked"
            msg = (
                f"Module 5 geospatial asset analysis evaluated for job '{job_id}'. "
                "2026 CASE STUDY: BLOCKED — waiting for actual Sentinel binary data. "
                "Real 2026 Sentinel binary downloads are currently blocked awaiting CDSE credentials; "
                "analysis reflects spatial contract specifications. Zero assets damaged."
            )
        else:
            msg = (
                f"Module 5 geospatial damage analysis completed for job '{job_id}'. "
                f"Evaluated {bldg_summary.total} buildings ({bldg_summary.potentially_affected + bldg_summary.heavily_affected} affected), "
                f"{road_summary.total_segments} road segments ({road_summary.potentially_blocked + road_summary.heavily_affected} affected), "
                f"and {bridge_summary.total} bridges ({bridge_summary.potentially_affected + bridge_summary.heavily_affected} affected)."
            )

        response = GeospatialAnalysisResponse(
            job_id=job_id,
            status=overall_status,
            source_data_status=source_data_status,
            osm_status=osm_status,
            aoi=aoi,
            crs=seg_res.crs,
            flood_metrics=flood_metrics,
            change_zone_metrics=change_metrics,
            asset_summary=asset_summary,
            flood_regions=flood_geojson,
            change_zones=change_geojson,
            buildings=bldg_geojson,
            roads=roads_geojson,
            bridges=bridge_geojson,
            attribution=[
                "© Copernicus Sentinel data (2026)",
                "© OpenStreetMap contributors",
                "Kuro Siwo dataset / Orion-AI-Lab",
            ],
            created_at=datetime.now(timezone.utc),
            message=msg,
        )

        # Persist response and separate GeoJSON layers to disk
        self._persist_outputs(job_id, response)
        self._geospatial_store[job_id] = response
        return response

    def _extract_flood_polygons(
        self,
        mask_raster_path: Path,
        job_id: str,
        min_area_m2: float = 100.0,
    ) -> Tuple[FloodExtentMetrics, List[Tuple[str, shapely.geometry.base.BaseGeometry, float]], Dict[str, Any]]:
        """Polygonize GeoTIFF mask, compute metric areas, and filter small noise clusters."""
        with rasterio.open(mask_raster_path) as src:
            mask_arr = src.read(1)
            transform = src.transform
            bounds = src.bounds
            crs_name = src.crs.to_string() if src.crs else "EPSG:4326"

        mean_lat = (bounds.bottom + bounds.top) / 2.0
        m_lat, m_lon = compute_geodesic_scales(mean_lat)

        pixel_w_m = abs(transform[0]) * m_lon
        pixel_h_m = abs(transform[4]) * m_lat
        pixel_area_m2 = pixel_w_m * pixel_h_m

        total_pixels = int(mask_arr.size)
        flooded_pixels = int(np.sum(mask_arr == 1))
        flooded_area_m2 = float(flooded_pixels * pixel_area_m2)
        flooded_area_km2 = float(flooded_area_m2 / 1_000_000.0)
        total_aoi_area_km2 = float((total_pixels * pixel_area_m2) / 1_000_000.0)
        flood_pct = round(float((flooded_pixels / max(1, total_pixels)) * 100.0), 3)

        # Polygonize connected regions
        flood_polys = []
        features = []
        region_areas = []

        if flooded_pixels > 0:
            extracted_shapes = shapes(mask_arr, mask=(mask_arr == 1), transform=transform)
            idx = 1
            for geom_dict, val in extracted_shapes:
                if int(val) != 1:
                    continue
                sh_geom = shapely.geometry.shape(geom_dict)
                if not sh_geom.is_valid:
                    sh_geom = sh_geom.buffer(0)
                if sh_geom.is_empty:
                    continue

                poly_area_m2 = calculate_polygon_metric_area(sh_geom, mean_lat)
                if poly_area_m2 < min_area_m2:
                    continue  # Filter out tiny pixel noise

                poly_area_km2 = float(poly_area_m2 / 1_000_000.0)
                region_id = f"flood_region_{idx:03d}"
                idx += 1

                flood_polys.append((region_id, sh_geom, poly_area_m2))
                region_areas.append(poly_area_km2)

                c = sh_geom.centroid
                b = sh_geom.bounds  # minx, miny, maxx, maxy
                prop = FloodRegionProperties(
                    region_id=region_id,
                    area_m2=round(poly_area_m2, 2),
                    area_km2=round(poly_area_km2, 5),
                    centroid=[round(c.x, 6), round(c.y, 6)],
                    bounding_box=[round(b[0], 6), round(b[1], 6), round(b[2], 6), round(b[3], 6)],
                    source_job_id=job_id,
                )
                features.append({
                    "type": "Feature",
                    "id": region_id,
                    "properties": prop.model_dump(),
                    "geometry": shapely.geometry.mapping(sh_geom),
                })

        region_count = len(region_areas)
        largest_km2 = float(max(region_areas)) if region_areas else 0.0
        smallest_km2 = float(min(region_areas)) if region_areas else 0.0
        mean_km2 = float(np.mean(region_areas)) if region_areas else 0.0

        metrics = FloodExtentMetrics(
            total_aoi_area_km2=round(total_aoi_area_km2, 4),
            valid_analysis_area_km2=round(total_aoi_area_km2, 4),
            flooded_pixel_count=flooded_pixels,
            total_valid_pixels=total_pixels,
            flooded_area_m2=round(flooded_area_m2, 2),
            flooded_area_km2=round(flooded_area_km2, 5),
            flooded_percentage=flood_pct,
            flood_region_count=region_count,
            largest_flood_region_km2=round(largest_km2, 5),
            smallest_flood_region_km2=round(smallest_km2, 5),
            mean_flood_region_km2=round(mean_km2, 5),
        )

        geojson_fc = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": crs_name}},
            "features": features,
        }

        return metrics, flood_polys, geojson_fc

    def _analyze_change_zones(
        self,
        ard_pkg: Optional[Any],
        flood_polygons: List[Tuple[str, shapely.geometry.base.BaseGeometry, float]],
        job_id: str,
    ) -> Tuple[ChangeZoneMetrics, Dict[str, Any]]:
        """
        Geospatial heuristic layer for terrain change / possible debris candidate zones.
        Uses Sentinel-1 backscatter difference (diff_vv, diff_vh) outside the main water mask.
        Strictly labels zones as 'possible_change_zone' (never claimed as confirmed debris).
        """
        features = []
        if not ard_pkg or not ard_pkg.sar_ard or ard_pkg.status == "contract_ready":
            return (
                ChangeZoneMetrics(
                    status="insufficient_data",
                    total_change_zones=0,
                    total_area_km2=0.0,
                    largest_zone_km2=0.0,
                    message="Physical satellite difference channels unavailable on disk; change zone analysis skipped.",
                ),
                {"type": "FeatureCollection", "features": []},
            )

        # Check for diff channels on disk
        diff_bands = [b for b in ard_pkg.sar_ard.channels if "diff" in b.band_name and b.file_path]
        if not diff_bands:
            return (
                ChangeZoneMetrics(
                    status="insufficient_data",
                    total_change_zones=0,
                    total_area_km2=0.0,
                    largest_zone_km2=0.0,
                    message="No difference GeoTIFF channels available for terrain change analysis.",
                ),
                {"type": "FeatureCollection", "features": []},
            )

        # Inspect diff channel
        try:
            diff_band = diff_bands[0]
            with rasterio.open(diff_band.file_path) as src:
                diff_arr = src.read(1)
                transform = src.transform
                bounds = src.bounds
            
            mean_lat = (bounds.bottom + bounds.top) / 2.0
            # Identify high change threshold (> +3 dB roughening/rubble or < -4 dB severe scouring outside open water)
            change_mask = (np.abs(diff_arr) > 4.0).astype(np.uint8)
            extracted = shapes(change_mask, mask=(change_mask == 1), transform=transform)

            # Union of water polygons to exclude primary inundation
            water_union = shapely.ops.unary_union([p[1] for p in flood_polygons]) if flood_polygons else None

            idx = 1
            zone_areas = []
            for geom_dict, val in extracted:
                sh_geom = shapely.geometry.shape(geom_dict)
                if not sh_geom.is_valid:
                    sh_geom = sh_geom.buffer(0)
                if water_union and not water_union.is_empty:
                    sh_geom = sh_geom.difference(water_union)
                if sh_geom.is_empty:
                    continue

                area_m2 = calculate_polygon_metric_area(sh_geom, mean_lat)
                if area_m2 < 500.0:
                    continue

                area_km2 = area_m2 / 1_000_000.0
                zone_id = f"change_zone_{idx:03d}"
                idx += 1
                zone_areas.append(area_km2)

                c = sh_geom.centroid
                prop = ChangeZoneProperties(
                    zone_id=zone_id,
                    area_m2=round(area_m2, 2),
                    area_km2=round(area_km2, 5),
                    centroid=[round(c.x, 6), round(c.y, 6)],
                    heuristic_score=0.75,
                    change_features_used=[diff_band.band_name],
                    source_job_id=job_id,
                )
                features.append({
                    "type": "Feature",
                    "id": zone_id,
                    "properties": prop.model_dump(),
                    "geometry": shapely.geometry.mapping(sh_geom),
                })

            total_km2 = float(sum(zone_areas))
            largest_km2 = float(max(zone_areas)) if zone_areas else 0.0
            return (
                ChangeZoneMetrics(
                    status="available",
                    total_change_zones=len(features),
                    total_area_km2=round(total_km2, 5),
                    largest_zone_km2=round(largest_km2, 5),
                    message=f"Identified {len(features)} candidate terrain change zones outside primary flood extents.",
                ),
                {"type": "FeatureCollection", "features": features},
            )
        except Exception as e:
            logger.warning("Terrain change analysis failed: %s", e)
            return (
                ChangeZoneMetrics(
                    status="insufficient_data",
                    total_change_zones=0,
                    total_area_km2=0.0,
                    largest_zone_km2=0.0,
                    message=f"Change zone analysis error: {str(e)}",
                ),
                {"type": "FeatureCollection", "features": []},
            )

    def _evaluate_buildings(
        self,
        buildings: List[Dict[str, Any]],
        flood_polygons: List[Tuple[str, shapely.geometry.base.BaseGeometry, float]],
        mean_lat: float,
        pot_thresh: float = 0.0,
        heavy_thresh: float = 50.0,
    ) -> Tuple[List[BuildingImpact], Dict[str, Any], BuildingAssetSummary]:
        """Compute spatial overlap and classify exposure for each OSM building."""
        m_lat, m_lon = compute_geodesic_scales(mean_lat)
        impact_list: List[BuildingImpact] = []
        features = []

        unaffected_c = 0
        pot_c = 0
        heavy_c = 0
        total_area = 0.0
        affected_area = 0.0

        flood_geoms = [p[1] for p in flood_polygons]
        flood_union = shapely.ops.unary_union(flood_geoms) if flood_geoms else None
        prep_flood = shapely.prepared.prep(flood_union) if (flood_union and not flood_union.is_empty) else None

        for b in buildings:
            b_id = b.get("id", f"bldg_{len(impact_list)+1}")
            props = b.get("properties", {})
            geom_dict = b.get("geometry", {})
            try:
                b_geom = shapely.geometry.shape(geom_dict)
                if not b_geom.is_valid:
                    b_geom = b_geom.buffer(0)
            except Exception:
                continue

            b_area_m2 = calculate_polygon_metric_area(b_geom, mean_lat)
            total_area += b_area_m2

            overlap_m2 = 0.0
            dist_m = 0.0

            if prep_flood and prep_flood.intersects(b_geom):
                intersection = b_geom.intersection(flood_union)
                overlap_m2 = calculate_polygon_metric_area(intersection, mean_lat)
                dist_m = 0.0
            elif flood_union and not flood_union.is_empty and len(buildings) <= 1000:
                deg_dist = b_geom.distance(flood_union)
                dist_m = round(deg_dist * ((m_lat + m_lon) / 2.0), 1)

            overlap_pct = round(min(100.0, (overlap_m2 / max(0.1, b_area_m2)) * 100.0), 2)

            # Classify exposure
            if overlap_pct <= pot_thresh:
                classification = BuildingExposureClassification.UNAFFECTED
                unaffected_c += 1
            elif overlap_pct < heavy_thresh:
                classification = BuildingExposureClassification.POTENTIALLY_AFFECTED
                pot_c += 1
                affected_area += b_area_m2
            else:
                classification = BuildingExposureClassification.HEAVILY_AFFECTED
                heavy_c += 1
                affected_area += b_area_m2

            c = b_geom.centroid
            impact = BuildingImpact(
                building_id=b_id,
                osm_id=props.get("osm_id"),
                name=props.get("name"),
                building_type=props.get("building_type", "yes"),
                area_m2=round(b_area_m2, 2),
                flooded_overlap_m2=round(overlap_m2, 2),
                overlap_percentage=overlap_pct,
                distance_to_flood_m=round(dist_m, 1),
                classification=classification,
                centroid=[round(c.x, 6), round(c.y, 6)],
                geometry=geom_dict,
            )
            impact_list.append(impact)

            features.append({
                "type": "Feature",
                "id": b_id,
                "properties": {
                    "building_id": b_id,
                    "osm_id": props.get("osm_id"),
                    "name": props.get("name"),
                    "building_type": props.get("building_type", "yes"),
                    "area_m2": round(b_area_m2, 2),
                    "flooded_overlap_m2": round(overlap_m2, 2),
                    "overlap_percentage": overlap_pct,
                    "distance_to_flood_m": round(dist_m, 1),
                    "classification": classification.value,
                },
                "geometry": geom_dict,
            })

        summary = BuildingAssetSummary(
            total=len(impact_list),
            unaffected=unaffected_c,
            potentially_affected=pot_c,
            heavily_affected=heavy_c,
            total_area_m2=round(total_area, 2),
            affected_area_m2=round(affected_area, 2),
        )

        return impact_list, {"type": "FeatureCollection", "features": features}, summary

    def _evaluate_roads(
        self,
        roads: List[Dict[str, Any]],
        flood_polygons: List[Tuple[str, shapely.geometry.base.BaseGeometry, float]],
        mean_lat: float,
        pot_thresh: float = 0.0,
        heavy_thresh: float = 50.0,
    ) -> Tuple[List[RoadImpact], Dict[str, Any], RoadAssetSummary]:
        """Compute intersecting flood length and classify exposure for each OSM road segment."""
        impact_list: List[RoadImpact] = []
        features = []

        unaffected_c = 0
        pot_c = 0
        heavy_c = 0
        total_len_m = 0.0
        affected_len_m = 0.0

        flood_geoms = [p[1] for p in flood_polygons]
        flood_union = shapely.ops.unary_union(flood_geoms) if flood_geoms else None
        prep_flood = shapely.prepared.prep(flood_union) if (flood_union and not flood_union.is_empty) else None

        for r in roads:
            r_id = r.get("id", f"road_{len(impact_list)+1}")
            props = r.get("properties", {})
            geom_dict = r.get("geometry", {})
            try:
                r_geom = shapely.geometry.shape(geom_dict)
                if not r_geom.is_valid:
                    r_geom = r_geom.buffer(0)
            except Exception:
                continue

            r_len_m = calculate_linestring_metric_length(r_geom, mean_lat)
            total_len_m += r_len_m

            flooded_m = 0.0
            nearest_region = None

            if prep_flood and prep_flood.intersects(r_geom):
                intersection = r_geom.intersection(flood_union)
                flooded_m = calculate_linestring_metric_length(intersection, mean_lat)
                # Find nearest intersecting flood region
                for reg_id, f_poly, _ in flood_polygons:
                    if r_geom.intersects(f_poly):
                        nearest_region = reg_id
                        break

            affected_pct = round(min(100.0, (flooded_m / max(0.1, r_len_m)) * 100.0), 2)

            if affected_pct <= pot_thresh:
                classification = RoadExposureClassification.UNAFFECTED
                unaffected_c += 1
            elif affected_pct < heavy_thresh:
                classification = RoadExposureClassification.POTENTIALLY_BLOCKED
                pot_c += 1
                affected_len_m += flooded_m
            else:
                classification = RoadExposureClassification.HEAVILY_AFFECTED
                heavy_c += 1
                affected_len_m += flooded_m

            impact = RoadImpact(
                road_id=r_id,
                osm_id=props.get("osm_id"),
                name=props.get("name"),
                highway_type=props.get("highway_type", "road"),
                total_length_m=round(r_len_m, 1),
                flooded_length_m=round(flooded_m, 1),
                affected_percentage=affected_pct,
                nearest_flood_region_id=nearest_region,
                classification=classification,
                geometry=geom_dict,
            )
            impact_list.append(impact)

            features.append({
                "type": "Feature",
                "id": r_id,
                "properties": {
                    "road_id": r_id,
                    "osm_id": props.get("osm_id"),
                    "name": props.get("name"),
                    "highway_type": props.get("highway_type", "road"),
                    "total_length_m": round(r_len_m, 1),
                    "flooded_length_m": round(flooded_m, 1),
                    "affected_percentage": affected_pct,
                    "nearest_flood_region_id": nearest_region,
                    "classification": classification.value,
                },
                "geometry": geom_dict,
            })

        summary = RoadAssetSummary(
            total_segments=len(impact_list),
            unaffected=unaffected_c,
            potentially_blocked=pot_c,
            heavily_affected=heavy_c,
            total_length_km=round(total_len_m / 1000.0, 3),
            affected_length_km=round(affected_len_m / 1000.0, 3),
        )

        return impact_list, {"type": "FeatureCollection", "features": features}, summary

    def _evaluate_bridges(
        self,
        bridges: List[Dict[str, Any]],
        flood_polygons: List[Tuple[str, shapely.geometry.base.BaseGeometry, float]],
        mean_lat: float,
        pot_thresh: float = 0.0,
        heavy_thresh: float = 50.0,
    ) -> Tuple[List[BridgeImpact], Dict[str, Any], BridgeAssetSummary]:
        """Compute flood intersection and classify exposure for each OSM bridge."""
        impact_list: List[BridgeImpact] = []
        features = []

        unaffected_c = 0
        pot_c = 0
        heavy_c = 0
        total_len_m = 0.0
        affected_len_m = 0.0

        flood_geoms = [p[1] for p in flood_polygons]
        flood_union = shapely.ops.unary_union(flood_geoms) if flood_geoms else None

        for b in bridges:
            b_id = b.get("id", f"bridge_{len(impact_list)+1}")
            props = b.get("properties", {})
            geom_dict = b.get("geometry", {})
            try:
                b_geom = shapely.geometry.shape(geom_dict)
                if not b_geom.is_valid:
                    b_geom = b_geom.buffer(0)
            except Exception:
                continue

            # Length or boundary length
            if b_geom.geom_type in ["LineString", "MultiLineString"]:
                b_len_m = calculate_linestring_metric_length(b_geom, mean_lat)
            else:
                b_len_m = calculate_linestring_metric_length(b_geom.boundary, mean_lat)

            total_len_m += b_len_m
            flooded_m = 0.0
            nearest_region = None

            if flood_union and not flood_union.is_empty and b_geom.intersects(flood_union):
                intersection = b_geom.intersection(flood_union)
                if intersection.geom_type in ["LineString", "MultiLineString"]:
                    flooded_m = calculate_linestring_metric_length(intersection, mean_lat)
                elif intersection.geom_type in ["Polygon", "MultiPolygon"]:
                    flooded_m = calculate_linestring_metric_length(intersection.boundary, mean_lat)
                else:
                    flooded_m = b_len_m * 0.5

                for reg_id, f_poly, _ in flood_polygons:
                    if b_geom.intersects(f_poly):
                        nearest_region = reg_id
                        break

            affected_pct = round(min(100.0, (flooded_m / max(0.1, b_len_m)) * 100.0), 2)

            if affected_pct <= pot_thresh:
                classification = BridgeExposureClassification.UNAFFECTED
                unaffected_c += 1
            elif affected_pct < heavy_thresh:
                classification = BridgeExposureClassification.POTENTIALLY_AFFECTED
                pot_c += 1
                affected_len_m += flooded_m
            else:
                classification = BridgeExposureClassification.HEAVILY_AFFECTED
                heavy_c += 1
                affected_len_m += flooded_m

            impact = BridgeImpact(
                bridge_id=b_id,
                osm_id=props.get("osm_id"),
                name=props.get("name"),
                bridge_type=props.get("bridge_type", "bridge"),
                total_length_m=round(b_len_m, 1),
                flooded_length_m=round(flooded_m, 1),
                affected_percentage=affected_pct,
                nearest_flood_region_id=nearest_region,
                classification=classification,
                geometry=geom_dict,
            )
            impact_list.append(impact)

            features.append({
                "type": "Feature",
                "id": b_id,
                "properties": {
                    "bridge_id": b_id,
                    "osm_id": props.get("osm_id"),
                    "name": props.get("name"),
                    "bridge_type": props.get("bridge_type", "bridge"),
                    "total_length_m": round(b_len_m, 1),
                    "flooded_length_m": round(flooded_m, 1),
                    "affected_percentage": affected_pct,
                    "nearest_flood_region_id": nearest_region,
                    "classification": classification.value,
                },
                "geometry": geom_dict,
            })

        summary = BridgeAssetSummary(
            total=len(impact_list),
            unaffected=unaffected_c,
            potentially_affected=pot_c,
            heavily_affected=heavy_c,
            total_length_km=round(total_len_m / 1000.0, 3),
            affected_length_km=round(affected_len_m / 1000.0, 3),
        )

        return impact_list, {"type": "FeatureCollection", "features": features}, summary

    def _persist_outputs(self, job_id: str, res: GeospatialAnalysisResponse) -> None:
        """Write structured JSON summary and GeoJSON feature layers to disk."""
        job_dir = self.outputs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # 1. Full response JSON
        with open(job_dir / "geospatial_summary.json", "w", encoding="utf-8") as f:
            json.dump(res.model_dump(mode="json"), f, indent=2)

        # 2. Individual GeoJSON layers
        layers = [
            ("flood_regions.geojson", res.flood_regions),
            ("change_zones.geojson", res.change_zones),
            ("buildings.geojson", res.buildings),
            ("roads.geojson", res.roads),
            ("bridges.geojson", res.bridges),
        ]
        for filename, geojson_data in layers:
            with open(job_dir / filename, "w", encoding="utf-8") as f:
                json.dump(geojson_data, f, indent=2)

    def get_geospatial_result(self, job_id: str) -> Optional[GeospatialAnalysisResponse]:
        """Retrieve stored geospatial analysis result from memory or disk."""
        if job_id in self._geospatial_store:
            return self._geospatial_store[job_id]

        summary_file = self.outputs_dir / job_id / "geospatial_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                res = GeospatialAnalysisResponse(**data)
                self._geospatial_store[job_id] = res
                return res
            except Exception as e:
                logger.error("Failed to load geospatial summary %s: %s", summary_file, e)

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
_geospatial_service_instance: Optional[GeospatialDamageAnalysisService] = None


def get_geospatial_service() -> GeospatialDamageAnalysisService:
    """Dependency provider for GeospatialDamageAnalysisService."""
    global _geospatial_service_instance
    if _geospatial_service_instance is None:
        _geospatial_service_instance = GeospatialDamageAnalysisService()
    return _geospatial_service_instance
