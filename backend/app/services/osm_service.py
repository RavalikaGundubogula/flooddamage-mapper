import hashlib
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import shapely.geometry
from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class OpenStreetMapService:
    """
    Service responsible for acquiring and caching PRE-EVENT OpenStreetMap (OSM)
    infrastructure assets (buildings, roads, bridges) within the analysis AOI.
    
    Guarantees:
    - Pre-event baseline asset inventory only (no post-event damage edits).
    - Caching keyed by AOI geometry to avoid repeated Overpass API calls.
    - Graceful fallback: If Overpass API is unreachable and no offline file is provided,
      reports osm_status='unavailable' with empty inventories instead of fabricating fake assets.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.cache_dir = self.settings.get_data_path(self.settings.DATA_OSM_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.overpass_url = self.settings.OVERPASS_URL
        self.timeout = self.settings.OVERPASS_TIMEOUT_SECONDS

    def _compute_aoi_hash(self, aoi: Dict[str, Any]) -> str:
        """Generate deterministic cache key for an AOI polygon."""
        aoi_str = json.dumps(aoi, sort_keys=True)
        return hashlib.sha256(aoi_str.encode("utf-8")).hexdigest()[:16]

    def _extract_bounding_box(self, aoi: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Extract [south, west, north, east] (min_lat, min_lon, max_lat, max_lon) from AOI."""
        geom = shapely.geometry.shape(aoi)
        min_lon, min_lat, max_lon, max_lat = geom.bounds
        return min_lat, min_lon, max_lat, max_lon

    def fetch_pre_event_assets(
        self,
        aoi: Dict[str, Any],
        custom_file_path: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retrieve pre-event OSM buildings, roads, and bridges for the specified AOI.
        
        Args:
            aoi: Normalized GeoJSON polygon of the area of interest.
            custom_file_path: Optional path to a local pre-downloaded GeoJSON/JSON file.
            force_refresh: Ignore local cache if True.
            
        Returns:
            Tuple of:
            - status: "available" or "unavailable"
            - buildings: List of GeoJSON Feature dicts
            - roads: List of GeoJSON Feature dicts
            - bridges: List of GeoJSON Feature dicts
        """
        # 1. If an offline local file is provided, load directly from disk
        if custom_file_path:
            p = Path(custom_file_path)
            if p.exists():
                logger.info("Loading pre-event OSM assets from local file: %s", p)
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return self._parse_local_osm_dataset(data, aoi)
                except Exception as e:
                    logger.warning("Failed to parse local OSM file %s: %s", p, e)

        # 2. Check disk cache
        aoi_hash = self._compute_aoi_hash(aoi)
        cache_file = self.cache_dir / f"osm_{aoi_hash}.json"

        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                logger.info("Loaded OSM assets from local disk cache (%s)", cache_file.name)
                return (
                    "available",
                    cached_data.get("buildings", []),
                    cached_data.get("roads", []),
                    cached_data.get("bridges", []),
                )
            except Exception as e:
                logger.warning("Failed to read OSM cache %s: %s. Re-querying.", cache_file, e)

        # 3. Query Overpass API
        min_lat, min_lon, max_lat, max_lon = self._extract_bounding_box(aoi)
        overpass_query = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["bridge"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["man_made"="bridge"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body geom;
        >;
        out skel qt;
        """

        try:
            logger.info("Querying Overpass API for AOI bbox [%.3f, %.3f, %.3f, %.3f]...", min_lat, min_lon, max_lat, max_lon)
            data = urllib.parse.urlencode({"data": overpass_query}).encode("utf-8")
            req = urllib.request.Request(
                self.overpass_url,
                data=data,
                headers={
                    "User-Agent": "FloodDamageMapper/0.3 (DisasterResponse; HumanitarianResearch)",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    raw_osm = json.loads(resp.read().decode("utf-8"))
                    buildings, roads, bridges = self._parse_overpass_elements(raw_osm, aoi)
                    
                    # Cache valid response to disk
                    cache_payload = {
                        "aoi_hash": aoi_hash,
                        "buildings": buildings,
                        "roads": roads,
                        "bridges": bridges,
                    }
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(cache_payload, f, indent=2)

                    logger.info(
                        "Overpass query successful: %d buildings, %d roads, %d bridges extracted.",
                        len(buildings),
                        len(roads),
                        len(bridges),
                    )
                    return "available", buildings, roads, bridges
        except urllib.error.URLError as e:
            logger.warning("Overpass API query unreachable or timed out (%s): %s", self.overpass_url, e)
        except Exception as e:
            logger.warning("Failed to execute Overpass query: %s", e)

        # 4. Graceful handling when external live network is unavailable
        logger.info("OSM data is marked as 'unavailable' for this run. Zero assets fabricated.")
        return "unavailable", [], [], []

    def _parse_overpass_elements(
        self,
        overpass_json: Dict[str, Any],
        aoi: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convert Overpass JSON elements with inline geometry into standardized GeoJSON features."""
        aoi_poly = shapely.geometry.shape(aoi)
        buildings = []
        roads = []
        bridges = []

        elements = overpass_json.get("elements", [])
        for elem in elements:
            tags = elem.get("tags", {})
            geom_coords = elem.get("geometry", [])
            if not geom_coords:
                continue

            elem_id = str(elem.get("id", ""))
            
            # 1. Bridge extraction (distinct from standard roads)
            is_bridge = (tags.get("bridge") == "yes") or (tags.get("man_made") == "bridge")

            # 2. Building extraction (ways or multipolygon relations)
            if "building" in tags:
                coords = [[pt["lon"], pt["lat"]] for pt in geom_coords]
                if len(coords) >= 3:
                    # Close polygon ring if needed
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    poly_geom = {"type": "Polygon", "coordinates": [coords]}
                    try:
                        sh_geom = shapely.geometry.shape(poly_geom)
                        if sh_geom.is_valid and sh_geom.intersects(aoi_poly):
                            buildings.append({
                                "type": "Feature",
                                "id": f"osm_bldg_{elem_id}",
                                "properties": {
                                    "osm_id": elem_id,
                                    "name": tags.get("name"),
                                    "building_type": tags.get("building", "yes"),
                                    "amenity": tags.get("amenity"),
                                },
                                "geometry": poly_geom,
                            })
                    except Exception:
                        pass
                continue

            # 3. Roads and Bridges
            if "highway" in tags:
                coords = [[pt["lon"], pt["lat"]] for pt in geom_coords]
                if len(coords) >= 2:
                    line_geom = {"type": "LineString", "coordinates": coords}
                    try:
                        sh_geom = shapely.geometry.shape(line_geom)
                        if sh_geom.is_valid and sh_geom.intersects(aoi_poly):
                            feature = {
                                "type": "Feature",
                                "id": f"osm_bridge_{elem_id}" if is_bridge else f"osm_road_{elem_id}",
                                "properties": {
                                    "osm_id": elem_id,
                                    "name": tags.get("name"),
                                    "highway_type": tags.get("highway", "road"),
                                    "surface": tags.get("surface"),
                                    "is_bridge": is_bridge,
                                },
                                "geometry": line_geom,
                            }
                            if is_bridge:
                                bridges.append(feature)
                            else:
                                roads.append(feature)
                    except Exception:
                        pass

        return buildings, roads, bridges

    def _parse_local_osm_dataset(
        self,
        data: Dict[str, Any],
        aoi: Dict[str, Any],
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse local GeoJSON FeatureCollection or dictionary containing buildings/roads/bridges."""
        aoi_poly = shapely.geometry.shape(aoi)
        buildings = []
        roads = []
        bridges = []

        # If data already pre-categorized
        if isinstance(data, dict) and any(k in data for k in ["buildings", "roads", "bridges"]):
            b_list = data.get("buildings", [])
            r_list = data.get("roads", [])
            br_list = data.get("bridges", [])
            return "available", b_list, r_list, br_list

        # If single FeatureCollection
        features = data.get("features", []) if isinstance(data, dict) else data
        for f in features:
            geom = f.get("geometry")
            if not geom:
                continue
            props = f.get("properties", {})
            try:
                sh_geom = shapely.geometry.shape(geom)
                if not sh_geom.intersects(aoi_poly):
                    continue

                if props.get("bridge") == "yes" or props.get("is_bridge") or "bridge" in props.get("name", "").lower():
                    bridges.append(f)
                elif "building" in props or geom.get("type") in ["Polygon", "MultiPolygon"]:
                    buildings.append(f)
                elif "highway" in props or geom.get("type") in ["LineString", "MultiLineString"]:
                    roads.append(f)
            except Exception:
                continue

        return "available", buildings, roads, bridges

    def fetch_pre_event_settlements_and_hospitals(
        self,
        aoi: Dict[str, Any],
        custom_file_path: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retrieve pre-event OSM settlements and hospitals within/relevant to the AOI.
        
        Args:
            aoi: Normalized GeoJSON polygon of the area of interest.
            custom_file_path: Optional path to a local pre-downloaded GeoJSON/JSON file.
            force_refresh: Ignore local cache if True.
            
        Returns:
            Tuple of:
            - status: "available" or "unavailable"
            - settlements: List of GeoJSON Feature dicts with Point geometries
            - hospitals: List of GeoJSON Feature dicts with Point geometries
        """
        # 1. If an offline local file is provided, load directly from disk
        if custom_file_path:
            p = Path(custom_file_path)
            if p.exists():
                logger.info("Loading pre-event settlements and hospitals from local file: %s", p)
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return self._parse_local_settlements_and_hospitals(data, aoi)
                except Exception as e:
                    logger.warning("Failed to parse local settlements/hospitals file %s: %s", p, e)

        # 2. Check disk cache
        aoi_hash = self._compute_aoi_hash(aoi)
        cache_file = self.cache_dir / f"osm_places_{aoi_hash}.json"

        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                logger.info("Loaded settlements/hospitals from local disk cache (%s)", cache_file.name)
                return (
                    "available",
                    cached_data.get("settlements", []),
                    cached_data.get("hospitals", []),
                )
            except Exception as e:
                logger.warning("Failed to read OSM places cache %s: %s. Re-querying.", cache_file, e)

        # 3. Query Overpass API for places and healthcare
        min_lat, min_lon, max_lat, max_lon = self._extract_bounding_box(aoi)
        pad = 0.02
        q_min_lat, q_min_lon = min_lat - pad, min_lon - pad
        q_max_lat, q_max_lon = max_lat + pad, max_lon + pad

        overpass_query = f"""
        [out:json][timeout:{self.timeout}];
        (
          node["place"]({q_min_lat},{q_min_lon},{q_max_lat},{q_max_lon});
          way["place"]({q_min_lat},{q_min_lon},{q_max_lat},{q_max_lon});
          node["amenity"="hospital"]({q_min_lat},{q_min_lon},{q_max_lat},{q_max_lon});
          way["amenity"="hospital"]({q_min_lat},{q_min_lon},{q_max_lat},{q_max_lon});
          node["healthcare"="hospital"]({q_min_lat},{q_min_lon},{q_max_lat},{q_max_lon});
          way["healthcare"="hospital"]({q_min_lat},{q_min_lon},{q_max_lat},{q_max_lon});
        );
        out center qt;
        """

        try:
            logger.info("Querying Overpass API for settlements & hospitals bbox [%.3f, %.3f, %.3f, %.3f]...", q_min_lat, q_min_lon, q_max_lat, q_max_lon)
            data = urllib.parse.urlencode({"data": overpass_query}).encode("utf-8")
            req = urllib.request.Request(
                self.overpass_url,
                data=data,
                headers={
                    "User-Agent": "FloodDamageMapper/0.3 (DisasterResponse; HumanitarianResearch)",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    raw_osm = json.loads(resp.read().decode("utf-8"))
                    settlements, hospitals = self._parse_overpass_places_elements(raw_osm)
                    
                    # Cache valid response to disk
                    cache_payload = {
                        "aoi_hash": aoi_hash,
                        "settlements": settlements,
                        "hospitals": hospitals,
                    }
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(cache_payload, f, indent=2)

                    logger.info(
                        "Overpass places query successful: %d settlements, %d hospitals extracted.",
                        len(settlements),
                        len(hospitals),
                    )
                    return "available", settlements, hospitals
        except urllib.error.URLError as e:
            logger.warning("Overpass API query unreachable or timed out (%s): %s", self.overpass_url, e)
        except Exception as e:
            logger.warning("Failed to execute Overpass places query: %s", e)

        # 4. Graceful fallback when external live network is unavailable
        logger.info("OSM settlements/hospitals data marked as 'unavailable' for this run. Zero assets fabricated.")
        return "unavailable", [], []

    def _parse_overpass_places_elements(
        self,
        overpass_json: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convert Overpass place and hospital elements into standardized GeoJSON Point features."""
        settlements = []
        hospitals = []

        elements = overpass_json.get("elements", [])
        for elem in elements:
            tags = elem.get("tags", {})
            elem_id = str(elem.get("id", ""))

            # Coordinates
            lon = elem.get("lon")
            lat = elem.get("lat")
            if lon is None or lat is None:
                center = elem.get("center", {})
                lon = center.get("lon")
                lat = center.get("lat")
            if lon is None or lat is None:
                continue

            pt_geom = {"type": "Point", "coordinates": [round(float(lon), 6), round(float(lat), 6)]}

            # 1. Settlement
            if "place" in tags:
                place_type = str(tags["place"]).lower()
                pop_val = None
                raw_pop = tags.get("population")
                if raw_pop is not None:
                    try:
                        pop_val = int(str(raw_pop).replace(",", "").strip())
                    except Exception:
                        pop_val = None

                settlements.append({
                    "type": "Feature",
                    "id": f"osm_settlement_{elem_id}",
                    "properties": {
                        "settlement_id": f"osm_settlement_{elem_id}",
                        "osm_id": elem_id,
                        "name": tags.get("name"),
                        "place_type": place_type,
                        "population": pop_val,
                        "osm_tags": tags,
                    },
                    "geometry": pt_geom,
                })

            # 2. Hospital / Healthcare
            is_hospital = (
                tags.get("amenity") == "hospital"
                or tags.get("healthcare") == "hospital"
                or tags.get("building") == "hospital"
            )
            if is_hospital:
                hospitals.append({
                    "type": "Feature",
                    "id": f"osm_hospital_{elem_id}",
                    "properties": {
                        "hospital_id": f"osm_hospital_{elem_id}",
                        "osm_id": elem_id,
                        "name": tags.get("name"),
                        "amenity_type": "hospital",
                        "osm_tags": tags,
                    },
                    "geometry": pt_geom,
                })

        return settlements, hospitals

    def _parse_local_settlements_and_hospitals(
        self,
        data: Dict[str, Any],
        aoi: Dict[str, Any],
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse local GeoJSON FeatureCollection or dict containing settlements and hospitals."""
        settlements = []
        hospitals = []

        if isinstance(data, dict):
            if "settlements" in data:
                settlements.extend(data["settlements"])
            if "hospitals" in data:
                hospitals.extend(data["hospitals"])

            # Also check if buildings have hospitals
            for b in data.get("buildings", []):
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
                            "id": f"hosp_{b.get('id', len(hospitals)+1)}",
                            "properties": {
                                "hospital_id": f"hosp_{b.get('id', len(hospitals)+1)}",
                                "osm_id": props.get("osm_id"),
                                "name": props.get("name"),
                                "amenity_type": "hospital",
                                "osm_tags": props,
                            },
                            "geometry": {"type": "Point", "coordinates": [round(c.x, 6), round(c.y, 6)]},
                        })
                    except Exception:
                        pass

        features = data.get("features", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        for f in features:
            props = f.get("properties", {})
            geom = f.get("geometry", {})
            try:
                sh_geom = shapely.geometry.shape(geom)
                c = sh_geom.centroid
                pt_geom = {"type": "Point", "coordinates": [round(c.x, 6), round(c.y, 6)]}
            except Exception:
                continue

            elem_id = str(props.get("osm_id") or f.get("id") or len(settlements) + len(hospitals) + 1)

            if "place" in props or "place_type" in props:
                pop_val = None
                raw_pop = props.get("population")
                if raw_pop is not None:
                    try:
                        pop_val = int(str(raw_pop).replace(",", "").strip())
                    except Exception:
                        pop_val = None

                settlements.append({
                    "type": "Feature",
                    "id": f.get("id") or f"osm_settlement_{elem_id}",
                    "properties": {
                        "settlement_id": props.get("settlement_id") or f.get("id") or f"osm_settlement_{elem_id}",
                        "osm_id": props.get("osm_id", elem_id),
                        "name": props.get("name"),
                        "place_type": str(props.get("place_type") or props.get("place", "settlement")).lower(),
                        "population": pop_val,
                        "osm_tags": props,
                    },
                    "geometry": pt_geom,
                })

            is_hosp = (
                props.get("amenity") == "hospital"
                or props.get("healthcare") == "hospital"
                or props.get("building") == "hospital"
                or props.get("building_type") == "hospital"
                or props.get("amenity_type") == "hospital"
            )
            if is_hosp:
                hospitals.append({
                    "type": "Feature",
                    "id": f.get("id") or f"osm_hospital_{elem_id}",
                    "properties": {
                        "hospital_id": props.get("hospital_id") or f.get("id") or f"osm_hospital_{elem_id}",
                        "osm_id": props.get("osm_id", elem_id),
                        "name": props.get("name"),
                        "amenity_type": "hospital",
                        "osm_tags": props,
                    },
                    "geometry": pt_geom,
                })

        return "available", settlements, hospitals


# Global singleton instance
_osm_service_instance: Optional[OpenStreetMapService] = None


def get_osm_service() -> OpenStreetMapService:
    """Dependency provider for OpenStreetMapService."""
    global _osm_service_instance
    if _osm_service_instance is None:
        _osm_service_instance = OpenStreetMapService()
    return _osm_service_instance

