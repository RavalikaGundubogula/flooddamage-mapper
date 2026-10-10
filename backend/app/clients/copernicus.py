import json
import logging
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.core.config import get_settings
from app.schemas.satellite import SceneMetadata

logger = logging.getLogger(__name__)


def geojson_to_wkt_polygon(geometry: Dict[str, Any]) -> str:
    """Convert a GeoJSON Polygon geometry into a standard WKT string."""
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geom_type != "Polygon" or not coords or not isinstance(coords, list):
        raise ValueError(f"Expected GeoJSON Polygon geometry, got {geom_type}")

    # Standard WKT expects exterior ring formatted as: POLYGON((lon lat, lon lat, ...))
    exterior_ring = coords[0]
    points_str = ", ".join(f"{float(pt[0])} {float(pt[1])}" for pt in exterior_ring)
    return f"POLYGON(({points_str}))"


def format_iso_timestamp(date_str: str, is_end_of_day: bool = False) -> str:
    """Ensure date is formatted as ISO 8601 UTC string for CDSE OData queries."""
    clean_date = date_str.strip()
    if len(clean_date) == 10:  # YYYY-MM-DD
        time_part = "23:59:59.999Z" if is_end_of_day else "00:00:00.000Z"
        return f"{clean_date}T{time_part}"
    return clean_date


class SatelliteCatalog(ABC):
    """Abstract base interface for satellite catalog search and discovery."""

    @abstractmethod
    def search(
        self,
        aoi: Dict[str, Any],
        start_date: str,
        end_date: str,
        collection: str,
        max_results: int = 20,
    ) -> List[SceneMetadata]:
        """Search the satellite catalog for scene metadata within spatial and temporal bounds."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify catalog connectivity and availability."""
        pass

    @abstractmethod
    def get_product(self, product_id_or_name: str) -> Optional[SceneMetadata]:
        """Fetch metadata for a single product by CDSE UUID or SAFE product name."""
        pass


class CopernicusCatalogClient(SatelliteCatalog):
    """
    Official Copernicus Data Space Ecosystem (CDSE) catalog discovery client.
    Uses CDSE OData public catalog endpoints with spatial and temporal filtering.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[int] = None) -> None:
        settings = get_settings()
        self.base_url = base_url or settings.COPERNICUS_CATALOG_URL
        self.timeout = timeout or settings.CATALOG_TIMEOUT_SECONDS

    def health_check(self) -> bool:
        """Check if CDSE catalog endpoint is reachable."""
        try:
            query = urllib.parse.urlencode({"$top": 1})
            url = f"{self.base_url}?{query}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FloodDamageMapper/0.1"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning("Copernicus catalog health check failed: %s", e)
            return False

    def search(
        self,
        aoi: Dict[str, Any],
        start_date: str,
        end_date: str,
        collection: str,
        max_results: int = 20,
    ) -> List[SceneMetadata]:
        """
        Execute spatial and temporal query against CDSE OData catalog.
        """
        wkt_polygon = geojson_to_wkt_polygon(aoi)
        start_iso = format_iso_timestamp(start_date, is_end_of_day=False)
        end_iso = format_iso_timestamp(end_date, is_end_of_day=True)

        # Build OData filter expression
        filter_expr = (
            f"Collection/Name eq '{collection.upper()}' and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt_polygon}') and "
            f"ContentDate/Start ge {start_iso} and "
            f"ContentDate/Start le {end_iso}"
        )

        query_params = {
            "$filter": filter_expr,
            "$top": max_results,
            "$expand": "Attributes",
            "$orderby": "ContentDate/Start desc",
        }

        url = f"{self.base_url}?{urllib.parse.urlencode(query_params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FloodDamageMapper/0.1"}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    logger.error("CDSE returned HTTP %s", resp.status)
                    return []
                data = json.loads(resp.read().decode("utf-8"))
                products = data.get("value", [])
                return [self._normalize_product(p, collection) for p in products]
        except urllib.error.HTTPError as e:
            logger.error("Copernicus catalog HTTP error %s: %s", e.code, e.reason)
            return []
        except Exception as e:
            logger.error("Copernicus catalog query failed: %s", e)
            return []

    def _normalize_product(self, raw: Dict[str, Any], requested_collection: str) -> SceneMetadata:
        """Convert CDSE raw OData product JSON into standardized SceneMetadata."""
        name = raw.get("Name", "UNKNOWN")
        content_date = raw.get("ContentDate", {})
        start_str = content_date.get("Start")
        
        # Parse datetime
        if start_str:
            clean_ts = start_str.replace("Z", "+00:00")
            acq_dt = datetime.fromisoformat(clean_ts)
        else:
            acq_dt = datetime.now(timezone.utc)

        # Parse attributes list into a dictionary
        attrs_list = raw.get("Attributes", [])
        attr_dict = {
            a.get("Name"): a.get("Value")
            for a in attrs_list
            if isinstance(a, dict) and "Name" in a
        }

        # Identify platform (e.g. S1A, S1B, S2A, S2B)
        platform = None
        if name.startswith(("S1A", "S1B", "S2A", "S2B")):
            platform = name[:3]

        # Extract cloud cover for optical (Sentinel-2)
        cloud_val = attr_dict.get("cloudCover")
        cloud_cover = float(cloud_val) if cloud_val is not None else None

        # Extract radar-specific attributes (Sentinel-1)
        orbit_dir = attr_dict.get("orbitDirection")
        rel_orbit = attr_dict.get("relativeOrbitNumber")
        polarization = attr_dict.get("polarisationChannels")
        product_type = attr_dict.get("productType") or raw.get("Origin", "PRODUCT")

        # GeoFootprint if available
        geometry = raw.get("GeoFootprint")

        # CDSE Product UUID & Storage Attributes
        cdse_id = raw.get("Id")
        if cdse_id:
            attr_dict["cdse_id"] = str(cdse_id)
        content_len = raw.get("ContentLength")
        if content_len is not None:
            attr_dict["content_length"] = int(content_len)
        checksums = raw.get("Checksum")
        if checksums:
            attr_dict["checksums"] = checksums

        return SceneMetadata(
            product_id=name,
            collection=requested_collection.upper(),
            platform=platform,
            acquisition_datetime=acq_dt,
            geometry=geometry,
            cloud_cover=cloud_cover,
            orbit_direction=orbit_dir,
            relative_orbit=int(rel_orbit) if rel_orbit is not None else None,
            polarization=polarization,
            product_type=str(product_type),
            catalog_id="copernicus_cdse",
            extra_attributes=attr_dict,
        )

    def get_product(self, product_id_or_name: str) -> Optional[SceneMetadata]:
        """Fetch metadata for a single product by CDSE UUID or SAFE product name."""
        import re
        clean_target = product_id_or_name.strip()
        is_uuid = bool(re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", clean_target))
        
        if is_uuid:
            url = f"{self.base_url}({clean_target})?$expand=Attributes"
        else:
            query_filter = f"Name eq '{clean_target}'"
            query_params = {
                "$filter": query_filter,
                "$expand": "Attributes",
                "$top": 1,
            }
            url = f"{self.base_url}?{urllib.parse.urlencode(query_params)}"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FloodDamageMapper/0.1"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    return None
                data = json.loads(resp.read().decode("utf-8"))
                if is_uuid:
                    raw_product = data
                else:
                    items = data.get("value", [])
                    if not items:
                        return None
                    raw_product = items[0]
                
                p_name = raw_product.get("Name", "")
                collection = "SENTINEL-1" if p_name.startswith("S1") else "SENTINEL-2" if p_name.startswith("S2") else "UNKNOWN"
                return self._normalize_product(raw_product, collection)
        except Exception as e:
            logger.error("Failed to fetch product %s from CDSE: %s", clean_target, e)
            return None
