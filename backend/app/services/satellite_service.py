from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from app.clients.copernicus import CopernicusCatalogClient, SatelliteCatalog
from app.core.config import get_settings
from app.schemas.analysis import normalize_and_validate_aoi
from app.schemas.satellite import (
    DiscoveryQuery,
    DiscoveryResult,
    RoutingDecisionResponse,
    SatelliteSubsystemStatus,
    SceneMetadata,
    Sentinel1Candidate,
    Sentinel1DiscoveryResponse,
    Sentinel2Candidate,
    Sentinel2DiscoveryResponse,
)
from app.services.satellite_cache import SatelliteDiscoveryCache, get_satellite_cache
from app.services.satellite_router import SmartSatelliteRouter, get_satellite_router

# Standard default AOI for Nepal Trishuli / Bhote Koshi river basin
DEFAULT_NEPAL_AOI: Dict[str, Any] = {
    "type": "Polygon",
    "coordinates": [
        [
            [85.20, 27.80],
            [85.50, 27.80],
            [85.50, 28.10],
            [85.20, 28.10],
            [85.20, 27.80],
        ]
    ],
}


class SatelliteDiscoveryService:
    """High-level service coordinating satellite catalog queries and metadata discovery."""

    def __init__(
        self,
        catalog_client: Optional[SatelliteCatalog] = None,
        router: Optional[SmartSatelliteRouter] = None,
        cache: Optional[SatelliteDiscoveryCache] = None,
    ) -> None:
        self.catalog = catalog_client or CopernicusCatalogClient()
        self.router = router or get_satellite_router()
        self.cache = cache or get_satellite_cache()
        self.settings = get_settings()

    def get_default_nepal_aoi(self) -> Dict[str, Any]:
        """Return standardized GeoJSON Polygon for Nepal Trishuli / Bhote Koshi basin."""
        return DEFAULT_NEPAL_AOI

    def _fetch_scenes(
        self,
        aoi: Dict[str, Any],
        start_date: str,
        end_date: str,
        collection: str,
        max_results: int,
    ) -> List[SceneMetadata]:
        """Fetch scenes from cache if enabled and present; otherwise query live catalog and cache."""
        if self.settings.CACHE_ENABLED and self.cache:
            cached = self.cache.get(
                aoi=aoi,
                start_date=start_date,
                end_date=end_date,
                collection=collection,
                max_results=max_results,
            )
            if cached is not None:
                return cached

        # Query live catalog
        scenes = self.catalog.search(
            aoi=aoi,
            start_date=start_date,
            end_date=end_date,
            collection=collection,
            max_results=max_results,
        )

        if self.settings.CACHE_ENABLED and self.cache and scenes:
            self.cache.set(
                aoi=aoi,
                start_date=start_date,
                end_date=end_date,
                collection=collection,
                max_results=max_results,
                scenes=scenes,
            )

        return scenes

    def get_status(self) -> SatelliteSubsystemStatus:
        """Inspect and return the health and capabilities of the satellite discovery layer."""
        is_healthy = self.catalog.health_check()
        status_str = "operational" if is_healthy else "degraded"

        auth_configured = bool(
            self.settings.COPERNICUS_CLIENT_ID and self.settings.COPERNICUS_CLIENT_SECRET
        )

        return SatelliteSubsystemStatus(
            status=status_str,
            catalog_name="Copernicus Data Space Ecosystem (CDSE)",
            catalog_endpoint=self.settings.COPERNICUS_CATALOG_URL,
            supported_satellites=["SENTINEL-1", "SENTINEL-2"],
            auth_configured=auth_configured,
            cache_enabled=self.settings.CACHE_ENABLED,
        )

    def search_scenes(self, query: DiscoveryQuery) -> DiscoveryResult:
        """Execute a scene discovery query via the configured satellite catalog."""
        scenes = self.catalog.search(
            aoi=query.aoi,
            start_date=query.start_date,
            end_date=query.end_date,
            collection=query.collection,
            max_results=query.max_results,
        )

        return DiscoveryResult(
            collection=query.collection,
            query_start_date=query.start_date,
            query_end_date=query.end_date,
            total_found=len(scenes),
            scenes=scenes,
        )

    def discover_sentinel_1(
        self,
        aoi: Optional[Any] = None,
        flood_date: str = "2024-08-26",
        days_before: int = 15,
        days_after: int = 7,
        max_results: int = 20,
    ) -> Sentinel1DiscoveryResponse:
        """
        Discover candidate Sentinel-1 radar scenes around a flood event date.
        
        Evaluates acquisition timing (pre/post-event), relative orbit tracks,
        orbit direction, and product type (GRD vs RAW/SLC) to prepare for
        change-detection analysis in later modules.
        """
        # Resolve and normalize AOI
        if aoi is None:
            resolved_aoi = DEFAULT_NEPAL_AOI
        else:
            resolved_aoi = normalize_and_validate_aoi(aoi)

        # Parse event date and establish temporal search window
        event_d = date.fromisoformat(flood_date.strip())
        event_dt = datetime(event_d.year, event_d.month, event_d.day, tzinfo=timezone.utc)

        start_d = event_d - timedelta(days=days_before)
        end_d = event_d + timedelta(days=days_after)

        start_str = start_d.isoformat()
        end_str = end_d.isoformat()

        # Query catalog or cache for Sentinel-1 products
        scenes = self._fetch_scenes(
            aoi=resolved_aoi,
            start_date=start_str,
            end_date=end_str,
            collection="SENTINEL-1",
            max_results=max_results,
        )

        candidates: List[Sentinel1Candidate] = []
        pre_count = 0
        post_count = 0
        grd_count = 0

        for s in scenes:
            # Normalize datetime to UTC
            s_dt = s.acquisition_datetime
            if s_dt.tzinfo is None:
                s_dt = s_dt.replace(tzinfo=timezone.utc)

            offset_days = round((s_dt - event_dt).total_seconds() / 86400.0, 2)
            timing = "pre_event" if s_dt < event_dt else "post_event"
            if timing == "pre_event":
                pre_count += 1
            else:
                post_count += 1

            is_grd = "GRD" in (s.product_type or "").upper()
            if is_grd:
                grd_count += 1

            candidates.append(
                Sentinel1Candidate(
                    scene=s,
                    timing=timing,
                    days_from_event=offset_days,
                    is_grd=is_grd,
                    relative_orbit=s.relative_orbit,
                    orbit_direction=s.orbit_direction,
                    polarization=s.polarization,
                )
            )

        # Sort candidates chronologically
        candidates.sort(key=lambda c: c.scene.acquisition_datetime)

        return Sentinel1DiscoveryResponse(
            collection="SENTINEL-1",
            flood_date=flood_date,
            temporal_window={"start": start_str, "end": end_str},
            total_candidates=len(candidates),
            pre_event_count=pre_count,
            post_event_count=post_count,
            grd_candidate_count=grd_count,
            candidates=candidates,
        )

    def discover_sentinel_2(
        self,
        aoi: Optional[Any] = None,
        flood_date: str = "2024-08-26",
        days_before: int = 15,
        days_after: int = 7,
        max_results: int = 20,
    ) -> Sentinel2DiscoveryResponse:
        """
        Discover candidate Sentinel-2 multispectral optical scenes around a flood event date.

        Evaluates acquisition timing, cloud coverage percentage, and atmospheric processing level (Level-2A BOA vs Level-1C TOA)
        to assess optical suitability for multispectral water/debris analysis.
        """
        # Resolve and normalize AOI
        if aoi is None:
            resolved_aoi = DEFAULT_NEPAL_AOI
        else:
            resolved_aoi = normalize_and_validate_aoi(aoi)

        # Parse event date and establish temporal search window
        event_d = date.fromisoformat(flood_date.strip())
        event_dt = datetime(event_d.year, event_d.month, event_d.day, tzinfo=timezone.utc)

        start_d = event_d - timedelta(days=days_before)
        end_d = event_d + timedelta(days=days_after)

        start_str = start_d.isoformat()
        end_str = end_d.isoformat()

        # Query catalog or cache for Sentinel-2 products
        scenes = self._fetch_scenes(
            aoi=resolved_aoi,
            start_date=start_str,
            end_date=end_str,
            collection="SENTINEL-2",
            max_results=max_results,
        )

        candidates: List[Sentinel2Candidate] = []
        pre_count = 0
        post_count = 0
        usable_count = 0
        cloud_values: List[float] = []

        for s in scenes:
            s_dt = s.acquisition_datetime
            if s_dt.tzinfo is None:
                s_dt = s_dt.replace(tzinfo=timezone.utc)

            offset_days = round((s_dt - event_dt).total_seconds() / 86400.0, 2)
            timing = "pre_event" if s_dt < event_dt else "post_event"
            if timing == "pre_event":
                pre_count += 1
            else:
                post_count += 1

            # Cloud cover evaluation
            cloud = s.cloud_cover
            if cloud is not None:
                cloud_values.append(cloud)
                if cloud <= 20.0:
                    usability = "clear"
                elif cloud <= 50.0:
                    usability = "moderate"
                else:
                    usability = "cloudy"
                
                if cloud <= 50.0:
                    usable_count += 1
            else:
                usability = "unknown"

            # Check for Level-2A (Bottom-of-Atmosphere surface reflectance)
            p_type = (s.product_type or "").upper()
            p_id = (s.product_id or "").upper()
            is_l2a = "L2A" in p_type or "MSIL2A" in p_id or "S2MSI2A" in p_type

            candidates.append(
                Sentinel2Candidate(
                    scene=s,
                    timing=timing,
                    days_from_event=offset_days,
                    cloud_cover=cloud,
                    is_l2a=is_l2a,
                    optical_usability=usability,
                )
            )

        # Sort candidates chronologically
        candidates.sort(key=lambda c: c.scene.acquisition_datetime)

        avg_cloud = round(sum(cloud_values) / len(cloud_values), 2) if cloud_values else None

        return Sentinel2DiscoveryResponse(
            collection="SENTINEL-2",
            flood_date=flood_date,
            temporal_window={"start": start_str, "end": end_str},
            total_candidates=len(candidates),
            pre_event_count=pre_count,
            post_event_count=post_count,
            usable_candidate_count=usable_count,
            average_cloud_cover=avg_cloud,
            candidates=candidates,
        )

    def route_analysis(
        self,
        aoi: Optional[Any] = None,
        flood_date: str = "2024-08-26",
        days_before: int = 15,
        days_after: int = 7,
        max_results: int = 20,
    ) -> RoutingDecisionResponse:
        """
        Execute full satellite discovery across both Sentinel-1 and Sentinel-2,
        then pass candidates through the Smart Satellite Router to decide optimal data strategy.
        """
        # Resolve and normalize AOI
        if aoi is None:
            resolved_aoi = DEFAULT_NEPAL_AOI
        else:
            resolved_aoi = normalize_and_validate_aoi(aoi)

        # Discover candidates from both constellations
        s1_data = self.discover_sentinel_1(
            aoi=resolved_aoi,
            flood_date=flood_date,
            days_before=days_before,
            days_after=days_after,
            max_results=max_results,
        )
        s2_data = self.discover_sentinel_2(
            aoi=resolved_aoi,
            flood_date=flood_date,
            days_before=days_before,
            days_after=days_after,
            max_results=max_results,
        )

        # Produce routing decision
        return self.router.route(
            aoi=resolved_aoi,
            flood_date=flood_date,
            s1_data=s1_data,
            s2_data=s2_data,
        )



# Global singleton instance for FastAPI dependency injection
_satellite_service_instance = SatelliteDiscoveryService()


def get_satellite_service() -> SatelliteDiscoveryService:
    """Dependency provider for route handlers."""
    return _satellite_service_instance
