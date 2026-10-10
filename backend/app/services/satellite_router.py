from typing import Any, Dict, List, Optional
from app.schemas.satellite import (
    RoutingDecisionResponse,
    Sentinel1DiscoveryResponse,
    Sentinel1Suitability,
    Sentinel2DiscoveryResponse,
    Sentinel2Suitability,
)


class SmartSatelliteRouter:
    """
    Intelligent decision engine that evaluates discovered Sentinel-1 and Sentinel-2
    candidates to determine the optimal satellite data strategy for flood mapping.
    """

    def evaluate_sentinel_1(self, s1_data: Sentinel1DiscoveryResponse) -> Sentinel1Suitability:
        """Evaluate whether Sentinel-1 radar candidates are suitable for change detection."""
        pre_grd = [c for c in s1_data.candidates if c.timing == "pre_event" and c.is_grd]
        post_grd = [c for c in s1_data.candidates if c.timing == "post_event" and c.is_grd]

        # Check for geometry consistency (matching relative orbit track or orbit direction)
        pre_orbits = {c.relative_orbit for c in pre_grd if c.relative_orbit is not None}
        post_orbits = {c.relative_orbit for c in post_grd if c.relative_orbit is not None}
        common_orbits = pre_orbits.intersection(post_orbits)

        pre_dirs = {c.orbit_direction for c in pre_grd if c.orbit_direction}
        post_dirs = {c.orbit_direction for c in post_grd if c.orbit_direction}
        common_dirs = pre_dirs.intersection(post_dirs)

        has_geom_pair = bool(common_orbits or common_dirs)
        is_suitable = bool(len(pre_grd) >= 1 and len(post_grd) >= 1 and has_geom_pair)

        if is_suitable:
            if common_orbits:
                orbit_str = f"matching relative orbit track(s) {sorted(list(common_orbits))}"
            else:
                orbit_str = f"compatible orbit direction(s) {sorted(list(common_dirs))}"
            assessment = (
                f"Optimal SAR change-detection pair available with {orbit_str} "
                f"and calibrated Ground Range Detected (GRD) intensity products."
            )
        elif len(s1_data.candidates) == 0:
            assessment = "No Sentinel-1 radar acquisitions found within the search window."
        elif s1_data.grd_candidate_count == 0:
            assessment = "Radar acquisitions found, but lack multi-looked Ground Range Detected (GRD) products."
        elif len(pre_grd) == 0:
            assessment = "Missing pre-disaster baseline Sentinel-1 GRD acquisitions for reference."
        elif len(post_grd) == 0:
            assessment = "Missing post-disaster flood impact Sentinel-1 GRD acquisitions."
        else:
            assessment = "Pre- and post-flood radar acquisitions exist but lack compatible viewing geometries."

        return Sentinel1Suitability(
            is_suitable=is_suitable,
            total_candidates=s1_data.total_candidates,
            grd_candidates=s1_data.grd_candidate_count,
            pre_event_count=s1_data.pre_event_count,
            post_event_count=s1_data.post_event_count,
            has_geometry_pairs=has_geom_pair,
            assessment=assessment,
        )

    def evaluate_sentinel_2(self, s2_data: Sentinel2DiscoveryResponse) -> Sentinel2Suitability:
        """Evaluate whether Sentinel-2 optical candidates have acceptable cloud conditions."""
        post_scenes = [c for c in s2_data.candidates if c.timing == "post_event"]
        post_clouds = [c.cloud_cover for c in post_scenes if c.cloud_cover is not None]

        best_post_cloud = min(post_clouds) if post_clouds else None

        # Usability thresholds:
        # <= 25%: clear and highly usable
        # 25% - 50%: moderate cloud, usable with cloud masking
        # > 50%: obscured by clouds, unsuitable
        if best_post_cloud is not None and best_post_cloud <= 50.0:
            is_suitable = True
            quality = "clear" if best_post_cloud <= 20.0 else "moderate"
            assessment = (
                f"Usable post-event optical imagery available with {quality} conditions "
                f"(best post-event cloud cover: {best_post_cloud:.1f}%)."
            )
        elif len(s2_data.candidates) == 0:
            is_suitable = False
            assessment = "No Sentinel-2 optical acquisitions found within the search window."
        elif len(post_scenes) == 0:
            is_suitable = False
            assessment = "No post-event Sentinel-2 acquisitions found; cannot observe disaster impact."
        else:
            is_suitable = False
            cloud_disp = f"{best_post_cloud:.1f}%" if best_post_cloud is not None else "unknown"
            assessment = (
                f"Severe cloud cover obscures ground observation (best post-event cloud cover: {cloud_disp}). "
                "Optical sensors cannot reliably detect surface water under these conditions."
            )

        return Sentinel2Suitability(
            is_suitable=is_suitable,
            total_candidates=s2_data.total_candidates,
            usable_candidates=s2_data.usable_candidate_count,
            pre_event_count=s2_data.pre_event_count,
            post_event_count=s2_data.post_event_count,
            average_cloud_cover=s2_data.average_cloud_cover,
            best_post_event_cloud_cover=best_post_cloud,
            assessment=assessment,
        )

    def route(
        self,
        aoi: Dict[str, Any],
        flood_date: str,
        s1_data: Sentinel1DiscoveryResponse,
        s2_data: Sentinel2DiscoveryResponse,
    ) -> RoutingDecisionResponse:
        """
        Generate structured routing decision from Sentinel-1 and Sentinel-2 candidate evaluations.
        """
        s1_eval = self.evaluate_sentinel_1(s1_data)
        s2_eval = self.evaluate_sentinel_2(s2_data)

        warnings: List[str] = []

        # Decision Matrix
        if s1_eval.is_suitable and s2_eval.is_suitable:
            strategy = "sentinel-1-and-sentinel-2"
            reason = (
                "Dual-sensor strategy recommended: Sentinel-1 radar provides robust all-weather ground "
                "penetration with matched viewing geometry, while Sentinel-2 provides clear optical imagery "
                "to assist in multispectral validation and debris identification."
            )
        elif s1_eval.is_suitable and not s2_eval.is_suitable:
            strategy = "sentinel-1"
            reason = (
                "Sentinel-1 radar strategy selected: Sentinel-1 provides suitable pre/post change detection pairs "
                f"with consistent viewing geometry, whereas Sentinel-2 optical imagery is unsuitable ({s2_eval.assessment})."
            )
            warnings.append("Optical imagery unavailable or cloud-obscured; multispectral validation will be bypassed.")
        elif not s1_eval.is_suitable and s2_eval.is_suitable:
            strategy = "sentinel-2"
            reason = (
                "Sentinel-2 optical strategy selected: Suitable optical imagery with acceptable cloud cover exists, "
                f"whereas suitable Sentinel-1 radar change-detection pairs are unavailable ({s1_eval.assessment})."
            )
            warnings.append("SAR change detection unavailable; flood extent will rely solely on optical spectral indices.")
        else:
            strategy = "no-suitable-data"
            reason = (
                "No suitable satellite data found: Neither Sentinel-1 radar nor Sentinel-2 optical imagery meet "
                "the necessary temporal, geometric, or cloud quality criteria for reliable flood damage analysis."
            )
            warnings.append("Cannot perform automated flood mapping without suitable satellite acquisitions.")

        return RoutingDecisionResponse(
            strategy=strategy,
            reason=reason,
            flood_date=flood_date,
            aoi=aoi,
            sentinel_1=s1_eval,
            sentinel_2=s2_eval,
            warnings=warnings,
        )


# Global singleton instance
_router_instance = SmartSatelliteRouter()


def get_satellite_router() -> SmartSatelliteRouter:
    """Dependency provider for router."""
    return _router_instance
