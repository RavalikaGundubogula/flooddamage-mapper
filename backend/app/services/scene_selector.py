import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from app.schemas.analysis import normalize_and_validate_aoi
from app.schemas.satellite import (
    Sentinel1Candidate,
    Sentinel2Candidate,
)
from app.schemas.scene_selection import (
    SceneSelectionResponse,
    SelectedScenePair,
)
from app.services.satellite_service import (
    DEFAULT_NEPAL_AOI,
    SatelliteDiscoveryService,
    get_satellite_service,
)

logger = logging.getLogger(__name__)


class SceneSelectionService:
    """
    Dedicated service for selecting optimal before and after satellite scenes
    for disaster change-detection analysis.
    """

    def __init__(self, discovery_service: Optional[SatelliteDiscoveryService] = None) -> None:
        self.discovery = discovery_service or get_satellite_service()

    def select_sentinel_1_pair(
        self,
        candidates: List[Sentinel1Candidate],
    ) -> Tuple[Optional[SelectedScenePair], List[str]]:
        """
        Select optimal pre-event and post-event Sentinel-1 radar pair.
        Prioritizes Ground Range Detected (GRD) products with identical relative orbit tracks
        and look geometry to eliminate terrain layover and shadow false positives.
        """
        warnings: List[str] = []

        # 1. Filter for GRD products (multi-looked intensity suitable for water detection)
        grd_candidates = [c for c in candidates if c.is_grd]
        if not grd_candidates:
            # If no GRD, check if any candidates exist at all
            if candidates:
                warnings.append("No GRD products available in Sentinel-1 candidates (found RAW/SLC only).")
            return None, warnings

        pre_grd = [c for c in grd_candidates if c.timing == "pre_event"]
        post_grd = [c for c in grd_candidates if c.timing == "post_event"]

        if not pre_grd:
            warnings.append("Missing pre-event baseline Sentinel-1 GRD acquisitions.")
            return None, warnings

        if not post_grd:
            warnings.append("Missing post-event flood impact Sentinel-1 GRD acquisitions.")
            return None, warnings

        # 2. Score candidate pairs for orbital and geometric compatibility
        best_pair = None
        best_tier = 999
        lowest_penalty = float("inf")

        for pre in pre_grd:
            for post in post_grd:
                # Same relative orbit track AND same direction (Tier 1 - Optimal)
                same_rel_orbit = (
                    pre.relative_orbit is not None
                    and post.relative_orbit is not None
                    and pre.relative_orbit == post.relative_orbit
                )
                same_dir = bool(
                    pre.orbit_direction
                    and post.orbit_direction
                    and pre.orbit_direction == post.orbit_direction
                )

                time_penalty = abs(pre.days_from_event) + abs(post.days_from_event)

                if same_rel_orbit and same_dir:
                    tier = 1
                    penalty = time_penalty
                elif same_dir:
                    # Tier 2: Same direction but different relative orbit tracks (incidence angle shift)
                    tier = 2
                    penalty = 1000.0 + time_penalty
                else:
                    # Incompatible: Opposing look directions (Ascending vs Descending)
                    continue

                if tier < best_tier or (tier == best_tier and penalty < lowest_penalty):
                    best_tier = tier
                    lowest_penalty = penalty
                    best_pair = (pre, post)

        if not best_pair:
            warnings.append(
                "No Sentinel-1 pairs found with compatible orbit direction (all opposing Ascending/Descending)."
            )
            return None, warnings

        chosen_pre, chosen_post = best_pair
        orbit_comp = (best_tier == 1)

        if not orbit_comp:
            warnings.append(
                f"Selected Sentinel-1 pair shares orbit direction ({chosen_pre.orbit_direction}) "
                f"but differs in relative orbit tracks ({chosen_pre.relative_orbit} vs {chosen_post.relative_orbit}). "
                "Minor geometric distortion may occur in steep terrain."
            )

        # Calculate temporal separation
        pre_dt = chosen_pre.scene.acquisition_datetime
        post_dt = chosen_post.scene.acquisition_datetime
        temporal_sep = round(abs((post_dt - pre_dt).total_seconds()) / 86400.0, 2)

        quality_notes = (
            f"Pre: {chosen_pre.scene.product_id} ({chosen_pre.days_from_event}d), "
            f"Post: {chosen_post.scene.product_id} (+{chosen_post.days_from_event}d). "
            f"Polarization: {chosen_pre.polarization or 'Dual-Pol'}. "
            f"Relative Orbit: {chosen_pre.relative_orbit}."
        )

        pair_result = SelectedScenePair(
            satellite="SENTINEL-1",
            before_scene=chosen_pre.scene,
            after_scene=chosen_post.scene,
            orbit_compatibility=orbit_comp,
            relative_orbit=chosen_pre.relative_orbit if orbit_comp else None,
            orbit_direction=chosen_pre.orbit_direction,
            temporal_separation_days=temporal_sep,
            before_days_from_event=chosen_pre.days_from_event,
            after_days_from_event=chosen_post.days_from_event,
            quality_notes=quality_notes,
        )

        return pair_result, warnings

    def select_sentinel_2_pair(
        self,
        candidates: List[Sentinel2Candidate],
    ) -> Tuple[Optional[SelectedScenePair], List[str]]:
        """
        Select optimal pre-event and post-event Sentinel-2 optical pair.
        Prioritizes lowest cloud cover and Level-2A surface reflectance products.
        """
        warnings: List[str] = []

        pre_scenes = [c for c in candidates if c.timing == "pre_event"]
        post_scenes = [c for c in candidates if c.timing == "post_event"]

        if not pre_scenes:
            warnings.append("No pre-event Sentinel-2 optical scenes found in search window.")
            return None, warnings

        if not post_scenes:
            warnings.append("No post-event Sentinel-2 optical scenes found in search window.")
            return None, warnings

        # Rank post-event: prioritize cloud <= 50%, prefer L2A, then lowest cloud
        post_usable = [c for c in post_scenes if c.cloud_cover is not None and c.cloud_cover <= 50.0]
        if not post_usable:
            best_cloud = min((c.cloud_cover for c in post_scenes if c.cloud_cover is not None), default=100.0)
            warnings.append(
                f"All post-event Sentinel-2 scenes have excessive cloud cover (best: {best_cloud:.1f}% > 50%)."
            )
            return None, warnings

        # Sort post-event by: L2A preference (desc), cloud cover (asc), closeness to event (asc)
        post_usable.sort(
            key=lambda c: (
                not c.is_l2a,
                c.cloud_cover if c.cloud_cover is not None else 100.0,
                abs(c.days_from_event),
            )
        )
        chosen_post = post_usable[0]

        # Rank pre-event similarly
        pre_usable = [c for c in pre_scenes if c.cloud_cover is not None and c.cloud_cover <= 50.0]
        if not pre_usable:
            pre_usable = pre_scenes  # Fall back to best available pre-event if needed

        pre_usable.sort(
            key=lambda c: (
                not c.is_l2a,
                c.cloud_cover if c.cloud_cover is not None else 100.0,
                abs(c.days_from_event),
            )
        )
        chosen_pre = pre_usable[0]

        pre_dt = chosen_pre.scene.acquisition_datetime
        post_dt = chosen_post.scene.acquisition_datetime
        temporal_sep = round(abs((post_dt - pre_dt).total_seconds()) / 86400.0, 2)

        quality_notes = (
            f"Pre: {chosen_pre.scene.product_id} (Cloud: {chosen_pre.cloud_cover}%, L2A={chosen_pre.is_l2a}), "
            f"Post: {chosen_post.scene.product_id} (Cloud: {chosen_post.cloud_cover}%, L2A={chosen_post.is_l2a})."
        )

        pair_result = SelectedScenePair(
            satellite="SENTINEL-2",
            before_scene=chosen_pre.scene,
            after_scene=chosen_post.scene,
            orbit_compatibility=True,  # Optical orthorectified products share UTM grid
            relative_orbit=None,
            orbit_direction=None,
            temporal_separation_days=temporal_sep,
            before_days_from_event=chosen_pre.days_from_event,
            after_days_from_event=chosen_post.days_from_event,
            quality_notes=quality_notes,
        )

        return pair_result, warnings

    def select_scenes(
        self,
        aoi: Optional[Any] = None,
        flood_date: str = "2026-08-26",
        days_before: int = 15,
        days_after: int = 7,
        max_results: int = 20,
    ) -> SceneSelectionResponse:
        """
        Execute end-to-end scene selection based on router guidance:
        1. Query candidate scenes for both constellations.
        2. Evaluate data strategy via Smart Satellite Router.
        3. Pick optimal before/after pairs matching the strategy.
        4. Return analysis-ready scene references.
        """
        resolved_aoi = DEFAULT_NEPAL_AOI if aoi is None else normalize_and_validate_aoi(aoi)

        # 1. Discover candidates
        s1_res = self.discovery.discover_sentinel_1(
            aoi=resolved_aoi,
            flood_date=flood_date,
            days_before=days_before,
            days_after=days_after,
            max_results=max_results,
        )
        s2_res = self.discovery.discover_sentinel_2(
            aoi=resolved_aoi,
            flood_date=flood_date,
            days_before=days_before,
            days_after=days_after,
            max_results=max_results,
        )

        # 2. Get routing strategy
        routing_decision = self.discovery.router.route(
            aoi=resolved_aoi,
            flood_date=flood_date,
            s1_data=s1_res,
            s2_data=s2_res,
        )

        strategy = routing_decision.strategy
        all_warnings: List[str] = list(routing_decision.warnings)

        s1_pair: Optional[SelectedScenePair] = None
        s2_pair: Optional[SelectedScenePair] = None

        if strategy in ("sentinel-1", "sentinel-1-and-sentinel-2"):
            s1_pair, s1_warns = self.select_sentinel_1_pair(s1_res.candidates)
            all_warnings.extend(s1_warns)

        if strategy in ("sentinel-2", "sentinel-1-and-sentinel-2"):
            s2_pair, s2_warns = self.select_sentinel_2_pair(s2_res.candidates)
            all_warnings.extend(s2_warns)

        # Determine overall selection status
        has_selection = bool(s1_pair or s2_pair)
        status = "selected" if has_selection else "insufficient_data"

        if status == "selected":
            selected_sources = []
            if s1_pair:
                selected_sources.append(
                    f"Sentinel-1 (Orbit {s1_pair.relative_orbit or s1_pair.orbit_direction}, {s1_pair.temporal_separation_days}d delta)"
                )
            if s2_pair:
                selected_sources.append(
                    f"Sentinel-2 (Cloud {s2_pair.after_scene.cloud_cover}%, {s2_pair.temporal_separation_days}d delta)"
                )
            reason = f"Successfully selected before/after pairs for {', '.join(selected_sources)} based on {strategy} strategy."
        else:
            reason = (
                f"No suitable before/after pair could be established for {strategy} strategy. "
                f"Router assessment: {routing_decision.reason}"
            )

        return SceneSelectionResponse(
            status=status,
            event_date=flood_date,
            selected_strategy=strategy,
            selection_reason=reason,
            sentinel_1_selection=s1_pair,
            sentinel_2_selection=s2_pair,
            warnings=all_warnings,
            aoi=resolved_aoi,
        )


# Global singleton instance
_scene_selection_service = SceneSelectionService()


def get_scene_selection_service() -> SceneSelectionService:
    """Dependency provider for SceneSelectionService."""
    return _scene_selection_service
