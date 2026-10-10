from unittest.mock import patch
import pytest

from app.services.llm_router import (
    LLMRouter,
    RoutingComplexity,
    get_llm_router,
)


def test_router_complexity_estimation():
    """Verify complexity classifier assigns appropriate tiers based on data density."""
    router = get_llm_router()

    # 1. Low complexity
    low_res = router.estimate_complexity("summary_only", {})
    assert low_res == RoutingComplexity.LOW

    # 2. Medium complexity (standard report, moderate settlements)
    med_data = {
        "connectivity": {"isolated_settlement_count": 3, "blocked_road_count": 5},
        "geospatial": {"change_zone_count": 2},
    }
    med_res = router.estimate_complexity("situation_report", med_data)
    assert med_res == RoutingComplexity.MEDIUM

    # 3. High complexity (>20 settlements or >30 blocked roads)
    high_data = {
        "connectivity": {"isolated_settlement_count": 25, "blocked_road_count": 35},
        "geospatial": {"change_zone_count": 18},
    }
    high_res = router.estimate_complexity("situation_report", high_data)
    assert high_res == RoutingComplexity.HIGH


def test_router_model_selection_cost_awareness():
    """Verify that model selection prefers the most cost-effective tier for each complexity."""
    router = get_llm_router()

    tier_low = router.select_model(RoutingComplexity.LOW)
    assert tier_low.cost_tier == "economy"
    assert "lite" in tier_low.model_name.lower() or "flash" in tier_low.model_name.lower()

    tier_med = router.select_model(RoutingComplexity.MEDIUM)
    assert tier_med.cost_tier == "standard"

    tier_high = router.select_model(RoutingComplexity.HIGH)
    assert tier_high.cost_tier == "reasoning" or "pro" in tier_high.model_name.lower()


def test_router_graceful_degradation_without_api_keys():
    """Verify that when no LLM API keys are configured, the router degrades gracefully without crashing."""
    with patch.dict("os.environ", {}, clear=True):
        fresh_router = LLMRouter()
        res = fresh_router.generate(
            prompt="Generate situation report",
            system_prompt="Factuality prompt",
            complexity=RoutingComplexity.MEDIUM,
        )
        assert res.status == "llm_unavailable"
        assert res.llm_used is False
        assert "No LLM credentials" in (res.reason or "")


def test_router_preferred_provider_override():
    """Verify specifying a preferred provider overrides default registry while preserving cost tier."""
    router = get_llm_router()
    custom_tier = router.select_model(RoutingComplexity.MEDIUM, preferred_provider="claude")
    assert custom_tier.provider == "claude"
    assert custom_tier.cost_tier == "standard"
