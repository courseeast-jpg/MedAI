from __future__ import annotations

from dataclasses import asdict, dataclass

from execution.metrics import CONNECTOR_COSTS


@dataclass(frozen=True)
class RoutingEfficiency:
    intended_route: str
    actual_route: str
    fallback_reason: str | None
    route_mismatch_flag: bool
    estimated_cost_units: float
    saved_cost_units: float
    quota_block_avoided: bool
    confidence_band: str | None
    review_recommendation: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_routing_efficiency(
    *,
    intended_route: str | None,
    actual_route: str | None,
    fallback_reason: str | None,
    quota_block_avoided: bool,
    confidence_band: str | None,
    review_recommendation: str | None,
) -> RoutingEfficiency:
    intended_value = str(intended_route or "unknown")
    actual_value = str(actual_route or "unknown")
    intended_cost = estimate_route_cost(intended_value)
    actual_cost = estimate_route_cost(actual_value)
    return RoutingEfficiency(
        intended_route=intended_value,
        actual_route=actual_value,
        fallback_reason=fallback_reason,
        route_mismatch_flag=intended_value != actual_value,
        estimated_cost_units=actual_cost,
        saved_cost_units=round(max(intended_cost - actual_cost, 0.0), 5),
        quota_block_avoided=bool(quota_block_avoided),
        confidence_band=confidence_band,
        review_recommendation=review_recommendation,
    )


def estimate_route_cost(route: str | None) -> float:
    return round(float(CONNECTOR_COSTS.get(str(route or "unknown"), 0.0)), 5)
