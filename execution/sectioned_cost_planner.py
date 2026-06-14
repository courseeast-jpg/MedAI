"""Cost planning for 17C-R2-R13 autonomous full/sectioned extraction."""
from __future__ import annotations

from dataclasses import dataclass

from execution import cost_chunk_planner as base


@dataclass(frozen=True)
class CostPlan:
    total_cap_usd: float
    per_chunk_cap_usd: float
    estimated_total_cost_usd: float
    selected_chunk_size: int
    full_schema_safe: bool
    sectioned_safe: bool
    subsection_safe: bool


def build_cost_plan(per_doc_input_tokens: list[int], *, total_cap_usd: float,
                    per_chunk_cap_usd: float, full_max_output_tokens: int,
                    section_max_output_tokens: int, input_usd_per_m: float,
                    output_usd_per_m: float, max_chunk_size: int = 25) -> CostPlan:
    full_total = base.estimate_total_cost(
        per_doc_input_tokens,
        full_max_output_tokens,
        input_usd_per_m,
        output_usd_per_m,
    )
    selected = base.select_chunk_size(
        per_doc_input_tokens,
        full_max_output_tokens,
        input_usd_per_m,
        output_usd_per_m,
        per_chunk_cap_usd,
        max_chunk_size,
    )
    section_one = base.per_request_cost(
        max(per_doc_input_tokens or [1]),
        section_max_output_tokens,
        input_usd_per_m,
        output_usd_per_m,
    )
    return CostPlan(
        total_cap_usd=total_cap_usd,
        per_chunk_cap_usd=per_chunk_cap_usd,
        estimated_total_cost_usd=full_total,
        selected_chunk_size=selected,
        full_schema_safe=full_total <= total_cap_usd and selected > 0,
        sectioned_safe=section_one <= per_chunk_cap_usd,
        subsection_safe=section_one <= per_chunk_cap_usd,
    )


__all__ = ["CostPlan", "build_cost_plan"]
