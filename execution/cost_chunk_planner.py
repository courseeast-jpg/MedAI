"""Adaptive cost + chunk-size planning for the 17C-R2 live batch (17C-R2-R10).

Local-only arithmetic. No provider/network/billing/model call. Cost is estimated
conservatively at the FULL output-token ceiling (worst case: every response uses the
maximum), which is the cap-governing figure the live runner enforces.

Key helpers:
  - per_request_cost(...)         worst-case USD for one request at the output ceiling
  - estimate_total_cost(...)      worst-case USD for the whole batch
  - select_chunk_size(...)        largest chunk size <= max whose worst-case chunk cost
                                  stays within the per-chunk cap (0 if even one request
                                  exceeds the per-chunk cap)
  - worst_case_chunk_cost(...)    worst-case USD for a chunk of the given size

`select_chunk_size` is intentionally conservative: a chunk of size n is bounded by the
sum of the n most expensive requests, so ANY contiguous chunk of that size is guaranteed
to be within the cap. Total request count never changes — only the chunk size adapts.
"""
from __future__ import annotations


def per_request_cost(input_tokens: int, max_output_tokens: int,
                     input_usd_per_m: float, output_usd_per_m: float) -> float:
    return (input_tokens / 1_000_000 * input_usd_per_m
            + max_output_tokens / 1_000_000 * output_usd_per_m)


def estimate_total_cost(per_doc_input_tokens: "list[int]", max_output_tokens: int,
                        input_usd_per_m: float, output_usd_per_m: float) -> float:
    total_in = sum(per_doc_input_tokens)
    total_out = max_output_tokens * len(per_doc_input_tokens)
    return round(total_in / 1_000_000 * input_usd_per_m
                 + total_out / 1_000_000 * output_usd_per_m, 6)


def _sorted_costs(per_doc_input_tokens: "list[int]", max_output_tokens: int,
                  input_usd_per_m: float, output_usd_per_m: float) -> "list[float]":
    return sorted(
        (per_request_cost(t, max_output_tokens, input_usd_per_m, output_usd_per_m)
         for t in per_doc_input_tokens),
        reverse=True,
    )


def worst_case_chunk_cost(per_doc_input_tokens: "list[int]", max_output_tokens: int,
                          input_usd_per_m: float, output_usd_per_m: float,
                          chunk_size: int) -> float:
    if chunk_size <= 0 or not per_doc_input_tokens:
        return 0.0
    costs = _sorted_costs(per_doc_input_tokens, max_output_tokens, input_usd_per_m, output_usd_per_m)
    return round(sum(costs[:chunk_size]), 6)


def select_chunk_size(per_doc_input_tokens: "list[int]", max_output_tokens: int,
                      input_usd_per_m: float, output_usd_per_m: float,
                      per_chunk_cap_usd: float, max_chunk_size: int) -> int:
    """Largest chunk size in [1, max_chunk_size] whose worst-case cost <= per-chunk cap.
    Returns 0 when even a single request exceeds the per-chunk cap."""
    if not per_doc_input_tokens or max_chunk_size <= 0:
        return 0
    costs = _sorted_costs(per_doc_input_tokens, max_output_tokens, input_usd_per_m, output_usd_per_m)
    n = min(max_chunk_size, len(costs))
    while n >= 1:
        if round(sum(costs[:n]), 6) <= per_chunk_cap_usd:
            return n
        n -= 1
    return 0


__all__ = [
    "per_request_cost", "estimate_total_cost", "worst_case_chunk_cost", "select_chunk_size",
]
