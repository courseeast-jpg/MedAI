from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    timestamp: str
    dataset: str
    attempted: int
    processed: int
    written: int
    written_with_review: int
    external_quota_blocked: int
    hard_failures: int
    avg_confidence: float
    route_distribution_requested: dict[str, int]
    route_distribution_actual: dict[str, int]
    review_counts: dict[str, int]
    duration_sec: float
    determinism: dict
    quota_behavior: dict

    def to_dict(self) -> dict:
        return asdict(self)
