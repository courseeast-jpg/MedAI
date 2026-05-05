"""Stage-level execution metrics collector."""

from __future__ import annotations


CONNECTOR_COSTS = {
    "spacy": 0.0,
    "gemini": 0.02,
    "phi3": 0.005,
}

DEFAULT_CONNECTOR_PROFILES = {
    "spacy": {"latency_ms": 5.0, "confidence": 0.78, "success_rate": 0.99},
    "gemini": {"latency_ms": 120.0, "confidence": 0.9, "success_rate": 0.9},
    "phi3": {"latency_ms": 25.0, "confidence": 0.76, "success_rate": 0.95},
}


class PipelineMetrics:
    """Lightweight in-memory metrics for execution pipeline runs."""

    def __init__(self):
        self.total_records_processed = 0
        self.accepted_count = 0
        self.review_count = 0
        self.rejected_count = 0
        self.promoted_count = 0
        self.spacy_count = 0
        self.gemini_count = 0
        self.fallback_count = 0
        self.failure_count = 0
        self._confidence_total = 0.0
        self._agreement_total = 0.0
        self._agreement_samples = 0
        self._connector_stats = {
            name: {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "latency_total": 0.0,
                "confidence_total": 0.0,
                "cost_total": 0.0,
            }
            for name in CONNECTOR_COSTS
        }

    def record_routing(self, *, extractor_actual: str, fallback_used: bool, failure_count: int) -> None:
        if extractor_actual == "spacy":
            self.spacy_count += 1
        elif extractor_actual == "gemini":
            self.gemini_count += 1
        if fallback_used:
            self.fallback_count += 1
        self.failure_count += int(failure_count)

    def record_connector_result(
        self,
        *,
        connector: str,
        latency_ms: float,
        confidence: float,
        success: bool,
        cost_estimate: float | None = None,
    ) -> None:
        if connector not in self._connector_stats:
            self._connector_stats[connector] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "latency_total": 0.0,
                "confidence_total": 0.0,
                "cost_total": 0.0,
            }
        stats = self._connector_stats[connector]
        stats["attempts"] += 1
        if success:
            stats["successes"] += 1
            stats["latency_total"] += float(latency_ms)
            stats["confidence_total"] += float(confidence)
        else:
            stats["failures"] += 1
        stats["cost_total"] += float(cost_estimate if cost_estimate is not None else self.get_cost_estimate(connector))

    def get_cost_estimate(self, connector: str) -> float:
        return float(CONNECTOR_COSTS.get(connector, 0.0))

    def connector_profile(self, connector: str) -> dict[str, float]:
        stats = self._connector_stats.get(connector)
        defaults = DEFAULT_CONNECTOR_PROFILES.get(
            connector,
            {"latency_ms": 50.0, "confidence": 0.7, "success_rate": 0.9},
        )
        if not stats or stats["attempts"] == 0:
            return {
                "avg_latency_ms": defaults["latency_ms"],
                "avg_confidence": defaults["confidence"],
                "cost_estimate": self.get_cost_estimate(connector),
                "success_rate": defaults["success_rate"],
                "attempts": 0.0,
            }

        success_count = max(stats["successes"], 1)
        return {
            "avg_latency_ms": round(stats["latency_total"] / success_count, 3) if stats["successes"] else defaults["latency_ms"],
            "avg_confidence": round(stats["confidence_total"] / success_count, 3) if stats["successes"] else defaults["confidence"],
            "cost_estimate": round(stats["cost_total"] / max(stats["attempts"], 1), 5),
            "success_rate": round(stats["successes"] / max(stats["attempts"], 1), 3),
            "attempts": float(stats["attempts"]),
        }

    def record_validation(self, *, record_count: int, validation_status: str, confidence: float, agreement_score: float) -> None:
        self.total_records_processed += int(record_count)
        self._confidence_total += float(confidence) * int(record_count)
        self._agreement_total += float(agreement_score)
        self._agreement_samples += 1
        if validation_status == "accepted":
            self.accepted_count += int(record_count)
        elif validation_status == "needs_review":
            self.review_count += int(record_count)
        elif validation_status == "rejected":
            self.rejected_count += int(record_count)

    def record_review(self, *, review_count: int) -> None:
        self.review_count += int(review_count)

    def record_promotion(self, *, promoted_count: int) -> None:
        self.promoted_count += int(promoted_count)

    def snapshot(self) -> dict[str, float | int | dict[str, float]]:
        total = self.total_records_processed
        return {
            "total_records_processed": total,
            "accepted_count": self.accepted_count,
            "review_count": self.review_count,
            "rejected_count": self.rejected_count,
            "promoted_count": self.promoted_count,
            "spacy_count": self.spacy_count,
            "gemini_count": self.gemini_count,
            "fallback_count": self.fallback_count,
            "failure_count": self.failure_count,
            "avg_confidence": round(self._confidence_total / total, 3) if total else 0.0,
            "avg_agreement_score": round(self._agreement_total / self._agreement_samples, 3) if self._agreement_samples else 0.0,
            "avg_latency_per_connector": {
                connector: self.connector_profile(connector)["avg_latency_ms"]
                for connector in self._connector_stats
            },
            "avg_confidence_per_connector": {
                connector: self.connector_profile(connector)["avg_confidence"]
                for connector in self._connector_stats
            },
            "cost_estimate_per_connector": {
                connector: self.connector_profile(connector)["cost_estimate"]
                for connector in self._connector_stats
            },
            "success_rate_per_connector": {
                connector: self.connector_profile(connector)["success_rate"]
                for connector in self._connector_stats
            },
        }
