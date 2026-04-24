"""Stage-level execution metrics collector."""

from __future__ import annotations


class PipelineMetrics:
    """Lightweight in-memory metrics for execution pipeline runs."""

    def __init__(self):
        self.total_records_processed = 0
        self.accepted_count = 0
        self.review_count = 0
        self.rejected_count = 0
        self.promoted_count = 0
        self._confidence_total = 0.0
        self._agreement_total = 0.0
        self._agreement_samples = 0

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

    def snapshot(self) -> dict[str, float | int]:
        total = self.total_records_processed
        return {
            "total_records_processed": total,
            "accepted_count": self.accepted_count,
            "review_count": self.review_count,
            "rejected_count": self.rejected_count,
            "promoted_count": self.promoted_count,
            "avg_confidence": round(self._confidence_total / total, 3) if total else 0.0,
            "avg_agreement_score": round(self._agreement_total / self._agreement_samples, 3) if self._agreement_samples else 0.0,
        }
