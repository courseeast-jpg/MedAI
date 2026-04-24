"""Minimal audit logging for execution pipeline runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class ExecutionMetrics:
    """In-memory aggregate metrics for execution runs."""

    def __init__(self):
        self.total_jobs = 0
        self.spacy_count = 0
        self.gemini_count = 0
        self.fallback_count = 0
        self.failure_count = 0
        self._confidence_total = 0.0
        self._review_count = 0
        self.accepted_count = 0
        self.review_count = 0
        self.rejected_count = 0
        self.validation_error_count = 0
        self._confidence_by_status = {
            "accepted": {"count": 0, "total": 0.0},
            "needs_review": {"count": 0, "total": 0.0},
            "rejected": {"count": 0, "total": 0.0},
        }

    def record(
        self,
        *,
        extractor_route: str,
        extractor_actual: str,
        confidence: float,
        outcome: str,
        validation_status: str = "accepted",
        validation_error_count: int = 0,
        fallback_used: bool = False,
        failure_count: int = 0,
    ) -> None:
        self.total_jobs += 1
        if extractor_actual == "spacy":
            self.spacy_count += 1
        elif extractor_actual == "gemini":
            self.gemini_count += 1
        if fallback_used:
            self.fallback_count += 1
        self.failure_count += int(failure_count)
        confidence_value = float(confidence)
        self._confidence_total += confidence_value
        if outcome == "queued_for_review":
            self._review_count += 1
        self.validation_error_count += int(validation_error_count)

        if validation_status == "accepted":
            self.accepted_count += 1
        elif validation_status == "needs_review":
            self.review_count += 1
        elif validation_status == "rejected":
            self.rejected_count += 1

        if validation_status in self._confidence_by_status:
            status_bucket = self._confidence_by_status[validation_status]
            status_bucket["count"] += 1
            status_bucket["total"] += confidence_value

    def snapshot(self) -> dict[str, float | int]:
        total = self.total_jobs
        avg_confidence = round(self._confidence_total / total, 3) if total else 0.0
        review_rate = round(self._review_count / total, 3) if total else 0.0
        avg_confidence_by_status = {}
        for status, bucket in self._confidence_by_status.items():
            count = bucket["count"]
            avg_confidence_by_status[status] = round(bucket["total"] / count, 3) if count else 0.0
        return {
            "total_jobs": total,
            "spacy_count": self.spacy_count,
            "gemini_count": self.gemini_count,
            "fallback_count": self.fallback_count,
            "failure_count": self.failure_count,
            "avg_confidence": avg_confidence,
            "review_rate": review_rate,
            "accepted_count": self.accepted_count,
            "review_count": self.review_count,
            "rejected_count": self.rejected_count,
            "validation_error_count": self.validation_error_count,
            "avg_confidence_by_status": avg_confidence_by_status,
        }


class AuditLogger:
    """Append-only JSONL audit log with Phase 1 required fields."""

    def __init__(self, path: Path | str = "data/audit/execution.jsonl", metrics: ExecutionMetrics | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics or ExecutionMetrics()

    def log(
        self,
        *,
        extractor: str,
        extractor_route: str,
        extractor_actual: str,
        entity_count: int,
        confidence: float,
        outcome: str,
        validation_status: str = "accepted",
        validation_error_count: int = 0,
        fallback_used: bool = False,
        failure_count: int = 0,
        run_id: str | None = None,
        document_id: str | None = None,
        fallback_reason: str | None = None,
        confidence_band: str | None = None,
        quality_gate_decision: str | None = None,
        error_category: str | None = None,
    ) -> dict[str, Any]:
        self.metrics.record(
            extractor_route=extractor_route,
            extractor_actual=extractor_actual,
            confidence=confidence,
            outcome=outcome,
            validation_status=validation_status,
            validation_error_count=validation_error_count,
            fallback_used=fallback_used,
            failure_count=failure_count,
        )
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "document_id": document_id,
            "extractor": extractor,
            "extractor_route": extractor_route,
            "extractor_actual": extractor_actual,
            "entity_count": int(entity_count),
            "confidence": float(confidence),
            "confidence_band": confidence_band,
            "fallback_reason": fallback_reason,
            "quality_gate_decision": quality_gate_decision or outcome,
            "validation_status": validation_status,
            "validation_error_count": int(validation_error_count),
            "fallback_used": fallback_used,
            "failure_count": int(failure_count),
            "outcome": outcome,
            "final_status": outcome,
        }
        if error_category is not None:
            event["error_category"] = error_category
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        logger.info(
            "execution audit extractor_route={} extractor_actual={} extractor={} entity_count={} confidence={:.2f} final_status={}",
            extractor_route,
            extractor_actual,
            extractor,
            entity_count,
            confidence,
            outcome,
        )
        return event
