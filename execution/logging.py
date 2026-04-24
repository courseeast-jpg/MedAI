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
        self._confidence_total = 0.0
        self._review_count = 0

    def record(self, *, extractor_route: str, confidence: float, outcome: str) -> None:
        self.total_jobs += 1
        if extractor_route == "spacy":
            self.spacy_count += 1
        elif extractor_route == "gemini":
            self.gemini_count += 1
        self._confidence_total += float(confidence)
        if outcome == "queued_for_review":
            self._review_count += 1

    def snapshot(self) -> dict[str, float | int]:
        total = self.total_jobs
        avg_confidence = round(self._confidence_total / total, 3) if total else 0.0
        review_rate = round(self._review_count / total, 3) if total else 0.0
        return {
            "total_jobs": total,
            "spacy_count": self.spacy_count,
            "gemini_count": self.gemini_count,
            "avg_confidence": avg_confidence,
            "review_rate": review_rate,
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
    ) -> dict[str, Any]:
        self.metrics.record(extractor_route=extractor_route, confidence=confidence, outcome=outcome)
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "extractor": extractor,
            "extractor_route": extractor_route,
            "extractor_actual": extractor_actual,
            "entity_count": int(entity_count),
            "confidence": float(confidence),
            "outcome": outcome,
            "final_status": outcome,
        }
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
