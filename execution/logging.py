"""Minimal audit logging for execution pipeline runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class AuditLogger:
    """Append-only JSONL audit log with the required Phase 1 fields."""

    def __init__(self, path: Path | str = "data/audit/execution.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, extractor: str, entity_count: int, confidence: float, outcome: str) -> dict[str, Any]:
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "extractor": extractor,
            "entity_count": int(entity_count),
            "confidence": float(confidence),
            "outcome": outcome,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        logger.info(
            "execution audit extractor={} entities={} confidence={:.2f} outcome={}",
            extractor,
            entity_count,
            confidence,
            outcome,
        )
        return event
