"""Stage-level execution audit logging."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class StageAuditLogger:
    """Append-only JSONL audit log for pipeline stages."""

    def __init__(self, path: Path | str = "data/audit/pipeline_stages.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        record_id: str,
        stage: str,
        action: str,
        confidence: float,
        decision_reason: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "record_id": record_id,
            "stage": stage,
            "action": action,
            "confidence": float(confidence),
            "decision_reason": decision_reason,
        }
        if extra:
            event.update(extra)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return event
