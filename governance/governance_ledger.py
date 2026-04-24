from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class GovernanceLedger:
    """Append-only governance audit ledger."""

    def __init__(self, path: Path | str = "data/audit/governance_ledger.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        event_type: str,
        record_id: str | None = None,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "record_id": record_id,
            "action": action,
            "details": details or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return event

