"""Local in-memory enrichment queue for CKA-B06.

Queue reasons:
- pending_ddi_check
- blocked_high_ddi
- pending_medium_ack
- truth_resolution_review
- safe_mode_enrichment_disabled
- auto_promotion_blocked

No persistent file written in this block. No private queue payloads
written to public reports.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from clinical_knowledge.enrichment.models import EnrichmentQueueItem

_SALT = "medai_cka_b06_queue_v1"

QUEUE_REASONS = frozenset({
    "pending_ddi_check",
    "blocked_high_ddi",
    "pending_medium_ack",
    "truth_resolution_review",
    "safe_mode_enrichment_disabled",
    "auto_promotion_blocked",
})

QUEUE_STATUS_PENDING = "pending"
QUEUE_STATUS_RESOLVED = "resolved"
QUEUE_STATUS_DISCARDED = "discarded"


def _new_queue_id() -> str:
    return str(uuid.uuid4())


def _safe_queue_id(queue_id: str) -> str:
    digest = hashlib.sha256(f"{_SALT}:q:{queue_id}".encode()).hexdigest()[:16]
    return f"cka_q_{digest}"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class EnrichmentQueue:
    """Deterministic in-memory enrichment queue."""

    def __init__(self) -> None:
        self._items: List[EnrichmentQueueItem] = []

    def enqueue(
        self,
        candidate_safe_id: str,
        reason: str,
    ) -> EnrichmentQueueItem:
        if reason not in QUEUE_REASONS:
            reason = "pending_ddi_check"
        qid = _new_queue_id()
        safe_qid = _safe_queue_id(qid)
        item = EnrichmentQueueItem(
            queue_id=qid,
            safe_queue_id=safe_qid,
            candidate_safe_id=candidate_safe_id,
            reason=reason,
            status=QUEUE_STATUS_PENDING,
            created_at=_now_utc(),
            safe_public_summary={
                "safe_queue_id": safe_qid,
                "candidate_safe_id": candidate_safe_id,
                "reason": reason,
                "status": QUEUE_STATUS_PENDING,
            },
        )
        self._items.append(item)
        return item

    def list_items(self) -> List[EnrichmentQueueItem]:
        return list(self._items)

    def list_pending(self) -> List[EnrichmentQueueItem]:
        return [i for i in self._items if i.status == QUEUE_STATUS_PENDING]

    def mark_status(self, queue_id: str, status: str) -> bool:
        for item in self._items:
            if item.queue_id == queue_id:
                item.status = status
                item.safe_public_summary["status"] = status
                return True
        return False

    def count(self) -> int:
        return len(self._items)

    def count_by_reason(self, reason: str) -> int:
        return sum(1 for i in self._items if i.reason == reason)
