"""
MedAI v1.1 - Quality Gate (Track A)
4 sequential checks before any MKB write.
Calls Truth Resolution on conflict.
"""
from datetime import datetime
from typing import Optional, Tuple
from loguru import logger

from app.config import MIN_CONFIDENCE_CLAUDE, MIN_CONFIDENCE_RULES
from app.schemas import MKBRecord, LedgerEvent
from mkb.sqlite_store import SQLiteStore
from mkb.vector_store import VectorStore
from mkb.truth_resolution import TruthResolutionEngine, TruthResolutionInput


class QualityGate:
    def __init__(self, sql: SQLiteStore, vec: VectorStore):
        self.sql = sql
        self.vec = vec
        self.resolver = TruthResolutionEngine()

    def check(
        self, candidate: MKBRecord, session_id: str = ""
    ) -> Tuple[bool, str, Optional[MKBRecord]]:
        """
        Returns (approved: bool, reason: str, final_record: Optional[MKBRecord])
        final_record is the record to write (may differ from candidate after resolution)
        """

        # Check 4 - Minimum quality (run first, cheap)
        min_conf = MIN_CONFIDENCE_RULES if candidate.extraction_method == "rules_based" else MIN_CONFIDENCE_CLAUDE
        if candidate.confidence < min_conf:
            self._log_ledger(
                "quality_reject",
                candidate,
                session_id,
                {"reason": f"confidence {candidate.confidence:.2f} < threshold {min_conf}"},
            )
            return False, f"Confidence {candidate.confidence:.2f} below threshold", None

        if not candidate.content or len(candidate.content.strip()) < 5:
            self._log_ledger("quality_reject", candidate, session_id, {"reason": "empty content"})
            return False, "Content too short", None

        if candidate.fact_type not in [
            "diagnosis", "medication", "test_result", "symptom",
            "note", "recommendation", "relationship", "event",
        ]:
            return False, f"Invalid fact_type: {candidate.fact_type}", None

        # Check 3 - Freshness (mark stale, don't reject)
        if candidate.first_recorded:
            age_days = (datetime.utcnow() - candidate.first_recorded).days
            if age_days > 365 * 5:
                candidate = candidate.model_copy(update={
                    "tags": candidate.tags + ["possibly_outdated"],
                    "status": "active",
                })

        existing_records = self.sql.get_by_specialty(candidate.specialty)
        for existing in existing_records:
            if self._is_exact_duplicate(candidate, existing):
                self.sql.update_status(existing.id, existing.status)
                self._log_ledger(
                    "dedup_confirm",
                    candidate,
                    session_id,
                    {"existing_id": existing.id, "action": "exact_duplicate_confirmed"},
                )
                return False, f"Duplicate of {existing.id} - confirmation timestamp updated", None

        # Check 1 - Deduplication
        dup_id = self.vec.check_duplicate(candidate.content)
        if dup_id and dup_id != candidate.id:
            existing = self.sql.get_record(dup_id)
            if existing:
                # Same-level: update confirmation timestamp
                if existing.trust_level == candidate.trust_level:
                    self.sql.update_status(dup_id, existing.status)
                    self._log_ledger(
                        "dedup_confirm",
                        candidate,
                        session_id,
                        {"existing_id": dup_id, "action": "timestamp_updated"},
                    )
                    return False, f"Duplicate of {dup_id} - confirmation timestamp updated", None
                # Different trust: run resolution
                return self._resolve_conflict(candidate, existing, "value_conflict", session_id)

        # Check 2 - Conflict detection (same fact_type + specialty, different content)
        for existing in existing_records:
            if existing.content.strip().lower() == candidate.content.strip().lower():
                continue
            if (
                existing.fact_type == candidate.fact_type
                and existing.status == "active"
                and existing.id != candidate.id
                and self._is_same_entity(candidate, existing)
            ):
                if self._has_pending_review_for_entity(candidate):
                    self._log_ledger(
                        "dedup_pending_review",
                        candidate,
                        session_id,
                        {"existing_id": existing.id, "action": "pending_review_already_exists"},
                    )
                    return False, "Existing record retained: pending review already exists", None
                return self._resolve_conflict(candidate, existing, "value_conflict", session_id)

        # All checks passed
        return True, "approved", candidate

    def _resolve_conflict(
        self, candidate: MKBRecord, existing: MKBRecord,
        conflict_type: str, session_id: str
    ) -> Tuple[bool, str, Optional[MKBRecord]]:
        inp = TruthResolutionInput(
            candidate_fact=candidate,
            existing_fact=existing,
            conflict_type=conflict_type,
        )
        result = self.resolver.resolve(inp)
        ledger_event = self.resolver.build_ledger_event(result, session_id)
        self.sql.write_ledger(ledger_event)

        if result.resolution == "quarantine":
            quarantined = candidate.model_copy(update={
                "tier": "quarantined",
                "status": "quarantined",
                "requires_review": True,
                "resolution_id": str(self.sql.write_ledger(ledger_event)),
            })
            logger.warning(f"Quarantined: {candidate.id} - {result.explanation}")
            return True, "quarantined", quarantined

        if result.resolution in ("replace_with_new", "merge"):
            self.sql.update_status(result.loser_id, "superseded", "superseded")
            self.vec.delete_record(result.loser_id)
            logger.info(f"Resolved: {result.rule_applied} - {result.explanation}")
            return True, result.explanation, result.winner

        logger.info(f"Kept existing: {result.rule_applied} - {result.explanation}")
        return False, f"Existing record retained: {result.explanation}", None

    def _is_same_entity(self, a: MKBRecord, b: MKBRecord) -> bool:
        """Heuristic: same medication name or same diagnosis name."""
        a_name = a.structured.get("name", "").lower()
        b_name = b.structured.get("name", "").lower()
        if a_name and b_name and a_name == b_name:
            return True
        # Fallback: high word overlap
        a_words = set(a.content.lower().split())
        b_words = set(b.content.lower().split())
        if len(a_words) > 2 and len(b_words) > 2:
            overlap = len(a_words & b_words) / max(len(a_words), len(b_words))
            return overlap > 0.7
        return False

    def _is_exact_duplicate(self, a: MKBRecord, b: MKBRecord) -> bool:
        return (
            a.id != b.id
            and a.fact_type == b.fact_type
            and a.specialty == b.specialty
            and b.status == "active"
            and self._norm(a.content) == self._norm(b.content)
        )

    def _has_pending_review_for_entity(self, candidate: MKBRecord) -> bool:
        for review in self.sql.get_records_requiring_review():
            if (
                review.fact_type == candidate.fact_type
                and review.specialty == candidate.specialty
                and self._is_same_entity(candidate, review)
            ):
                return True
        return False

    def _norm(self, value: str) -> str:
        return " ".join((value or "").casefold().split())

    def _log_ledger(self, event_type: str, record: MKBRecord, session_id: str, details: dict):
        self.sql.write_ledger(LedgerEvent(
            event_type=event_type,
            record_id=record.id,
            source_type=record.source_type,
            details=details,
            session_id=session_id,
        ))
