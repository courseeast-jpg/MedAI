"""CKA-B04 Truth Resolution Engine orchestrator."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from clinical_knowledge.ledger import make_quarantine_event, make_truth_resolution_event
from clinical_knowledge.models import KnowledgeTier, RecordStatus
from clinical_knowledge.truth_resolution.conflict_detection import detect_conflict
from clinical_knowledge.truth_resolution.models import (
    ConflictPair,
    ResolutionAction,
    TruthResolutionResult,
)
from clinical_knowledge.truth_resolution.quarantine import apply_quarantine, apply_supersede
from clinical_knowledge.truth_resolution.rules import ORDERED_RULES, rule_unresolvable

if TYPE_CHECKING:
    from clinical_knowledge.store import MKBStore


def resolve_conflict(pair: ConflictPair) -> TruthResolutionResult:
    """Apply the 7 ordered rules. First match wins. Rule 7 always fires."""
    for rule_fn in ORDERED_RULES:
        result = rule_fn(pair)
        if result is not None:
            return result
    return rule_unresolvable(pair)


def apply_truth_resolution(
    candidate,
    existing,
    store: "MKBStore",
) -> Optional[TruthResolutionResult]:
    """Full pipeline: detect conflict → resolve → update store → write ledger.

    Returns None if no conflict is detected (no action taken).
    """
    pair = detect_conflict(candidate, existing)
    if pair is None:
        return None

    result = resolve_conflict(pair)
    _apply_store_updates(result, candidate, store)
    _write_ledger_events(result, pair, store)
    return result


def _apply_store_updates(
    result: TruthResolutionResult,
    candidate,
    store: "MKBStore",
) -> None:
    action = result.resolution

    if action == ResolutionAction.QUARANTINE:
        apply_quarantine(result, store)
        return

    # Supersede losers for any winning rule
    apply_supersede(result, store)

    now = datetime.now(timezone.utc).isoformat()
    if action == ResolutionAction.REPLACE_WITH_NEW:
        # Insert candidate as confirmed active if it isn't already in the store
        try:
            existing_row = store.fetch_by_record_id(candidate.record_id)
        except Exception:
            existing_row = None
        if existing_row is None:
            store.insert_record(candidate)
        else:
            store.update_record_tier(candidate.record_id, KnowledgeTier.ACTIVE, now)
            store.update_record_status(candidate.record_id, RecordStatus.CONFIRMED, now)

    elif action == ResolutionAction.MERGE:
        merged = result.merged_record
        if merged is not None:
            try:
                existing_row = store.fetch_by_record_id(merged.record_id)
            except Exception:
                existing_row = None
            if existing_row is None:
                store.insert_record(merged)


def _write_ledger_events(
    result: TruthResolutionResult,
    pair: ConflictPair,
    store: "MKBStore",
) -> None:
    candidate = pair.candidate_fact
    safe_id = getattr(candidate, "safe_record_id", "") or ""
    record_id = getattr(candidate, "record_id", "") or ""

    if result.resolution == ResolutionAction.QUARANTINE:
        evt = make_quarantine_event(
            record_id=record_id,
            safe_record_id=safe_id,
            quarantined_safe_ids=[
                getattr(pair.candidate_fact, "safe_record_id", ""),
                getattr(pair.existing_fact, "safe_record_id", ""),
            ],
            conflict_type=pair.conflict_type.value,
            explanation=result.explanation,
        )
        store.append_ledger_event(evt)
    else:
        winner_safe = getattr(result.winner, "safe_record_id", "") if result.winner else ""
        loser_safe = ""
        if result.loser_id:
            # Determine loser safe_id from pair
            for rec in [pair.candidate_fact, pair.existing_fact]:
                if getattr(rec, "record_id", "") == result.loser_id:
                    loser_safe = getattr(rec, "safe_record_id", "")
                    break

        evt = make_truth_resolution_event(
            record_id=record_id,
            safe_record_id=safe_id,
            rule_applied=result.rule_applied.value,
            resolution=result.resolution.value,
            winner_safe_id=winner_safe,
            loser_safe_id=loser_safe,
            confidence=result.confidence,
            requires_review=result.requires_review,
            explanation=result.explanation,
        )
        store.append_ledger_event(evt)
