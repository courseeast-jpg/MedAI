"""Integration helpers for CKA-B08 Consensus Engine.

route_consensus_contradictions_to_truth_resolution(contradictions, store) -> list[dict]
consensus_facts_to_enrichment_candidates(consensus_facts, store) -> list

Rules:
- Medication dose contradictions: Truth Resolution quarantine-only (no DDI check invoked).
- Truth Resolution in B08 does NOT write active records from consensus.
- Consensus → enrichment path: results remain hypothesis-only (CKA-B06 rules apply).
- No auto-write of active facts from connector consensus.
- Safe IDs only in return values.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from clinical_knowledge.consensus.models import ConsensusContradiction, ConsensusFact

if TYPE_CHECKING:
    from clinical_knowledge.store import MKBStore

_SALT = "medai_cka_b08_integration_v1"


def _safe_id(value: str) -> str:
    digest = hashlib.sha256(f"{_SALT}:{value}".encode()).hexdigest()[:16]
    return f"cka_b08_{digest}"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def route_consensus_contradictions_to_truth_resolution(
    contradictions: List[ConsensusContradiction],
    store: "MKBStore",
) -> List[Dict[str, Any]]:
    """Hand contradictions off to CKA-B04 Truth Resolution.

    Behavior per spec:
    - Medication dose contradiction → quarantine-only via Truth Resolution.
    - DDI check is NOT invoked from here.
    - No active write from contradiction routing.
    - Returns list of safe-only summaries (no raw record_id, no PHI).
    """
    from clinical_knowledge.truth_resolution.models import (
        ConflictPair,
        ConflictType,
        ResolutionAction,
        ResolutionRule,
        TruthResolutionResult,
    )
    from clinical_knowledge.ledger import make_quarantine_event

    results: List[Dict[str, Any]] = []

    for contradiction in contradictions:
        conflict_type = (
            ConflictType.MEDICATION_DOSE_CONFLICT
            if contradiction.is_medication_dose_conflict
            else ConflictType.VALUE_CONFLICT
        )

        # For B08 report-only mode: produce a safe summary without actual store writes
        # (no active record is inserted; we only write a quarantine ledger event)
        safe_contradiction_id = _safe_id(
            f"contradiction:{contradiction.fact_type}:{contradiction.entity_text}"
        )

        resolution_summary: Dict[str, Any] = {
            "safe_contradiction_id": safe_contradiction_id,
            "fact_type": contradiction.fact_type,
            "entity_text": contradiction.entity_text,
            "specialty": contradiction.specialty,
            "conflict_type": conflict_type.value,
            "resolution_action": ResolutionAction.QUARANTINE.value,
            "rule_applied": ResolutionRule.MEDICATION_DOSE_CONFLICT.value
            if contradiction.is_medication_dose_conflict
            else ResolutionRule.UNRESOLVABLE.value,
            "is_medication_dose_conflict": contradiction.is_medication_dose_conflict,
            "ddi_invoked": False,   # explicitly never True in B08
            "active_write": False,
            "quarantine_only": True,
            "timestamp": _now_utc(),
        }

        # Write a safe quarantine ledger event
        try:
            evt = make_quarantine_event(
                record_id="",   # no real record; synthetic contradiction
                safe_record_id=safe_contradiction_id,
                quarantined_safe_ids=contradiction.safe_ids,
                conflict_type=conflict_type.value,
                explanation=(
                    f"B08 consensus contradiction quarantine: "
                    f"{contradiction.fact_type}/{contradiction.entity_text}"
                ),
            )
            store.append_ledger_event(evt)
            resolution_summary["ledger_event_written"] = True
        except Exception as exc:
            resolution_summary["ledger_event_written"] = False
            resolution_summary["ledger_event_error"] = str(exc)

        results.append(resolution_summary)

    return results


def consensus_facts_to_enrichment_candidates(
    consensus_facts: List[ConsensusFact],
    *,
    allow_active_write: bool = False,
) -> List[Dict[str, Any]]:
    """Convert ConsensusFacts to enrichment candidate descriptors.

    These are NOT written to the MKB directly — they are hypothesis-only
    descriptors that must go through CKA-B06 Controlled Enrichment before
    any store write.

    allow_active_write must remain False in B08 (raises if set True).
    """
    if allow_active_write:
        raise ValueError(
            "allow_active_write=True is not permitted in B08. "
            "Consensus facts must remain hypothesis-only until enrichment review."
        )

    candidates = []
    for fact in consensus_facts:
        candidates.append({
            "safe_fact_id": fact.safe_fact_id,
            "fact_type": fact.fact_type,
            "entity_text": fact.entity_text,
            "structured": fact.structured,
            "specialty": fact.specialty,
            "confidence": fact.confidence,
            "agreement_ratio": fact.agreement_ratio,
            "status": fact.status.value,
            "tier": "hypothesis",          # always hypothesis from consensus
            "active_write": False,
            "requires_enrichment_review": True,
        })

    return candidates
