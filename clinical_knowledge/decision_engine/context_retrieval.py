"""MKB context retrieval for CKA-B03 Decision Engine."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from clinical_knowledge.decision_engine.models import DecisionContext, QueryClassification

if TYPE_CHECKING:
    from clinical_knowledge.store import MKBStore


_MAX_SNIPPETS = 5


def retrieve_context(classification: QueryClassification, store: "MKBStore") -> DecisionContext:
    """Retrieve relevant MKB records for the given classification.

    Returns safe snippets only — no raw PHI, no private fields.
    """
    snippets: List[str] = []
    tiers: List[str] = []
    total_found = 0

    specialty = classification.specialty.value

    # Pull active records first, then hypothesis
    for record in store.list_active():
        if _record_matches(record, specialty, classification):
            snippets.append(_safe_snippet(record))
            tiers.append("active")
            total_found += 1
            if total_found >= _MAX_SNIPPETS:
                break

    if total_found < _MAX_SNIPPETS:
        for record in store.list_hypothesis():
            if _record_matches(record, specialty, classification):
                snippets.append(_safe_snippet(record))
                tiers.append("hypothesis")
                total_found += 1
                if total_found >= _MAX_SNIPPETS:
                    break

    return DecisionContext(
        query_hash=classification.raw_query_hash,
        mkb_records_found=total_found,
        mkb_snippets=snippets,
        context_tiers=tiers,
    )


def _record_matches(record: dict, specialty: str, classification: QueryClassification) -> bool:
    rec_specialty = record.get("specialty", "general") or "general"
    if specialty != "unknown" and rec_specialty not in (specialty, "general"):
        return False
    fact_type = record.get("fact_type", "") or ""
    entity = record.get("entity_text", "") or ""
    task = classification.task_type.value
    if task == "medication":
        for term in classification.medication_terms_detected:
            if term in entity.lower() or term in fact_type.lower():
                return True
        return "drug" in fact_type.lower() or "medication" in fact_type.lower()
    return True


def _safe_snippet(record: dict) -> str:
    safe_id = record.get("safe_record_id", "") or ""
    fact_type = record.get("fact_type", "") or ""
    tier = record.get("tier", "") or ""
    return f"[{safe_id}] {fact_type} (tier={tier})"
