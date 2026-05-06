"""Contradiction detection for CKA-B08.

detect_consensus_contradictions(consensus_facts) -> list[ConsensusContradiction]

Rules:
- Same (specialty, fact_type, entity_text) with conflicting structured values → contradiction.
- Medication dose conflict → routed to Truth Resolution, quarantine-only.
- Contradictions must NOT be merged or synthesized over.
- Contradiction output uses safe IDs only.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from clinical_knowledge.consensus.models import (
    ConsensusFact,
    ConsensusContradiction,
    ConsensusFactStatus,
    _safe_fact_id,
)

_MEDICATION_FACT_TYPES = frozenset({
    "medication",
    "medication_dose",
    "ddi_check",
    "prescription",
    "drug",
})


def _is_medication_dose_conflict(fact_type: str, structured_values: List[Dict[str, Any]]) -> bool:
    """Heuristic: medication-related fact_type with differing dose/severity."""
    if fact_type.lower() not in _MEDICATION_FACT_TYPES:
        return False
    # Check for dose or severity fields that differ
    dose_keys = {"dose", "dosage", "severity", "amount", "strength"}
    for key in dose_keys:
        values_for_key = [sv.get(key) for sv in structured_values if key in sv]
        if len(set(str(v) for v in values_for_key)) > 1:
            return True
    return True  # Any medication fact conflict is treated as dose conflict


def detect_consensus_contradictions(
    consensus_facts: List[ConsensusFact],
) -> List[ConsensusContradiction]:
    """Detect contradictions in a list of ConsensusFact.

    A contradiction exists when a fact has status=CONTRADICTED (set by fact_extractor
    when multiple connectors returned the same (specialty, fact_type, entity_text)
    with different structured values).

    Returns a list of ConsensusContradiction objects.
    """
    contradictions: List[ConsensusContradiction] = []

    for fact in consensus_facts:
        if fact.status != ConsensusFactStatus.CONTRADICTED:
            continue

        med_conflict = _is_medication_dose_conflict(fact.fact_type, [fact.structured])

        contradiction = ConsensusContradiction(
            fact_type=fact.fact_type,
            entity_text=fact.entity_text,
            specialty=fact.specialty,
            conflicting_structured_values=[fact.structured],  # merged in fact_extractor
            connector_names=fact.supporting_connectors,
            safe_ids=[fact.safe_fact_id],
            is_medication_dose_conflict=med_conflict,
        )
        contradictions.append(contradiction)

    return contradictions
