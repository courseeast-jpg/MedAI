"""Fact extraction from normalized connector responses for CKA-B08.

extract_consensus_facts(normalized_responses) -> list[ConsensusFact]

Rules:
- Use structured facts only — no free-text clinical inference.
- Group by (specialty, fact_type, entity_text).
- Same fact from 2+ connectors → candidate for agreement scoring.
- Single-source fact: retained with confidence penalty and SINGLE_SOURCE_PENALIZED.
- Does NOT write to MKB directly.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from clinical_knowledge.consensus.models import (
    ConsensusFact,
    ConsensusFactStatus,
    _safe_fact_id,
)

_SINGLE_SOURCE_PENALTY = 0.75


def _fact_key(specialty: str, fact_type: str, entity_text: str) -> str:
    """Deterministic grouping key for a fact."""
    return f"{specialty.lower()}:{fact_type.lower()}:{entity_text.lower().strip()}"


def _merge_structured(values: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge structured dicts for agreed facts (first wins for conflicts)."""
    merged: Dict[str, Any] = {}
    for v in values:
        for k, val in v.items():
            if k not in merged:
                merged[k] = val
    return merged


def extract_consensus_facts(
    normalized_responses: List[Dict[str, Any]],
    total_successful_connectors: int,
) -> List[ConsensusFact]:
    """Extract and group facts from normalized connector responses.

    Args:
        normalized_responses: list of dicts from normalize_connector_response()
        total_successful_connectors: count of connectors that returned SUCCESS
            (used for agreement_ratio calculation)

    Returns:
        List of ConsensusFact, one per unique (specialty, fact_type, entity_text).
    """
    # Group: key -> list of (connector_name, structured, confidence)
    groups: Dict[str, List[Tuple[str, Dict[str, Any], float]]] = defaultdict(list)

    for resp in normalized_responses:
        if not isinstance(resp, dict):
            continue
        connector_name = resp.get("connector_name", "unknown")
        facts = resp.get("facts", [])
        if not isinstance(facts, list):
            continue
        resp_confidence = float(resp.get("confidence", 0.0))

        for fact in facts:
            if not isinstance(fact, dict):
                continue
            fact_type = fact.get("fact_type", "")
            entity_text = fact.get("entity_text", "")
            specialty = fact.get("specialty", "general")
            structured = fact.get("structured", {})
            fact_confidence = float(fact.get("confidence", resp_confidence))

            if not fact_type or not entity_text:
                continue

            key = _fact_key(specialty, fact_type, entity_text)
            groups[key].append((connector_name, structured, fact_confidence))

    consensus_facts: List[ConsensusFact] = []
    denom = max(total_successful_connectors, 1)

    for key, entries in groups.items():
        supporting = [e[0] for e in entries]
        structured_values = [e[1] for e in entries]
        confidences = [e[2] for e in entries]

        specialty, fact_type, entity_text = key.split(":", 2)
        agreement_ratio = len(supporting) / denom

        # Detect structural contradiction within this fact group
        # (different structured values from different connectors)
        unique_structured = []
        for sv in structured_values:
            if sv not in unique_structured:
                unique_structured.append(sv)

        if len(unique_structured) > 1:
            # Contradiction — handled by contradiction detector; mark as contradicted
            status = ConsensusFactStatus.CONTRADICTED
            merged = {}
            base_confidence = min(confidences) if confidences else 0.0
        elif len(supporting) == 1:
            # Single source — apply penalty
            status = ConsensusFactStatus.SINGLE_SOURCE_PENALIZED
            merged = structured_values[0] if structured_values else {}
            base_confidence = confidences[0] * _SINGLE_SOURCE_PENALTY if confidences else 0.0
        else:
            status = ConsensusFactStatus.AGREED
            merged = _merge_structured(structured_values)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            base_confidence = avg_confidence

        # Weight by agreement_ratio
        confidence = base_confidence * agreement_ratio if status == ConsensusFactStatus.AGREED else base_confidence

        consensus_facts.append(ConsensusFact(
            fact_id=key,
            safe_fact_id=_safe_fact_id(key),
            fact_type=fact_type,
            entity_text=entity_text,
            structured=merged,
            specialty=specialty,
            supporting_connectors=supporting,
            contradicting_connectors=[],   # filled in by contradiction detector
            agreement_ratio=agreement_ratio,
            confidence=round(min(1.0, max(0.0, confidence)), 4),
            status=status,
        ))

    return consensus_facts
