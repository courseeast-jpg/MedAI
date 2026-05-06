"""Consensus engine orchestrator for CKA-B08.

run_consensus(connector_results, *, min_confidence, safe_mode) -> ConsensusResult

Integrates:
- Response filtering (discard malformed/blocked/failed)
- Agreement scoring
- Contradiction detection
- Truth Resolution handoff for contradictions

Rules:
- Discarded responses do NOT enter consensus.
- Contradictions are NOT synthesized over.
- No auto-write of active facts.
- Truth Resolution for contradictions: quarantine-only, no DDI invocation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from clinical_knowledge.connectors.models import ConnectorExecutionResult, ConnectorStatus
from clinical_knowledge.consensus.agreement import calculate_agreement
from clinical_knowledge.consensus.contradiction import detect_consensus_contradictions
from clinical_knowledge.consensus.models import (
    ConsensusResult,
    ConsensusStatus,
)

if TYPE_CHECKING:
    from clinical_knowledge.store import MKBStore

# Responses with these statuses are DISCARDED before consensus
_DISCARD_STATUSES = frozenset({
    ConnectorStatus.TIMEOUT,
    ConnectorStatus.ERROR,
    ConnectorStatus.MALFORMED_RESPONSE,
    ConnectorStatus.BLOCKED_PRIVACY,
    ConnectorStatus.SKIPPED_SAFE_MODE,
    ConnectorStatus.DISABLED,
})

_DEFAULT_MIN_CONFIDENCE = 0.60


def run_consensus(
    connector_results: List[ConnectorExecutionResult],
    *,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    safe_mode: bool = False,
    store: "Optional[MKBStore]" = None,
) -> ConsensusResult:
    """Full consensus pipeline.

    1. Filter: discard failed/malformed/blocked results.
    2. Score agreement across valid normalized responses.
    3. Detect contradictions.
    4. Route contradictions to Truth Resolution (quarantine-only).
    5. Return ConsensusResult.
    """
    # Step 1 — Filter
    successful = [
        r for r in connector_results
        if r.status == ConnectorStatus.SUCCESS
        and r.normalized_response is not None
    ]
    discarded_count = len(connector_results) - len(successful)

    # Step 2 — Filter by min_confidence threshold
    passing = []
    low_conf_discarded = 0
    for r in successful:
        resp_confidence = r.normalized_response.get("confidence", 0.0) if r.normalized_response else 0.0
        if resp_confidence >= min_confidence:
            passing.append(r.normalized_response)
        else:
            low_conf_discarded += 1

    discarded_count += low_conf_discarded
    normalized_responses = [r for r in passing if r is not None]

    # Step 3 — Agreement scoring
    result = calculate_agreement(normalized_responses, discarded_count)

    # Step 4 — Contradiction detection
    contradictions = detect_consensus_contradictions(result.consensus_facts)
    result.contradictions = contradictions

    if contradictions:
        if result.status == ConsensusStatus.CONSENSUS_READY:
            result.status = ConsensusStatus.CONTRADICTION_DETECTED

    # Step 5 — Truth Resolution handoff for contradictions
    tr_results = []
    if contradictions and store is not None:
        from clinical_knowledge.consensus.integration import route_consensus_contradictions_to_truth_resolution
        tr_results = route_consensus_contradictions_to_truth_resolution(contradictions, store)
        result.truth_resolution_results = tr_results
        if tr_results:
            result.status = ConsensusStatus.TRUTH_RESOLUTION_REQUIRED

    return result
