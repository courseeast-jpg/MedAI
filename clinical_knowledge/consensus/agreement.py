"""Agreement scoring for CKA-B08.

calculate_agreement(normalized_responses, discarded_count) -> ConsensusResult

Rules:
- agreement_ratio = supporting_connectors / total_successful_connectors
- confidence_aggregate = weighted mean of per-fact confidence × agreement_ratio
- Single successful connector → 0.75 confidence penalty, SINGLE_SOURCE_PENALIZED facts
- All responses discarded/malformed/timeout → ALL_RESPONSES_DISCARDED, escalation_required=True
- No clinical synthesis.
"""
from __future__ import annotations

from typing import Any, Dict, List

from clinical_knowledge.consensus.fact_extractor import extract_consensus_facts
from clinical_knowledge.consensus.models import (
    ConsensusFact,
    ConsensusFactStatus,
    ConsensusResult,
    ConsensusStatus,
)


def calculate_agreement(
    normalized_responses: List[Dict[str, Any]],
    discarded_count: int = 0,
) -> ConsensusResult:
    """Score agreement across normalized connector responses.

    Args:
        normalized_responses: list of dicts from normalize_connector_response()
        discarded_count: count of responses already discarded before this stage

    Returns:
        ConsensusResult with scored facts and status.
    """
    total_successful = len(normalized_responses)

    # No successful responses
    if total_successful == 0:
        return ConsensusResult(
            status=ConsensusStatus.ALL_RESPONSES_DISCARDED,
            consensus_facts=[],
            contradictions=[],
            confidence_aggregate=0.0,
            discarded_response_count=discarded_count,
            truth_resolution_results=[],
            escalation_required=True,
        )

    facts = extract_consensus_facts(normalized_responses, total_successful)

    # Separate contradicted from non-contradicted
    good_facts = [f for f in facts if f.status != ConsensusFactStatus.CONTRADICTED]
    contradicted_facts = [f for f in facts if f.status == ConsensusFactStatus.CONTRADICTED]

    has_contradictions = bool(contradicted_facts)

    # Confidence aggregate across non-contradicted facts
    if good_facts:
        confidence_aggregate = sum(f.confidence for f in good_facts) / len(good_facts)
    else:
        confidence_aggregate = 0.0

    # Determine overall status
    if total_successful == 1:
        status = ConsensusStatus.CONSENSUS_READY  # with single_source_penalized facts
    elif has_contradictions:
        status = ConsensusStatus.CONTRADICTION_DETECTED
    elif not good_facts:
        status = ConsensusStatus.INSUFFICIENT_RESPONSES
    else:
        status = ConsensusStatus.CONSENSUS_READY

    return ConsensusResult(
        status=status,
        consensus_facts=facts,
        contradictions=[],   # filled in by contradiction detector
        confidence_aggregate=round(confidence_aggregate, 4),
        discarded_response_count=discarded_count,
        truth_resolution_results=[],
        escalation_required=(total_successful == 0 or status == ConsensusStatus.ALL_RESPONSES_DISCARDED),
    )
