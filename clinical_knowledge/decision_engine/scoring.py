"""Response scoring for CKA-B03 Decision Engine.

Weights: mkb_consistency=0.35, internal_coherence=0.25,
         citation_presence=0.20, ddi_safety_score=0.20
Discard threshold: composite < 0.30
Bands: discarded(<0.30) / low(0.30-0.49) / acceptable(0.50-0.74) / high(>=0.75)
"""
from __future__ import annotations

from typing import List

from clinical_knowledge.decision_engine.models import (
    ConnectorResponse,
    DecisionContext,
    QueryClassification,
    ScoreBand,
    ScoredResponse,
)

_DISCARD_THRESHOLD = 0.30

SCORE_WEIGHTS = {
    "mkb_consistency": 0.35,
    "internal_coherence": 0.25,
    "citation_presence": 0.20,
    "ddi_safety_score": 0.20,
}


def score_response(
    response: ConnectorResponse,
    context: DecisionContext,
    classification: QueryClassification,
    ddi_layer1_modifier: float = 1.0,
) -> ScoredResponse:
    """Score a connector response against MKB context."""
    if not response.success:
        composite = 0.0
        return ScoredResponse(
            connector_id=response.connector_id,
            raw_content=response.content,
            mkb_consistency_score=0.0,
            internal_coherence_score=0.0,
            citation_presence_score=0.0,
            ddi_safety_score=0.0,
            composite_score=0.0,
            score_band=ScoreBand.DISCARDED,
            discarded=True,
        )

    mkb_score = _score_mkb_consistency(response, context)
    coherence_score = _score_internal_coherence(response)
    citation_score = _score_citation_presence(response)
    ddi_score = _score_ddi_safety(response, classification, ddi_layer1_modifier)

    composite = ScoredResponse.compute_composite(
        mkb_score, coherence_score, citation_score, ddi_score
    )
    band = _band(composite)
    discarded = composite < _DISCARD_THRESHOLD

    return ScoredResponse(
        connector_id=response.connector_id,
        raw_content=response.content,
        mkb_consistency_score=mkb_score,
        internal_coherence_score=coherence_score,
        citation_presence_score=citation_score,
        ddi_safety_score=ddi_score,
        composite_score=composite,
        score_band=band,
        discarded=discarded,
    )


def _score_mkb_consistency(response: ConnectorResponse, context: DecisionContext) -> float:
    """Higher score when active-tier MKB records are available and response confidence is good."""
    active_count = context.context_tiers.count("active")
    base = min(0.9, 0.5 + active_count * 0.1)
    return round(base * response.confidence, 4)


def _score_internal_coherence(response: ConnectorResponse) -> float:
    content = response.content or ""
    if not content or len(content) < 20:
        return 0.20
    # Coherence heuristic: longer structured content scores higher
    words = len(content.split())
    score = min(0.90, 0.45 + words * 0.003)
    return round(score, 4)


def _score_citation_presence(response: ConnectorResponse) -> float:
    n = len(response.citations)
    if n == 0:
        return 0.10
    if n == 1:
        return 0.60
    return min(0.95, 0.60 + (n - 1) * 0.15)


def _score_ddi_safety(
    response: ConnectorResponse,
    classification: QueryClassification,
    modifier: float,
) -> float:
    base = 0.80 if not classification.requires_ddi_check else 0.65
    return round(min(1.0, base * modifier), 4)


def _band(score: float) -> ScoreBand:
    if score < 0.30:
        return ScoreBand.DISCARDED
    if score < 0.50:
        return ScoreBand.LOW
    if score < 0.75:
        return ScoreBand.ACCEPTABLE
    return ScoreBand.HIGH


def score_all_responses(
    responses: List[ConnectorResponse],
    context: DecisionContext,
    classification: QueryClassification,
    ddi_layer1_modifier: float = 1.0,
) -> List[ScoredResponse]:
    return [
        score_response(r, context, classification, ddi_layer1_modifier)
        for r in responses
    ]
