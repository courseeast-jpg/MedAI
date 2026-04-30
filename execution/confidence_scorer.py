"""Computed extraction confidence scoring."""

from __future__ import annotations

from typing import Any


EXTRACTOR_RELIABILITY_WEIGHTS = {
    "spacy": 0.8,
    "phi3": 0.6,
    "gemini": 0.9,
}


def score_extraction_result(result: dict[str, Any], *, overwrite_confidence: bool = True) -> dict[str, Any]:
    """Return a copy of an extraction result with computed confidence fields."""

    scored = dict(result)
    entities = [entity for entity in scored.get("entities", []) if isinstance(entity, dict)]
    raw_text = str(scored.get("raw_text") or "")
    extractor = str(scored.get("actual_extractor") or scored.get("extractor") or "unknown").lower()

    breakdown = {
        "entity_count": entity_count_score(len(entities)),
        "coverage": text_coverage_score(entities, raw_text),
        "diversity": entity_diversity_score(entities),
        "extractor_weight": extractor_reliability_weight(extractor),
    }
    confidence = (
        breakdown["entity_count"] * 0.35
        + breakdown["coverage"] * 0.25
        + breakdown["diversity"] * 0.25
        + breakdown["extractor_weight"] * 0.15
    )
    computed_confidence = round(max(0.0, min(confidence, 1.0)), 3)
    if overwrite_confidence:
        scored["confidence"] = computed_confidence
    else:
        scored.setdefault("confidence", computed_confidence)
    scored["confidence_breakdown"] = {key: round(float(value), 3) for key, value in breakdown.items()}
    return scored


def entity_count_score(entity_count: int) -> float:
    if entity_count <= 0:
        return 0.0
    if entity_count <= 2:
        return 0.3
    if entity_count <= 5:
        return 0.6
    return 0.8


def text_coverage_score(entities: list[dict[str, Any]], raw_text: str) -> float:
    text = str(raw_text or "")
    text_length = len("".join(char for char in text if char.isalnum()))
    if not entities or text_length <= 0:
        return 0.0

    covered_terms = {
        str(entity.get("text", "")).strip().lower()
        for entity in entities
        if str(entity.get("text", "")).strip()
    }
    covered_length = sum(len("".join(char for char in term if char.isalnum())) for term in covered_terms)
    coverage_ratio = covered_length / max(text_length, 1)
    if coverage_ratio == 0 and entities:
        return 0.45
    if coverage_ratio >= 0.15:
        return 1.0
    if coverage_ratio >= 0.08:
        return 0.75
    if coverage_ratio >= 0.03:
        return 0.45
    if coverage_ratio > 0:
        return 0.2
    return 0.0


def entity_diversity_score(entities: list[dict[str, Any]]) -> float:
    if not entities:
        return 0.0
    unique_keys = {
        (
            str(entity.get("type", "")).strip().lower(),
            str(entity.get("text", "")).strip().lower(),
        )
        for entity in entities
        if str(entity.get("text", "")).strip()
    }
    unique_count = len(unique_keys)
    if unique_count <= 0:
        return 0.0
    if unique_count == 1 and len(entities) > 1:
        return 0.3
    if unique_count == 1:
        return 0.6
    if unique_count == 2:
        return 0.9
    return 1.0


def extractor_reliability_weight(extractor: str) -> float:
    return EXTRACTOR_RELIABILITY_WEIGHTS.get(str(extractor or "").lower(), 0.5)
