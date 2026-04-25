from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
RELATIVE_TEMPORAL_RE = re.compile(
    r"\b(today|yesterday|tomorrow|currently|current|previously|last week|last month|last year)\b",
    re.IGNORECASE,
)
SINCE_RE = re.compile(r"\bsince\s+([A-Za-z0-9,\- ]+)", re.IGNORECASE)
HISTORY_RE = re.compile(r"\bhistory of\b", re.IGNORECASE)
NEGATION_RE = re.compile(
    r"\b(no|not|denies|denied|without|negative for|free of|rule out|ruled out)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EnrichedEntity:
    entity_index: int
    entity_type: str
    entity_text: str
    negation_flag: bool
    temporal_info: dict[str, Any] | None
    relationships: list[dict[str, Any]]
    enrichment_confidence: float
    enrichment_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticEnrichmentResult:
    applied: bool
    enrichment_source: str
    entities: list[EnrichedEntity]
    enriched_entity_count: int
    negation_detected_count: int
    temporal_detected_count: int
    relationships_detected_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["entities"] = [item.to_dict() for item in self.entities]
        return payload


def enrich_semantics(*, raw_text: str, entities: list[dict[str, Any]]) -> SemanticEnrichmentResult:
    normalized_text = str(raw_text or "")
    source = "rules_based"
    if not entities:
        return SemanticEnrichmentResult(
            applied=False,
            enrichment_source=source,
            entities=[],
            enriched_entity_count=0,
            negation_detected_count=0,
            temporal_detected_count=0,
            relationships_detected_count=0,
        )

    sentences = _split_sentences(normalized_text)
    enriched_entities: list[EnrichedEntity] = []

    for index, entity in enumerate(entities):
        entity_text = str(entity.get("text", "")).strip()
        entity_type = str(entity.get("type", "unknown"))
        context = _find_context(entity_text, normalized_text, sentences)
        negation_flag = _detect_negation(context)
        temporal_info = _detect_temporal(context)
        relationships = _detect_relationships(index, entity_text, entity_type, context, entities)
        signal_count = int(negation_flag) + int(temporal_info is not None) + int(bool(relationships))
        enrichment_confidence = round(min(0.75 + (signal_count * 0.1), 0.95), 3)

        enriched_entities.append(
            EnrichedEntity(
                entity_index=index,
                entity_type=entity_type,
                entity_text=entity_text,
                negation_flag=negation_flag,
                temporal_info=temporal_info,
                relationships=relationships,
                enrichment_confidence=enrichment_confidence,
                enrichment_source=source,
            )
        )

    return SemanticEnrichmentResult(
        applied=True,
        enrichment_source=source,
        entities=enriched_entities,
        enriched_entity_count=len(enriched_entities),
        negation_detected_count=sum(int(item.negation_flag) for item in enriched_entities),
        temporal_detected_count=sum(int(item.temporal_info is not None) for item in enriched_entities),
        relationships_detected_count=sum(len(item.relationships) for item in enriched_entities),
    )


def _split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [segment.strip() for segment in SENTENCE_SPLIT_RE.split(stripped) if segment.strip()]


def _find_context(entity_text: str, raw_text: str, sentences: list[str]) -> str:
    if not entity_text:
        return raw_text
    lowered = entity_text.lower()
    for sentence in sentences:
        if lowered in sentence.lower():
            return sentence
    return raw_text


def _detect_negation(context: str) -> bool:
    return bool(NEGATION_RE.search(context))


def _detect_temporal(context: str) -> dict[str, Any] | None:
    date_match = DATE_RE.search(context)
    if date_match:
        return {"kind": "date", "value": date_match.group(0)}
    relative_match = RELATIVE_TEMPORAL_RE.search(context)
    if relative_match:
        return {"kind": "relative", "value": relative_match.group(1).lower()}
    since_match = SINCE_RE.search(context)
    if since_match:
        return {"kind": "since", "value": since_match.group(1).strip()}
    if HISTORY_RE.search(context):
        return {"kind": "historical", "value": "history of"}
    return None


def _detect_relationships(
    entity_index: int,
    entity_text: str,
    entity_type: str,
    context: str,
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    lowered_context = context.lower()
    seen_targets: set[tuple[int, str]] = set()
    for other_index, other in enumerate(entities):
        if other_index == entity_index:
            continue
        other_text = str(other.get("text", "")).strip()
        other_type = str(other.get("type", "unknown"))
        if not other_text or other_text.lower() not in lowered_context:
            continue
        key = (other_index, other_text.lower())
        if key in seen_targets:
            continue
        seen_targets.add(key)
        relationships.append(
            {
                "type": "co_mentioned_with",
                "target_index": other_index,
                "target_type": other_type,
                "target_text": other_text,
                "source_type": entity_type,
                "source_text": entity_text,
            }
        )
    return relationships
