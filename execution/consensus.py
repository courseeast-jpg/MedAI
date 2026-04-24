"""Phase 2.1 consensus merge before validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def consensus_merge(results: list[dict[str, Any]], *, extractor_route: str) -> dict[str, Any]:
    """Merge one or more extractor outputs into a consensus payload."""

    if not results:
        raise ValueError("consensus_merge requires at least one extractor result")

    normalized_results = [deepcopy(result) for result in results]
    source_names = [str(result.get("extractor", "unknown")) for result in normalized_results]
    total_sources = len(normalized_results)
    primary_result = next(
        (result for result in normalized_results if str(result.get("extractor", "")) == extractor_route),
        normalized_results[0],
    )

    merged_entities_map: dict[tuple[str, str], dict[str, Any]] = {}
    raw_text = str(normalized_results[0].get("raw_text", ""))
    latency_ms = 0
    notes: list[str] = []
    confidence_total = 0.0

    for result in normalized_results:
        source_name = str(result.get("extractor", "unknown"))
        latency_ms += int(result.get("latency_ms", 0))
        confidence_total += float(result.get("confidence", 0.0))
        notes.extend(result.get("notes", []))

        for entity in result.get("entities", []):
            if not isinstance(entity, dict):
                continue
            key = _entity_key(entity)
            entry = merged_entities_map.setdefault(key, {
                "entity": None,
                "sources": [],
                "source_count": 0,
            })
            if source_name not in entry["sources"]:
                entry["sources"].append(source_name)
                entry["source_count"] += 1
            if entry["entity"] is None or _entity_weight(entity) > _entity_weight(entry["entity"]):
                entry["entity"] = deepcopy(entity)

    merged_entities: list[dict[str, Any]] = []
    supported_score_total = 0.0
    disagreement_flag = False

    for entry in merged_entities_map.values():
        entity = entry["entity"] or {}
        entity["consensus_support"] = list(entry["sources"])
        entity["consensus_support_count"] = int(entry["source_count"])
        merged_entities.append(entity)

        support_ratio = _support_ratio(int(entry["source_count"]), total_sources)
        supported_score_total += support_ratio
        if entry["source_count"] < total_sources:
            disagreement_flag = True

    if not merged_entities:
        agreement_score = 1.0 if total_sources == 1 else 0.0
    else:
        agreement_score = round(supported_score_total / len(merged_entities), 3)

    avg_confidence = round(confidence_total / total_sources, 3)
    consensus_confidence = round(avg_confidence * agreement_score, 3)
    if total_sources == 1:
        agreement_score = 1.0
        consensus_confidence = round(avg_confidence, 3)

    return {
        "extractor": extractor_route,
        "actual_extractor": str(primary_result.get("actual_extractor", primary_result.get("extractor", "unknown"))),
        "entities": merged_entities,
        "confidence": consensus_confidence,
        "latency_ms": latency_ms,
        "raw_text": raw_text,
        "notes": notes,
        "consensus_sources": source_names,
        "consensus_source_count": total_sources,
        "agreement_score": agreement_score,
        "agreement_avg_confidence": avg_confidence,
        "disagreement_flag": disagreement_flag,
    }


def _entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    entity_type = str(entity.get("type", "")).strip().lower()
    text = str(entity.get("text", "")).strip().lower()
    return entity_type, text


def _entity_weight(entity: dict[str, Any]) -> int:
    structured = entity.get("structured")
    structured_weight = len(structured) if isinstance(structured, dict) else 0
    return len(entity) + structured_weight


def _support_ratio(source_count: int, total_sources: int) -> float:
    if total_sources <= 1:
        return 1.0
    return max(source_count - 1, 0) / max(total_sources - 1, 1)
