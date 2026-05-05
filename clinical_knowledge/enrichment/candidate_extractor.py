"""Deterministic candidate extraction for CKA-B06.

Extracts EnrichmentCandidate objects from structured synthetic JSON responses.

Rules:
- Source must be structured synthetic data — no free-text clinical prose.
- No LLM calls. No external API calls.
- ai_response / connector_stub → proposed_trust_level=3
- web_unverified + source_quality=high → trust_level=4
- web_unverified + source_quality=low or missing → trust_level=5
- proposed_tier is always hypothesis
- Invalid/missing facts are skipped with safe counts only
- No clinical text generated during extraction
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List

from clinical_knowledge.enrichment.models import EnrichmentCandidate, EnrichmentSourceKind
from clinical_knowledge.safe_ids import make_safe_record_id, new_record_id

_SALT = "medai_cka_b06_extractor_v1"

_REQUIRED_FACT_FIELDS = {"fact_type", "entity_text"}
_MAX_ENTITY_LENGTH = 500

_VALID_SOURCE_KINDS = {k.value for k in EnrichmentSourceKind}


def _hash_response(payload: Any) -> str:
    import json
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(f"{_SALT}:{raw}".encode()).hexdigest()[:24]


def _safe_candidate_id(candidate_id: str) -> str:
    digest = hashlib.sha256(f"{_SALT}:cand:{candidate_id}".encode()).hexdigest()[:16]
    return f"cka_cand_{digest}"


def _resolve_trust_level(source_kind: EnrichmentSourceKind, source_quality: str) -> int:
    if source_kind in (EnrichmentSourceKind.AI_RESPONSE, EnrichmentSourceKind.CONNECTOR_STUB):
        return 3
    if source_kind == EnrichmentSourceKind.MANUAL_REVIEW_PREPARED:
        return 3
    # WEB_UNVERIFIED
    if source_quality == "high":
        return 4
    return 5


def _is_valid_fact(fact: Any) -> bool:
    if not isinstance(fact, dict):
        return False
    for f in _REQUIRED_FACT_FIELDS:
        if not fact.get(f) or not isinstance(fact[f], str) or not fact[f].strip():
            return False
    entity = fact.get("entity_text", "")
    if len(entity) > _MAX_ENTITY_LENGTH:
        return False
    conf = fact.get("confidence", 0.5)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        return False
    if not (0.0 <= conf <= 1.0):
        return False
    return True


def extract_enrichment_candidates_from_structured_response(
    response_payload: Dict[str, Any],
) -> List[EnrichmentCandidate]:
    """Extract EnrichmentCandidate list from a structured synthetic response.

    Input must be structured synthetic JSON-like data only.
    Returns empty list (not an exception) for invalid/empty payloads.
    """
    if not isinstance(response_payload, dict):
        return []

    source_name = str(response_payload.get("source_name", "unknown_stub"))
    source_kind_raw = str(response_payload.get("source_kind", "connector_stub"))
    specialty = str(response_payload.get("specialty", "general"))
    source_quality = str(response_payload.get("source_quality", ""))

    if source_kind_raw not in _VALID_SOURCE_KINDS:
        source_kind_raw = EnrichmentSourceKind.CONNECTOR_STUB.value
    source_kind = EnrichmentSourceKind(source_kind_raw)

    trust_level = _resolve_trust_level(source_kind, source_quality)
    response_hash = _hash_response(response_payload)

    facts = response_payload.get("facts", [])
    if not isinstance(facts, list):
        return []

    candidates: List[EnrichmentCandidate] = []
    for fact in facts:
        if not _is_valid_fact(fact):
            continue

        cid = new_record_id()
        safe_cid = _safe_candidate_id(cid)
        entity_text = fact["entity_text"].strip()[:_MAX_ENTITY_LENGTH]
        fact_type = str(fact["fact_type"]).strip()
        confidence = float(fact.get("confidence", 0.5))
        structured = fact.get("structured", {})
        if not isinstance(structured, dict):
            structured = {}

        safe_summary = {
            "safe_candidate_id": safe_cid,
            "source_kind": source_kind.value,
            "source_name": source_name,
            "specialty": specialty,
            "fact_type": fact_type,
            "proposed_trust_level": trust_level,
            "proposed_tier": "hypothesis",
            "synthetic": True,
        }

        candidates.append(
            EnrichmentCandidate(
                candidate_id=cid,
                safe_candidate_id=safe_cid,
                source_kind=source_kind,
                source_name=source_name,
                source_response_hash=response_hash,
                fact_type=fact_type,
                entity_text=entity_text,
                structured=structured,
                specialty=specialty,
                confidence=confidence,
                proposed_trust_level=trust_level,
                proposed_tier="hypothesis",
                extraction_method="synthetic_structured_enrichment",
                safe_public_summary=safe_summary,
            )
        )

    return candidates
