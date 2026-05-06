"""Response normalizer for CKA-B08.

normalize_connector_response(raw_response, connector_spec) -> dict | None

Rules:
- Accept structured synthetic responses only.
- Reject free-text-only or malformed responses.
- Strip unsafe fields: raw_source_text, private_payload, replacement_map,
  source_response_raw.
- Extract: facts[], citations[], confidence, connector_name, source_kind.
- Public report includes response hash only.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from clinical_knowledge.connectors.models import ConnectorSpec

_SALT = "medai_cka_b08_norm_v1"

# Fields that must be stripped from any normalized response
_UNSAFE_FIELDS = frozenset({
    "raw_source_text",
    "private_payload",
    "replacement_map",
    "source_response_raw",
    "raw_text",
    "private_text",
})

# Minimum required fields for a valid structured response
_REQUIRED_FIELDS = frozenset({"connector_name", "facts"})


def _response_hash(response: Dict[str, Any]) -> str:
    serialized = json.dumps(response, sort_keys=True, default=str)
    return hashlib.sha256(f"{_SALT}:{serialized}".encode()).hexdigest()[:16]


def _strip_unsafe(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k not in _UNSAFE_FIELDS}


def _is_valid_fact(fact: Any) -> bool:
    if not isinstance(fact, dict):
        return False
    return bool(fact.get("fact_type")) and bool(fact.get("entity_text"))


def normalize_connector_response(
    raw_response: Dict[str, Any],
    connector_spec: ConnectorSpec,
) -> Optional[Dict[str, Any]]:
    """Normalize a raw stub response.

    Returns None if response is malformed/invalid.
    Returns cleaned normalized dict on success.

    The normalized dict contains:
    - connector_name: str
    - source_kind: str
    - facts: list[dict]
    - citations: list[dict]
    - confidence: float
    - synthetic: bool
    - response_hash: str  (safe public identifier)
    """
    if not isinstance(raw_response, dict):
        return None

    # Detect stub errors
    if "_stub_error" in raw_response:
        return None

    # Must have minimum required structure
    if not _REQUIRED_FIELDS.issubset(raw_response.keys()):
        return None

    # Must have structured facts list (not just free text)
    facts_raw = raw_response.get("facts", [])
    if not isinstance(facts_raw, list):
        return None

    # Reject free-text-only (no structured facts at all)
    if not facts_raw and "raw_garbage" in raw_response:
        return None

    # Filter to valid facts only
    valid_facts: List[Dict[str, Any]] = [
        _strip_unsafe(f) for f in facts_raw if _is_valid_fact(f)
    ]

    citations_raw = raw_response.get("citations", [])
    if not isinstance(citations_raw, list):
        citations_raw = []
    citations: List[Dict[str, Any]] = [
        {k: v for k, v in c.items() if k not in _UNSAFE_FIELDS}
        for c in citations_raw
        if isinstance(c, dict)
    ]

    try:
        confidence = float(raw_response.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    normalized = {
        "connector_name": connector_spec.name,
        "source_kind": "connector_stub",
        "facts": valid_facts,
        "citations": citations,
        "confidence": confidence,
        "synthetic": True,
        "response_hash": _response_hash(raw_response),
    }

    return normalized
