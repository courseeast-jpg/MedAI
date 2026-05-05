"""PII/PHI sanitizer for CKA-B02.

sanitize_text(text) → SanitizedText

- Replaces detected sensitive spans with tokens like [PERSON_1], [DOB_1].
- Repeated identical values map to the same token within one call.
- replacement_map is private-only; safe_public_findings contains only
  category/count/hashed refs.
- Does not call any external API.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from clinical_knowledge.privacy.patterns import (
    ALWAYS_BLOCK_CATEGORIES,
    PHI_CATEGORIES,
    PRIVATE_REF_CATEGORIES,
    PRIVACY_PATTERNS,
    PrivacyPattern,
)

_SALT = "medai_cka_b02_sanitizer_v1"


def _hash_value(raw: str) -> str:
    return hashlib.sha256(f"{_SALT}:{raw}".encode()).hexdigest()[:12]


@dataclass
class Finding:
    category: str
    severity: str
    start: int
    end: int
    original: str        # private — never in public reports
    token: str           # replacement token e.g. [PERSON_1]


@dataclass
class SanitizedText:
    sanitized_text: str
    findings: List[Finding]

    # PRIVATE — maps token → original value. Must NOT appear in public reports.
    replacement_map: Dict[str, str]

    # Safe for public: category/count/hash only
    safe_public_findings: List[Dict]

    raw_phi_detected: bool
    private_reference_detected: bool
    secret_detected: bool


def sanitize_text(text: str) -> SanitizedText:
    """Detect and replace all sensitive spans in *text*.

    Returns a SanitizedText. The caller is responsible for keeping
    replacement_map private.
    """
    # Step 1: collect all non-overlapping matches across all patterns,
    # sorted by start position (earlier first; longer match wins on ties).
    raw_spans: List[Tuple[int, int, str, PrivacyPattern]] = []

    for pat in PRIVACY_PATTERNS:
        for m in pat.pattern.finditer(text):
            raw_spans.append((m.start(), m.end(), m.group(0), pat))

    # Sort by start; on tie prefer longer span
    raw_spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Remove overlapping spans (keep first/longest)
    selected: List[Tuple[int, int, str, PrivacyPattern]] = []
    last_end = -1
    for start, end, matched, pat in raw_spans:
        if start >= last_end:
            selected.append((start, end, matched, pat))
            last_end = end

    # Step 2: build replacement tokens, tracking seen values for consistency
    value_to_token: Dict[str, str] = {}    # original → token (within this call)
    category_counters: Dict[str, int] = {}

    def _get_token(original: str, category: str) -> str:
        key = f"{category}:{original}"
        if key not in value_to_token:
            category_counters[category] = category_counters.get(category, 0) + 1
            token = f"[{category}_{category_counters[category]}]"
            value_to_token[key] = token
        return value_to_token[key]

    # Step 3: build findings and replacement_map
    findings: List[Finding] = []
    replacement_map: Dict[str, str] = {}  # token → original (PRIVATE)

    for start, end, original, pat in selected:
        token = _get_token(original, pat.replacement_prefix)
        replacement_map[token] = original
        findings.append(Finding(
            category=pat.category,
            severity=pat.severity.value,
            start=start,
            end=end,
            original=original,
            token=token,
        ))

    # Step 4: build sanitized text (replace in reverse order to preserve positions)
    sanitized = text
    for finding in reversed(sorted(findings, key=lambda f: f.start)):
        sanitized = sanitized[: finding.start] + finding.token + sanitized[finding.end :]

    # Step 5: build safe_public_findings (no raw values)
    safe: List[Dict] = []
    for f in findings:
        safe.append({
            "category": f.category,
            "severity": f.severity,
            "token": f.token,
            "value_hash": _hash_value(f.original),
        })

    raw_phi = any(f.category in PHI_CATEGORIES for f in findings)
    private_ref = any(f.category in PRIVATE_REF_CATEGORIES for f in findings)
    secret = any(f.category in ALWAYS_BLOCK_CATEGORIES for f in findings)

    return SanitizedText(
        sanitized_text=sanitized,
        findings=findings,
        replacement_map=replacement_map,
        safe_public_findings=safe,
        raw_phi_detected=raw_phi,
        private_reference_detected=private_ref,
        secret_detected=secret,
    )


def sanitize_dict_values(obj: object) -> Tuple[object, List[Finding], Dict[str, str]]:
    """Recursively sanitize all string values in a dict/list/scalar.

    Returns (sanitized_obj, all_findings, merged_replacement_map).
    replacement_map is PRIVATE.
    """
    all_findings: List[Finding] = []
    merged_map: Dict[str, str] = {}

    def _walk(node: object) -> object:
        if isinstance(node, str):
            result = sanitize_text(node)
            all_findings.extend(result.findings)
            merged_map.update(result.replacement_map)
            return result.sanitized_text
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    sanitized = _walk(obj)
    return sanitized, all_findings, merged_map
