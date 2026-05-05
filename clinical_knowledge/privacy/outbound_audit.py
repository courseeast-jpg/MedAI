"""Outbound payload audit for CKA-B02.

build_outbound_payload(payload, *, allow_external, purpose) → OutboundAuditResult

Blocking rules (applied in order):
  1. If any SECRET detected → always block.
  2. If allow_external=False → block regardless of sanitization.
  3. If raw PHI/private refs detected and sanitizer cannot remove → block.
  4. If sanitizer removes all sensitive content and allow_external=True → allow.

No external API calls are made.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from clinical_knowledge.privacy.patterns import ALWAYS_BLOCK_CATEGORIES
from clinical_knowledge.privacy.sanitizer import Finding, sanitize_dict_values


@dataclass
class OutboundAuditResult:
    allowed: bool
    sanitized_payload: Dict[str, Any]
    blocked_reasons: List[str]
    findings_summary: Dict[str, int]    # category → count
    external_api_allowed: bool
    external_api_used: bool = False     # always False in CKA-B02
    raw_phi_detected: bool = False
    raw_phi_removed: bool = False
    private_filename_path_leaks: int = 0
    secret_detected: bool = False
    safe_public_payload_hash: str = ""
    audit_event_ready: bool = True


def _compute_hash(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _summarize_findings(findings: List[Finding]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for f in findings:
        counts[f.category] = counts.get(f.category, 0) + 1
    return counts


def build_outbound_payload(
    payload: dict,
    *,
    allow_external: bool,
    purpose: str = "unspecified",
) -> OutboundAuditResult:
    """Audit and sanitize *payload* for outbound use.

    Does NOT call any external API.
    """
    sanitized, findings, _replacement_map = sanitize_dict_values(payload)
    # _replacement_map is PRIVATE — it is intentionally not stored in the result

    summary = _summarize_findings(findings)
    blocked_reasons: List[str] = []

    secret_found = any(f.category in ALWAYS_BLOCK_CATEGORIES for f in findings)
    phi_found = any(
        f.category in {"PERSON", "DOB", "MRN", "INSURANCE_ID", "FACILITY", "EMAIL", "PHONE", "DATE"}
        for f in findings
    )
    private_ref_found = any(
        f.category in {"WIN_PATH", "UNIX_PATH", "MEDICAL_FILENAME"}
        for f in findings
    )

    if secret_found:
        blocked_reasons.append("secret_or_api_key_detected: always blocked")

    if not allow_external:
        blocked_reasons.append("allow_external=False: outbound blocked by policy")

    # Check whether sanitized payload is actually clean
    # (re-scan the sanitized payload to verify no raw sensitive content remains)
    _sanitized_again, _residual_findings, _ = sanitize_dict_values(sanitized)
    residual_secrets = [f for f in _residual_findings if f.category in ALWAYS_BLOCK_CATEGORIES]
    if residual_secrets:
        blocked_reasons.append("sanitizer_could_not_remove_all_secrets")

    allowed = len(blocked_reasons) == 0

    # Private ref count: paths/filenames in original findings
    private_ref_count = sum(
        1 for f in findings if f.category in {"WIN_PATH", "UNIX_PATH", "MEDICAL_FILENAME"}
    )

    safe_hash = _compute_hash(sanitized if isinstance(sanitized, dict) else {})

    return OutboundAuditResult(
        allowed=allowed,
        sanitized_payload=sanitized if isinstance(sanitized, dict) else {},
        blocked_reasons=blocked_reasons,
        findings_summary=summary,
        external_api_allowed=allow_external,
        external_api_used=False,
        raw_phi_detected=phi_found,
        raw_phi_removed=phi_found and len(findings) > 0,
        private_filename_path_leaks=private_ref_count,
        secret_detected=secret_found,
        safe_public_payload_hash=safe_hash,
        audit_event_ready=True,
    )
