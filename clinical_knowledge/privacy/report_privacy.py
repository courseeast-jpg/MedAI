"""Public report privacy checker for CKA-B02.

check_public_report_payload(payload) → ReportPrivacyCheck

Recursively scans any dict/list/string payload for obvious private content.
Must be called before writing any public report.
Never includes raw leaked values in its own output.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, List

from clinical_knowledge.privacy.patterns import (
    ALWAYS_BLOCK_CATEGORIES,
    PHI_CATEGORIES,
    PRIVATE_REF_CATEGORIES,
    PRIVACY_PATTERNS,
)
from clinical_knowledge.privacy.sanitizer import Finding, sanitize_text

_SALT = "medai_cka_b02_report_check_v1"


def _redact_example(original: str, token: str) -> str:
    """Return a safe redacted example: token + hash prefix."""
    h = hashlib.sha256(f"{_SALT}:{original}".encode()).hexdigest()[:8]
    return f"{token} [hash:{h}]"


@dataclass
class ReportPrivacyCheck:
    passed: bool
    raw_phi_logged_in_public_reports: bool
    private_filename_path_leaks: int
    secret_leaks: int
    leak_examples_redacted: List[str]   # safe examples only (no raw values)
    checked_keys_count: int
    checked_strings_count: int


def check_public_report_payload(payload: Any) -> ReportPrivacyCheck:
    """Recursively scan *payload* for private/sensitive content.

    Returns ReportPrivacyCheck. The result itself is safe for public logs.
    """
    all_findings: List[Finding] = []
    strings_checked = [0]
    keys_checked = [0]

    def _walk(node: Any, _depth: int = 0) -> None:
        if _depth > 50:
            return
        if isinstance(node, str):
            strings_checked[0] += 1
            result = sanitize_text(node)
            all_findings.extend(result.findings)
        elif isinstance(node, dict):
            for k, v in node.items():
                keys_checked[0] += 1
                strings_checked[0] += 1
                # Also check the key string
                key_result = sanitize_text(str(k))
                all_findings.extend(key_result.findings)
                _walk(v, _depth + 1)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item, _depth + 1)
        elif node is not None:
            strings_checked[0] += 1
            result = sanitize_text(str(node))
            all_findings.extend(result.findings)

    _walk(payload)

    phi_leaks = [f for f in all_findings if f.category in PHI_CATEGORIES]
    private_ref_leaks = [f for f in all_findings if f.category in PRIVATE_REF_CATEGORIES]
    secret_leaks = [f for f in all_findings if f.category in ALWAYS_BLOCK_CATEGORIES]

    # Build redacted examples (safe — no raw values)
    redacted_examples: List[str] = []
    seen_tokens: set = set()
    for f in (phi_leaks + private_ref_leaks + secret_leaks):
        if f.token not in seen_tokens:
            redacted_examples.append(_redact_example(f.original, f.token))
            seen_tokens.add(f.token)
        if len(redacted_examples) >= 10:
            break

    passed = len(phi_leaks) == 0 and len(private_ref_leaks) == 0 and len(secret_leaks) == 0

    return ReportPrivacyCheck(
        passed=passed,
        raw_phi_logged_in_public_reports=len(phi_leaks) > 0,
        private_filename_path_leaks=len(private_ref_leaks),
        secret_leaks=len(secret_leaks),
        leak_examples_redacted=redacted_examples,
        checked_keys_count=keys_checked[0],
        checked_strings_count=strings_checked[0],
    )
