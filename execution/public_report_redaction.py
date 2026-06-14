"""Public-report redaction helper (17C-R2-R9).

Replaces private Windows/Unix filesystem paths and secret-like strings in *public*
report payloads with safe, debuggable labels, so a report can pass
`clinical_knowledge.privacy.check_public_report_payload`.

It is driven by the SAME detector the privacy checker uses
(`clinical_knowledge.privacy.sanitizer.sanitize_text`), so anything the checker would
flag as a private path (WIN_PATH / UNIX_PATH / MEDICAL_FILENAME) or a secret
(ALWAYS_BLOCK / SECRET) is replaced. This never weakens the checker — it removes the
leak from the report text rather than the rule.

Labels (status-preserving, no raw value):
  - PRIVATE_STAGING_PATH_REDACTED     (live request/response staging)
  - PRIVATE_CHECKPOINT_PATH_REDACTED  (durable checkpoint)
  - PRIVATE_EVIDENCE_PATH_REDACTED    (preserved failure evidence)
  - PRIVATE_PATH_REDACTED             (other private path)
  - SECRET_REDACTED                   (secret-like token / 64-hex digest)

No raw response body, tokenized payload, token map, private identifier, or credential is
ever emitted by this module.
"""
from __future__ import annotations

import json
from typing import Any

from clinical_knowledge.privacy.patterns import ALWAYS_BLOCK_CATEGORIES, PRIVATE_REF_CATEGORIES
from clinical_knowledge.privacy.sanitizer import sanitize_text

SECRET_LABEL = "SECRET_REDACTED"
DEFAULT_PATH_LABEL = "PRIVATE_PATH_REDACTED"
EVIDENCE_LABEL = "PRIVATE_EVIDENCE_PATH_REDACTED"
CHECKPOINT_LABEL = "PRIVATE_CHECKPOINT_PATH_REDACTED"
STAGING_LABEL = "PRIVATE_STAGING_PATH_REDACTED"

_STAGING_HINTS = ("staging", "live_batch", "outbound", "manifest", "response", "request", "batch")


def _path_label_for(blob: str) -> str:
    low = blob.lower()
    if "evidence" in low or "failed_evidence" in low:
        return EVIDENCE_LABEL
    if "checkpoint" in low:
        return CHECKPOINT_LABEL
    if any(h in low for h in _STAGING_HINTS):
        return STAGING_LABEL
    return DEFAULT_PATH_LABEL


def _classify(value: str) -> "str | None":
    """Return 'secret' / 'path' / None for a single string per the privacy detector."""
    cats = {f.category for f in sanitize_text(value).findings}
    if cats & ALWAYS_BLOCK_CATEGORIES:
        return "secret"
    if cats & PRIVATE_REF_CATEGORIES:
        return "path"
    return None


def redact_json_value(obj: Any) -> "tuple[Any, int, int]":
    """Recursively redact a parsed JSON value. Returns (redacted, path_count, secret_count)."""
    counts = {"path": 0, "secret": 0}

    def rec(node: Any, key: str = "") -> Any:
        if isinstance(node, str):
            kind = _classify(node)
            if kind == "secret":
                counts["secret"] += 1
                return SECRET_LABEL
            if kind == "path":
                counts["path"] += 1
                return _path_label_for(str(key) + " " + node)
            return node
        if isinstance(node, dict):
            return {k: rec(v, str(k)) for k, v in node.items()}
        if isinstance(node, list):
            return [rec(v, key) for v in node]
        return node

    redacted = rec(obj)
    return redacted, counts["path"], counts["secret"]


def redact_text(text: str) -> "tuple[str, int, int]":
    """Redact private paths/secrets in free text (markdown/csv) using detector offsets.

    Replaces from rightmost finding to leftmost so offsets stay valid."""
    findings = [f for f in sanitize_text(text).findings
                if f.category in (PRIVATE_REF_CATEGORIES | ALWAYS_BLOCK_CATEGORIES)]
    findings.sort(key=lambda f: f.start, reverse=True)
    out = text
    n_path = n_secret = 0
    for f in findings:
        original = out[f.start:f.end]
        if f.category in ALWAYS_BLOCK_CATEGORIES:
            label = SECRET_LABEL
            n_secret += 1
        else:
            label = _path_label_for(original)
            n_path += 1
        out = out[:f.start] + label + out[f.end:]
    return out, n_path, n_secret


def redact_report_file(text: str, is_json: bool) -> "tuple[str, int, int]":
    """Redact one report's text. JSON is redacted structurally (stays valid JSON);
    other text is redacted by detector offsets. Returns (text, path_count, secret_count)."""
    if is_json:
        try:
            obj = json.loads(text)
        except ValueError:
            return redact_text(text)
        redacted, np, ns = redact_json_value(obj)
        return json.dumps(redacted, indent=2, ensure_ascii=True), np, ns
    return redact_text(text)


__all__ = [
    "SECRET_LABEL", "DEFAULT_PATH_LABEL", "EVIDENCE_LABEL", "CHECKPOINT_LABEL",
    "STAGING_LABEL", "redact_json_value", "redact_text", "redact_report_file",
]
