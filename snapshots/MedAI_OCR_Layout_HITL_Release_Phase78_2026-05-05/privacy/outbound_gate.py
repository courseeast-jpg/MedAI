"""Fail-closed privacy gate for cloud/external API payloads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from app.config import MEDAI_ALLOW_EXTERNAL_API, MEDAI_LOCAL_ONLY, MEDAI_REQUIRE_PII_SCRUB
from privacy.pii_detector import detect_pii
from privacy.pii_redactor import redact_pii


LOCAL_PROVIDERS = {"local", "spacy", "phi3", "ollama", "tesseract", "local_ocr"}
EXTERNAL_PROVIDERS = {"gemini", "claude", "openai", "deepl", "dxgpt", "patientnotes_ddi", "anthropic"}


@dataclass(frozen=True)
class OutboundGateDecision:
    allowed: bool
    mode: str
    provider: str
    payload_text: str = ""
    pii_findings_count: int = 0
    redaction_counts: dict[str, int] = field(default_factory=dict)
    payload_hash: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def payload_redacted(self) -> bool:
        return bool(self.redaction_counts)

    def to_safe_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "mode": self.mode,
            "provider": self.provider,
            "pii_findings_count": self.pii_findings_count,
            "redaction_counts": dict(self.redaction_counts),
            "payload_hash": self.payload_hash,
            "warnings": list(self.warnings),
            "payload_redacted": self.payload_redacted,
        }


def guard_external_payload(
    *,
    provider: str,
    text: str | None,
    local_only: bool | None = None,
    allow_external_api: bool | None = None,
    require_pii_scrub: bool | None = None,
    redaction_failed: bool = False,
) -> OutboundGateDecision:
    provider_name = (provider or "unknown").lower()
    source_text = text or ""
    effective_local_only = MEDAI_LOCAL_ONLY if local_only is None else local_only
    effective_allow_external = MEDAI_ALLOW_EXTERNAL_API if allow_external_api is None else allow_external_api
    effective_require_scrub = MEDAI_REQUIRE_PII_SCRUB if require_pii_scrub is None else require_pii_scrub

    if provider_name in LOCAL_PROVIDERS:
        return _decision(True, "local_provider", provider_name, source_text, {}, [], 0)
    if effective_local_only:
        report = detect_pii(source_text)
        return _decision(False, "local_only", provider_name, "", {}, list(report.warnings), len(report.findings))
    if not effective_allow_external:
        report = detect_pii(source_text)
        return _decision(False, "blocked_unknown_privacy_state", provider_name, "", {}, list(report.warnings), len(report.findings))
    if redaction_failed:
        report = detect_pii(source_text)
        return _decision(False, "blocked_redaction_failed", provider_name, "", {}, list(report.warnings), len(report.findings))

    report = detect_pii(source_text)
    if not effective_require_scrub and report.pii_found:
        return _decision(False, "blocked_pii_detected", provider_name, "", {}, list(report.warnings), len(report.findings))

    redacted = redact_pii(source_text)
    if not redacted.redaction_passed:
        return _decision(
            False,
            "blocked_redaction_failed",
            provider_name,
            "",
            redacted.redaction_counts,
            redacted.warnings,
            len(report.findings),
        )

    post_report = detect_pii(redacted.redacted_text)
    if post_report.pii_found:
        return _decision(
            False,
            "blocked_pii_detected",
            provider_name,
            "",
            redacted.redaction_counts,
            redacted.warnings + post_report.warnings,
            len(report.findings),
        )

    return _decision(
        True,
        "external_allowed_redacted",
        provider_name,
        redacted.redacted_text,
        redacted.redaction_counts,
        redacted.warnings,
        len(report.findings),
    )


def _decision(
    allowed: bool,
    mode: str,
    provider: str,
    payload_text: str,
    redaction_counts: dict[str, int],
    warnings: list[str],
    pii_findings_count: int,
) -> OutboundGateDecision:
    payload_hash = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()[:16] if payload_text else ""
    return OutboundGateDecision(
        allowed=allowed,
        mode=mode,
        provider=provider,
        payload_text=payload_text,
        pii_findings_count=pii_findings_count,
        redaction_counts=redaction_counts,
        payload_hash=payload_hash,
        warnings=warnings,
    )
