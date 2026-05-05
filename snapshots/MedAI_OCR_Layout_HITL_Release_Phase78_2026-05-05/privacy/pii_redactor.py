"""Token-based PII redaction for privacy-gated outbound payloads."""

from __future__ import annotations

from dataclasses import dataclass, field

from privacy.pii_detector import PIIFinding, detect_pii


TOKEN_TYPES = {
    "PERSON": "PERSON",
    "DATE": "DATE",
    "DOB": "DOB",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
    "ADDRESS": "ADDRESS",
    "MRN": "MRN",
    "INSURANCE_ID": "INSURANCE_ID",
    "SSN": "ID",
    "FACILITY": "FACILITY",
    "RU_PERSON": "RU_PERSON",
    "RU_DOB": "DOB",
    "RU_PHONE": "PHONE",
    "RU_ADDRESS": "ADDRESS",
    "RU_INSURANCE_ID": "INSURANCE_ID",
    "RU_ID": "RU_ID",
    "RU_FACILITY": "FACILITY",
}


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    redaction_counts: dict[str, int]
    redaction_passed: bool
    warnings: list[str] = field(default_factory=list)
    redaction_map: dict[str, str] = field(default_factory=dict, repr=False)

    def to_safe_dict(self) -> dict:
        return {
            "redaction_counts": dict(self.redaction_counts),
            "redaction_passed": self.redaction_passed,
            "warnings": list(self.warnings),
        }


def redact_pii(text: str | None) -> RedactionResult:
    source = text or ""
    report = detect_pii(source)
    if not report.findings:
        return RedactionResult(
            redacted_text=source,
            redaction_counts={},
            redaction_passed=True,
            warnings=list(report.warnings),
            redaction_map={},
        )

    counters: dict[str, int] = {}
    redaction_map: dict[str, str] = {}
    redacted = source
    for finding in sorted(report.findings, key=lambda item: item.start, reverse=True):
        token = _token_for(finding, counters)
        redaction_map[token] = source[finding.start:finding.end]
        redacted = redacted[:finding.start] + token + redacted[finding.end:]

    post_report = detect_pii(redacted)
    warnings = list(report.warnings)
    if post_report.pii_found:
        warnings.append("redacted_payload_still_contains_pii")
    return RedactionResult(
        redacted_text=redacted,
        redaction_counts=dict(counters),
        redaction_passed=not post_report.pii_found,
        warnings=warnings,
        redaction_map=redaction_map,
    )


def _token_for(finding: PIIFinding, counters: dict[str, int]) -> str:
    token_type = TOKEN_TYPES.get(finding.type, finding.type)
    counters[token_type] = counters.get(token_type, 0) + 1
    return f"[{token_type}_{counters[token_type]}]"
