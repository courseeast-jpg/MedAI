"""Deterministic PII/PHI detection for outbound privacy gating.

Findings intentionally omit matched raw text. Reports and audit artifacts should
store only type, count, offsets, confidence, and source rule metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class PIIFinding:
    type: str
    start: int
    end: int
    confidence: float
    source_rule: str

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source_rule": self.source_rule,
        }


@dataclass(frozen=True)
class PIIReport:
    pii_found: bool
    findings: list[PIIFinding] = field(default_factory=list)
    risk_level: str = "none"
    warnings: list[str] = field(default_factory=list)

    @property
    def counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.type] = counts.get(finding.type, 0) + 1
        return counts

    def to_safe_dict(self) -> dict:
        return {
            "pii_found": self.pii_found,
            "findings": [finding.to_dict() for finding in self.findings],
            "risk_level": self.risk_level,
            "warnings": list(self.warnings),
            "counts_by_type": self.counts_by_type,
        }


RULES: tuple[tuple[str, str, str, float], ...] = (
    ("EMAIL", r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "email_regex", 0.99),
    ("PHONE", r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)", "us_phone_regex", 0.95),
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b", "ssn_regex", 0.99),
    ("DOB", r"\b(?:DOB|Date of Birth|Birth Date)\s*[:#-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "dob_label_regex", 0.98),
    ("DATE", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "date_regex", 0.72),
    ("DATE", r"\b\d{4}-\d{2}-\d{2}\b", "iso_date_regex", 0.72),
    ("MRN", r"\b(?:MRN|Medical Record(?: Number)?|Patient ID)\s*[:#-]?\s*[A-Z0-9-]{5,}\b", "mrn_label_regex", 0.98),
    ("INSURANCE_ID", r"\b(?:Insurance|Policy|Member ID|Account)\s*(?:ID|No\.?|Number|#)?\s*[:#-]?\s*[A-Z0-9-]{5,}\b", "insurance_label_regex", 0.94),
    ("ADDRESS", r"\b\d{1,6}\s+[A-Z][A-Za-z0-9.'-]*(?:\s+[A-Z][A-Za-z0-9.'-]*){0,4}\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Drive|Dr)\b", "address_regex", 0.85),
    ("FACILITY", r"\b(?:Hospital|Clinic|Medical Center|Laboratory|Lab|Health System)\s*[:#-]?\s*[A-Z][A-Za-z&.' -]{2,}\b", "facility_label_regex", 0.80),
    ("PERSON", r"\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Patient|Name)\s*[:#-]?\s*[A-Z][a-z]{1,30}\s+[A-Z][a-z]{1,30}\b", "person_label_regex", 0.88),
    ("RU_DOB", r"(?:дата\s+рождения|д\.?\s*р\.?|родился|родилась)\s*[:#-]?\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", "ru_dob_regex", 0.98),
    ("RU_PHONE", r"(?:телефон|тел\.?)\s*[:#-]?\s*(?:\+?7|8)?[\s(-]*\d{3}[\s)-]*\d{3}[-\s]?\d{2}[-\s]?\d{2}", "ru_phone_regex", 0.95),
    ("RU_ADDRESS", r"(?:адрес)\s*[:#-]?\s*[А-ЯЁа-яё0-9.,\-\s]{8,80}", "ru_address_regex", 0.86),
    ("RU_INSURANCE_ID", r"(?:полис|страховой\s+полис)\s*[:#-]?\s*[А-ЯЁA-Z0-9-]{5,}", "ru_insurance_regex", 0.94),
    ("RU_ID", r"\b\d{3}-\d{3}-\d{3}\s?\d{2}\b", "snils_regex", 0.97),
    ("RU_ID", r"(?:паспорт)\s*[:#-]?\s*\d{2}\s?\d{2}\s?\d{6}\b", "ru_passport_regex", 0.96),
    ("RU_PERSON", r"(?:ФИО|Пациент|Пациентка|Врач)\s*[:#-]?\s*[А-ЯЁ][а-яё]{2,30}\s+[А-ЯЁ][а-яё]{2,30}", "ru_person_label_regex", 0.90),
    ("RU_FACILITY", r"(?:клиника|больница|медицинский\s+центр|лаборатория)\s*[:#-]?\s*[А-ЯЁA-Z][А-ЯЁа-яёA-Za-z0-9\"' .-]{3,60}", "ru_facility_regex", 0.82),
)


def detect_pii(text: str | None) -> PIIReport:
    if not text:
        return PIIReport(pii_found=False, risk_level="none")

    findings = _dedupe_findings(_scan_rules(text))
    warnings: list[str] = []
    if _has_cyrillic(text) and not any(f.type.startswith("RU_") for f in findings):
        warnings.append("cyrillic_text_without_structured_pii_match")
    return PIIReport(
        pii_found=bool(findings),
        findings=findings,
        risk_level=_risk_level(findings),
        warnings=warnings,
    )


def _scan_rules(text: str) -> list[PIIFinding]:
    findings: list[PIIFinding] = []
    for pii_type, pattern, source_rule, confidence in RULES:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            findings.append(
                PIIFinding(
                    type=pii_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    source_rule=source_rule,
                )
            )
    return findings


def _dedupe_findings(findings: Iterable[PIIFinding]) -> list[PIIFinding]:
    ordered = sorted(findings, key=lambda item: (item.start, -(item.end - item.start), -item.confidence))
    selected: list[PIIFinding] = []
    occupied: list[range] = []
    for finding in ordered:
        span = range(finding.start, finding.end)
        if any(_overlaps(span, existing) for existing in occupied):
            continue
        selected.append(finding)
        occupied.append(span)
    return sorted(selected, key=lambda item: item.start)


def _overlaps(left: range, right: range) -> bool:
    return left.start < right.stop and right.start < left.stop


def _risk_level(findings: list[PIIFinding]) -> str:
    if not findings:
        return "none"
    high_types = {"PERSON", "RU_PERSON", "DOB", "RU_DOB", "SSN", "MRN", "INSURANCE_ID", "RU_INSURANCE_ID", "RU_ID"}
    if any(finding.type in high_types for finding in findings):
        return "high"
    if len(findings) >= 2:
        return "medium"
    return "low"


def _has_cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04ff" for char in text)
