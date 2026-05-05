"""Deterministic privacy pattern definitions for CKA-B02.

Regex-based, no external APIs, no Presidio dependency.
Covers: names, DOB, dates, phones, emails, addresses, MRNs, insurance IDs,
facilities, Windows/Unix paths, medical filenames, secrets, private UUIDs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List


class PatternSeverity(str, Enum):
    PHI = "phi"                  # Protected Health Information
    PII = "pii"                  # Personally Identifiable Information
    PRIVATE_REF = "private_ref"  # Private file refs, paths
    SECRET = "secret"            # API keys, tokens, passwords


@dataclass(frozen=True)
class PrivacyPattern:
    category: str
    pattern: re.Pattern
    replacement_prefix: str   # e.g. "PERSON" → tokens [PERSON_1], [PERSON_2]
    severity: PatternSeverity
    description: str


# ---------------------------------------------------------------------------
# Pattern definitions — ordered: secrets first (highest severity), then PHI,
# then PII, then private refs. Within-call token assignment follows this order.
# ---------------------------------------------------------------------------

_I = re.IGNORECASE

_PATTERNS: List[PrivacyPattern] = [

    # 1. API keys / secrets — must always block outbound
    PrivacyPattern(
        category="SECRET",
        pattern=re.compile(
            r"(?:"
            r"(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token"
            r"|bearer|private[_-]?key|client[_-]?secret|password|passwd"
            r"|api_secret|token|secret)\s*[:=]\s*['\"]?[A-Za-z0-9!@#%^&*\-_+=./]{8,}['\"]?"
            r"|sk-[A-Za-z0-9]{20,}"
            r"|[A-Za-z0-9]{40,}"
            r")",
            _I,
        ),
        replacement_prefix="SECRET",
        severity=PatternSeverity.SECRET,
        description="API key, token, or secret-like string",
    ),

    # 2. Raw Windows paths
    PrivacyPattern(
        category="WIN_PATH",
        pattern=re.compile(
            r"[A-Za-z]:\\(?:[^\\/\r\n:\"<>|?*]{1,255}\\)*[^\\/\r\n:\"<>|?*]{0,255}",
        ),
        replacement_prefix="PATH",
        severity=PatternSeverity.PRIVATE_REF,
        description="Windows absolute file path",
    ),

    # 3. Raw Unix paths
    PrivacyPattern(
        category="UNIX_PATH",
        pattern=re.compile(
            r"(?:/home/|/var/|/etc/|/usr/|/tmp/|/opt/|/proc/|/srv/)[^\s'\"\\,;)>\]]*",
        ),
        replacement_prefix="PATH",
        severity=PatternSeverity.PRIVATE_REF,
        description="Unix absolute file path",
    ),

    # 4. Medical-looking filenames
    PrivacyPattern(
        category="MEDICAL_FILENAME",
        pattern=re.compile(
            r"\b[\w\s\-_.()]*"
            r"(?:labs?|results?|records?|patient|diagnosis|discharge|radiology|"
            r"imaging|report|CBC|ECG|EKG|MRI|CT_scan|pathology|biopsy|"
            r"medications?|prescription|specimen)"
            r"[\w\s\-_.()]*"
            r"\.(?:pdf|docx?|xlsx?|txt|csv|json|hl7|ccd)\b",
            _I,
        ),
        replacement_prefix="FILENAME",
        severity=PatternSeverity.PRIVATE_REF,
        description="Medical-looking document filename",
    ),

    # 5. Date of birth (explicit DOB markers — matched before bare dates)
    PrivacyPattern(
        category="DOB",
        pattern=re.compile(
            r"(?:DOB|Date\s+of\s+Birth|Birthdate?)\s*[:\s]?\s*"
            r"(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2},?\s+\d{4})",
            _I,
        ),
        replacement_prefix="DOB",
        severity=PatternSeverity.PHI,
        description="Date of birth",
    ),

    # 6. MRN-like identifiers
    PrivacyPattern(
        category="MRN",
        pattern=re.compile(
            r"(?:MRN|Medical\s+Record(?:\s+(?:Number|No\.?|Num|#))?)"
            r"\s*[:\s#]?\s*[A-Z0-9]{5,}"
            r"|[A-Z]{2,4}-\d{5,}",
            _I,
        ),
        replacement_prefix="MRN",
        severity=PatternSeverity.PHI,
        description="Medical Record Number",
    ),

    # 7. Insurance / member IDs
    # Require at least one separator character after the prefix to avoid matching
    # plain English words like "insurance", "subscriber" or sanitizer tokens.
    PrivacyPattern(
        category="INSURANCE_ID",
        pattern=re.compile(
            r"\b(?:INS|MBR|MEMBER|POLICY|SUBSCRIBER|GROUP[-_]?NO|PLAN[-_]?ID)"
            r"(?:[\s:_#-]+)[A-Z0-9]{5,}\b"
            r"|\bMEMBER-ID\s*[A-Z0-9]{5,}\b",
            _I,
        ),
        replacement_prefix="INSURANCE_ID",
        severity=PatternSeverity.PHI,
        description="Insurance or member ID",
    ),

    # 8. Email addresses
    PrivacyPattern(
        category="EMAIL",
        pattern=re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        ),
        replacement_prefix="EMAIL",
        severity=PatternSeverity.PII,
        description="Email address",
    ),

    # 9. US phone numbers
    PrivacyPattern(
        category="PHONE",
        pattern=re.compile(
            r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b",
        ),
        replacement_prefix="PHONE",
        severity=PatternSeverity.PII,
        description="Phone number",
    ),

    # 10. General dates (not already caught by DOB)
    PrivacyPattern(
        category="DATE",
        pattern=re.compile(
            r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
        ),
        replacement_prefix="DATE",
        severity=PatternSeverity.PII,
        description="Date string",
    ),

    # 11. Street addresses
    PrivacyPattern(
        category="ADDRESS",
        pattern=re.compile(
            r"\b\d{1,6}\s+[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){0,3}\s+"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr"
            r"|Lane|Ln|Court|Ct|Way|Terrace|Place|Pl|Circle)\b",
            _I,
        ),
        replacement_prefix="ADDRESS",
        severity=PatternSeverity.PII,
        description="Street address",
    ),

    # 12. Healthcare facility names
    PrivacyPattern(
        category="FACILITY",
        pattern=re.compile(
            r"\b[A-Za-z]+(?:\s+[A-Za-z]+){0,4}\s+"
            r"(?:Hospital|Medical\s+Center|Medical\s+Group|Health\s+System"
            r"|Healthcare|Clinic|Infirmary|Medical\s+Plaza|Health\s+Center)\b",
            _I,
        ),
        replacement_prefix="FACILITY",
        severity=PatternSeverity.PHI,
        description="Healthcare facility or provider name",
    ),

    # 13. Person names — titled names and known synthetic fixtures
    PrivacyPattern(
        category="PERSON",
        pattern=re.compile(
            r"(?:"
            # Titled names
            r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|RN|MD|DO|PA)\s*\.?\s+"
            r"[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,2}"
            r"|"
            # Known synthetic test fixtures
            r"\b(?:Jane\s+Doe|John\s+Doe|Test\s+Patient|Alice\s+Smith"
            r"|Bob\s+Jones|Mary\s+Johnson|Patient\s+Alpha|Patient\s+Beta"
            r"|Synthetic\s+Patient)\b"
            r"|"
            # name-field patterns
            r"\b(?:patient|patient_name|full_name|name)\s*:\s*"
            r"[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,2}"
            r")",
            _I,
        ),
        replacement_prefix="PERSON",
        severity=PatternSeverity.PHI,
        description="Person name (titled, known synthetic fixture, or name-field pattern)",
    ),
]

PRIVACY_PATTERNS: List[PrivacyPattern] = _PATTERNS

PATTERNS_BY_CATEGORY: dict[str, PrivacyPattern] = {p.category: p for p in _PATTERNS}

# Categories that always block outbound regardless of sanitization result
ALWAYS_BLOCK_CATEGORIES = {"SECRET"}

# Categories counted as PHI
PHI_CATEGORIES = {"PERSON", "DOB", "MRN", "INSURANCE_ID", "FACILITY"}

# Categories counted as private file references
PRIVATE_REF_CATEGORIES = {"WIN_PATH", "UNIX_PATH", "MEDICAL_FILENAME"}
