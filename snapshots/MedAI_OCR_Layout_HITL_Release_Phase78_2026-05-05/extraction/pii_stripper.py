"""
MedAI v1.1 - PII stripper (Track B).
Uses Microsoft Presidio for HIPAA-compliant PII removal.
Falls back to regex patterns if Presidio is unavailable.
CRITICAL: Nothing with PII leaves the local machine.
"""

from __future__ import annotations

import re
from typing import Any, Tuple

from loguru import logger

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    PRESIDIO_AVAILABLE = True
    logger.info("Presidio PII engine loaded")
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.warning("Presidio not available - using regex fallback for PII stripping")


PRESIDIO_ENTITIES = [
    "PERSON",
    "LOCATION",
    "DATE_TIME",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "URL",
    "US_SSN",
    "MEDICAL_LICENSE",
    "NRP",
]

OPERATORS = {
    "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
    "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
    "DATE_TIME": OperatorConfig("replace", {"new_value": "[DATE]"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[CONTACT_REMOVED]"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[CONTACT_REMOVED]"}),
    "URL": OperatorConfig("replace", {"new_value": "[URL_REMOVED]"}),
    "US_SSN": OperatorConfig("replace", {"new_value": "[ID_REMOVED]"}),
    "MEDICAL_LICENSE": OperatorConfig("replace", {"new_value": "[ID_REMOVED]"}),
    "NRP": OperatorConfig("replace", {"new_value": "[ID_REMOVED]"}),
} if PRESIDIO_AVAILABLE else {}

REGEX_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[ID_REMOVED]"),
    (r"\b\d{10}\b", "[ID_REMOVED]"),
    (r"\b[A-Z]{2}\d{6}\b", "[ID_REMOVED]"),
    (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]"),
    (r"\b\d{4}-\d{2}-\d{2}\b", "[DATE]"),
    (r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[CONTACT_REMOVED]"),
    (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "[CONTACT_REMOVED]"),
    (r"\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b", "[PHYSICIAN]"),
]

MEDICAL_LABEL_ALLOWLIST = [
    "UA Blood",
    "UA RBC",
    "UA Crystals",
    "UA Calcium Oxalate Crystals",
    "UA Leukocytes",
    "UA Nitrite",
    "UA Protein",
    "UA Glucose",
    "UA Ketones",
    "UA Bilirubin",
    "UA Urobilinogen",
    "UA Specific Gravity",
    "UA pH",
    "RBC UA",
    "Color UA",
    "Glucose UA",
    "Nitrite UA",
]
MEDICAL_LABEL_PATTERNS = [
    re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    for term in sorted(MEDICAL_LABEL_ALLOWLIST, key=len, reverse=True)
]


class PIIStripper:
    def __init__(self):
        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            self.analyzer = None
            self.anonymizer = None
        self.last_audit = self._build_default_audit()

    def strip(self, text: str, doc_type: str = "clinical") -> Tuple[str, str]:
        """
        Returns (stripped_text, method_used).

        doc_type controls how aggressively PII is detected:
        - "clinical": full Presidio NER (PERSON, LOCATION, dates, IDs ...)
        - anything else: regex-only path for reference documents to avoid
          NER false positives on non-clinical content.
        """
        self.last_audit = self._build_default_audit()
        if not text:
            return text, "none"

        protected_text, restore_map, preserved_terms = self._protect_medical_labels(text)

        if doc_type != "clinical":
            stripped = self._strip_regex(protected_text)
            restored = self._restore_medical_labels(stripped, restore_map)
            self._record_preserved_terms(preserved_terms)
            return restored, "regex"

        if PRESIDIO_AVAILABLE:
            return self._strip_presidio(protected_text, restore_map, preserved_terms), "presidio"

        stripped = self._strip_regex(protected_text)
        restored = self._restore_medical_labels(stripped, restore_map)
        self._record_preserved_terms(preserved_terms)
        return restored, "regex"

    def _strip_presidio(
        self,
        text: str,
        restore_map: dict[str, str],
        preserved_terms: list[str],
    ) -> str:
        try:
            results = self.analyzer.analyze(text=text, entities=PRESIDIO_ENTITIES, language="en")
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=OPERATORS,
            )
            stripped = self._strip_regex(anonymized.text)
            restored = self._restore_medical_labels(stripped, restore_map)
            self._record_preserved_terms(preserved_terms)
            return restored
        except Exception as exc:
            logger.warning(f"Presidio failed: {exc}. Falling back to regex.")
            stripped = self._strip_regex(text)
            restored = self._restore_medical_labels(stripped, restore_map)
            self._record_preserved_terms(preserved_terms)
            return restored

    def _strip_regex(self, text: str) -> str:
        result = text
        for pattern, replacement in REGEX_PATTERNS:
            result = re.sub(pattern, replacement, result)
        return result

    def verify_clean(self, text: str) -> Tuple[bool, list]:
        findings = []
        for pattern, label in REGEX_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                findings.append(f"Possible PII ({label}): {matches[:2]}")
        return len(findings) == 0, findings

    def _protect_medical_labels(self, text: str) -> tuple[str, dict[str, str], list[str]]:
        protected_text = text
        restore_map: dict[str, str] = {}
        preserved_terms: list[str] = []
        placeholder_index = 0

        for pattern in MEDICAL_LABEL_PATTERNS:
            def replacer(match: re.Match[str]) -> str:
                nonlocal placeholder_index
                matched_text = match.group(0)
                canonical = self._canonicalize_medical_label(matched_text)
                placeholder = f"__MEDICAL_LABEL_{placeholder_index}__"
                placeholder_index += 1
                restore_map[placeholder] = canonical
                preserved_terms.append(canonical)
                return placeholder

            protected_text = pattern.sub(replacer, protected_text)

        return protected_text, restore_map, sorted(set(preserved_terms))

    def _restore_medical_labels(self, text: str, restore_map: dict[str, str]) -> str:
        restored = text
        for placeholder, original in restore_map.items():
            restored = restored.replace(placeholder, original)
        return restored

    def _record_preserved_terms(self, preserved_terms: list[str]) -> None:
        unique_terms = sorted(set(preserved_terms))
        self.last_audit = {
            "pii_medical_label_preserved_count": len(unique_terms),
            "pii_medical_label_preserved_terms": unique_terms,
        }

    def _build_default_audit(self) -> dict[str, Any]:
        return {
            "pii_medical_label_preserved_count": 0,
            "pii_medical_label_preserved_terms": [],
        }

    def _canonicalize_medical_label(self, value: str) -> str:
        normalized = re.sub(r"\s+", " ", value.strip())
        lowered = normalized.lower()
        for term in MEDICAL_LABEL_ALLOWLIST:
            if lowered == term.lower():
                return term
        return normalized
