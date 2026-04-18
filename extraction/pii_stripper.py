"""
MedAI v1.1 — PII Stripper (Track B)
Uses Microsoft Presidio for HIPAA-compliant PII removal.
Falls back to regex patterns if Presidio unavailable.
CRITICAL: Nothing with PII leaves the local machine.
"""
import re
from typing import Tuple
from loguru import logger

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
    logger.info("Presidio PII engine loaded")
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.warning("Presidio not available — using regex fallback for PII stripping")


# Entities Presidio will detect and replace
PRESIDIO_ENTITIES = [
    "PERSON", "LOCATION", "DATE_TIME", "PHONE_NUMBER",
    "EMAIL_ADDRESS", "URL", "US_SSN", "MEDICAL_LICENSE",
    "NRP",  # National Registration/ID
]

# Operator config: replace each entity type with a safe placeholder
OPERATORS = {
    "PERSON":          OperatorConfig("replace", {"new_value": "[PERSON]"}),
    "LOCATION":        OperatorConfig("replace", {"new_value": "[LOCATION]"}),
    "DATE_TIME":       OperatorConfig("replace", {"new_value": "[DATE]"}),
    "PHONE_NUMBER":    OperatorConfig("replace", {"new_value": "[CONTACT_REMOVED]"}),
    "EMAIL_ADDRESS":   OperatorConfig("replace", {"new_value": "[CONTACT_REMOVED]"}),
    "URL":             OperatorConfig("replace", {"new_value": "[URL_REMOVED]"}),
    "US_SSN":          OperatorConfig("replace", {"new_value": "[ID_REMOVED]"}),
    "MEDICAL_LICENSE": OperatorConfig("replace", {"new_value": "[ID_REMOVED]"}),
    "NRP":             OperatorConfig("replace", {"new_value": "[ID_REMOVED]"}),
} if PRESIDIO_AVAILABLE else {}

# Regex fallback patterns
REGEX_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b', '[ID_REMOVED]'),           # SSN
    (r'\b\d{10}\b', '[ID_REMOVED]'),                        # NPI
    (r'\b[A-Z]{2}\d{6}\b', '[ID_REMOVED]'),                 # Medical license
    (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[DATE]'),            # MM/DD/YYYY
    (r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]'),                   # ISO date
    (r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', '[CONTACT_REMOVED]'),  # Phone
    (r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '[CONTACT_REMOVED]'),  # Email
    (r'\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[PHYSICIAN]'),
]


class PIIStripper:
    def __init__(self):
        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            self.analyzer = None
            self.anonymizer = None

    def strip(self, text: str) -> Tuple[str, str]:
        """
        Returns (stripped_text, method_used)
        method_used: 'presidio' | 'regex'
        """
        if not text:
            return text, "none"

        if PRESIDIO_AVAILABLE:
            return self._strip_presidio(text), "presidio"
        else:
            return self._strip_regex(text), "regex"

    def _strip_presidio(self, text: str) -> str:
        try:
            results = self.analyzer.analyze(text=text, entities=PRESIDIO_ENTITIES, language="en")
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=OPERATORS
            )
            return anonymized.text
        except Exception as e:
            logger.warning(f"Presidio failed: {e}. Falling back to regex.")
            return self._strip_regex(text)

    def _strip_regex(self, text: str) -> str:
        result = text
        for pattern, replacement in REGEX_PATTERNS:
            result = re.sub(pattern, replacement, result)
        return result

    def verify_clean(self, text: str) -> Tuple[bool, list]:
        """
        Audit check: confirm no obvious PII remains.
        Returns (is_clean, findings[])
        """
        findings = []
        for pattern, label in REGEX_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                findings.append(f"Possible PII ({label}): {matches[:2]}")
        return len(findings) == 0, findings
