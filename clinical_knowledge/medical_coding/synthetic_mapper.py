"""Deterministic synthetic terminology mapper for CKA-B07.

Maps clearly-synthetic entity names to synthetic codes only.
Codes are NOT real SNOMED/UMLS/RxNorm/LOINC values.
Every code: synthetic=True, no_code_hallucinated=True.
Unknown entities return status=unmapped with codes=[].
"""
from __future__ import annotations

import re
from typing import Optional

from clinical_knowledge.medical_coding.models import (
    CodingResult,
    CodingStatus,
    CodingSystem,
    MedicalCode,
    TerminologySourceStatus,
)
from clinical_knowledge.medical_coding.terminology_source import TerminologySource

# Synthetic mapping table — clearly synthetic codes, no real clinical values
_SYNTHETIC_TABLE: dict[str, dict] = {
    "synthetic condition alpha": {
        "code": "SYN-DX-001",
        "display": "Synthetic Condition Alpha (test only)",
        "confidence": 1.0,
    },
    "synthetic condition beta": {
        "code": "SYN-DX-002",
        "display": "Synthetic Condition Beta (test only)",
        "confidence": 1.0,
    },
    "synthetic medication beta": {
        "code": "SYN-RX-001",
        "display": "Synthetic Medication Beta (test only)",
        "confidence": 1.0,
    },
    "synthetic lab gamma": {
        "code": "SYN-LAB-001",
        "display": "Synthetic Lab Gamma (test only)",
        "confidence": 1.0,
    },
    "synthetic procedure delta": {
        "code": "SYN-PROC-001",
        "display": "Synthetic Procedure Delta (test only)",
        "confidence": 1.0,
    },
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


class SyntheticTerminologySource(TerminologySource):
    """Deterministic synthetic-only terminology source.

    All codes are synthetic test codes. No real clinical assertions made.
    """

    @property
    def name(self) -> str:
        return "synthetic_stub"

    def status(self) -> TerminologySourceStatus:
        return TerminologySourceStatus.STUB_ONLY

    def lookup(
        self,
        normalized_text: str,
        fact_type: Optional[str] = None,
        specialty: Optional[str] = None,
    ) -> CodingResult:
        key = _normalize(normalized_text)
        entry = _SYNTHETIC_TABLE.get(key)

        if entry is None:
            return CodingResult(
                candidate_safe_id="",
                status=CodingStatus.UNMAPPED,
                codes=[],
                preferred_code=None,
                ambiguity_count=0,
                confidence=0.0,
                terminology_source_status=TerminologySourceStatus.STUB_ONLY,
                explanation="Synthetic mapper: no entry for normalized text.",
                no_code_hallucinated=True,
            )

        code = MedicalCode(
            system=CodingSystem.SYNTHETIC,
            code=entry["code"],
            display=entry["display"],
            version="synthetic-v1",
            source="synthetic_stub",
            synthetic=True,
            confidence=entry["confidence"],
        )
        return CodingResult(
            candidate_safe_id="",
            status=CodingStatus.CODED,
            codes=[code],
            preferred_code=code,
            ambiguity_count=0,
            confidence=entry["confidence"],
            terminology_source_status=TerminologySourceStatus.STUB_ONLY,
            explanation="Synthetic mapper: deterministic match found.",
            no_code_hallucinated=True,
        )
