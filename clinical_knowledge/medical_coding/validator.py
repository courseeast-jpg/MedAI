"""Coding validator for CKA-B07.

validate_code(code: MedicalCode) -> CodingValidationResult

Rules:
- synthetic system requires synthetic=True.
- synthetic=False requires source in allowed verified local source list.
- empty code is invalid.
- unknown system is invalid.
- confidence must be 0.0-1.0.
- no real-code validity claim unless source is verified local lookup.
- UMLS/SNOMED/RxNorm/LOINC external validation is NOT implemented in this block.
"""
from __future__ import annotations

from typing import List

from clinical_knowledge.medical_coding.models import (
    CodingStatus,
    CodingSystem,
    CodingValidationResult,
    MedicalCode,
)

# Sources trusted as verified local (hash-prefix pattern)
_VERIFIED_LOCAL_SOURCE_PREFIXES = ("synthetic_stub", "local_lookup:")


def _is_verified_local_source(source: str) -> bool:
    return any(source.startswith(p) for p in _VERIFIED_LOCAL_SOURCE_PREFIXES)


def validate_code(code: MedicalCode) -> CodingValidationResult:
    """Validate a MedicalCode according to B07 rules.

    Does NOT call external APIs. Does NOT validate against real UMLS/SNOMED.
    """
    reasons: List[str] = []

    # Empty code
    if not code.code or not code.code.strip():
        reasons.append("code string is empty")

    # Unknown system is invalid
    sys_val = code.system.value if hasattr(code.system, "value") else str(code.system)
    if sys_val == CodingSystem.UNKNOWN.value:
        reasons.append("system is 'unknown' which is not a valid coding system")

    # Synthetic system requires synthetic=True
    if sys_val == CodingSystem.SYNTHETIC.value and not code.synthetic:
        reasons.append("system=synthetic requires synthetic=True")

    # Non-synthetic code must come from a verified local source
    if not code.synthetic and not _is_verified_local_source(code.source):
        reasons.append(
            "synthetic=False requires source to be a verified local lookup "
            "(external terminology validation not implemented in B07)"
        )

    # Confidence range
    if not (0.0 <= code.confidence <= 1.0):
        reasons.append(f"confidence {code.confidence} is outside [0, 1]")

    valid = len(reasons) == 0
    status = CodingStatus.CODED if valid else CodingStatus.INVALID_CODE

    return CodingValidationResult(
        valid=valid,
        status=status,
        invalid_reasons=reasons,
    )
