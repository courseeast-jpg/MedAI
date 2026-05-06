"""Data models for CKA-B07 Medical Coding / SNOMED-UMLS Interface.

No real medical data. No real patient data. No real external API calls.
Synthetic coding stubs only unless explicit local lookup file provided.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

_SALT = "medai_cka_b07_models_v1"


class CodingSystem(str, Enum):
    SYNTHETIC = "synthetic"
    UMLS = "umls"
    SNOMED_CT = "snomed_ct"
    RXNORM = "rxnorm"
    LOINC = "loinc"
    UNKNOWN = "unknown"


class CodingStatus(str, Enum):
    CODED = "coded"
    UNMAPPED = "unmapped"
    AMBIGUOUS = "ambiguous"
    CODING_UNAVAILABLE = "coding_unavailable"
    INVALID_CODE = "invalid_code"
    SOURCE_UNAVAILABLE = "source_unavailable"


class TerminologySourceStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    STUB_ONLY = "stub_only"
    LOCAL_LOOKUP_ONLY = "local_lookup_only"


def _safe_code_id(system: str, code: str) -> str:
    digest = hashlib.sha256(f"{_SALT}:{system}:{code}".encode()).hexdigest()[:12]
    return f"cka_code_{digest}"


@dataclass
class MedicalCode:
    system: CodingSystem
    code: str
    display: str
    version: str = ""
    source: str = "synthetic_stub"   # never a raw private path
    synthetic: bool = True
    confidence: float = 1.0
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.safe_public_summary:
            sys_val = self.system.value if hasattr(self.system, "value") else str(self.system)
            self.safe_public_summary = {
                "safe_code_id": _safe_code_id(sys_val, self.code),
                "system": sys_val,
                "synthetic": self.synthetic,
                "confidence": self.confidence,
            }


@dataclass
class CodingCandidate:
    candidate_id: str
    safe_candidate_id: str          # hashed — never raw in public
    fact_type: str
    entity_text: str
    normalized_text: str
    specialty: str = "general"
    structured: Dict[str, Any] = field(default_factory=dict)
    source_record_id: str = ""      # internal — not exposed publicly
    source_tier: str = "unknown"
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.safe_public_summary:
            self.safe_public_summary = {
                "safe_candidate_id": self.safe_candidate_id,
                "fact_type": self.fact_type,
                "specialty": self.specialty,
                "source_tier": self.source_tier,
                "synthetic": True,
            }


@dataclass
class CodingResult:
    candidate_safe_id: str
    status: CodingStatus
    codes: List[MedicalCode]
    preferred_code: Optional[MedicalCode]
    ambiguity_count: int
    confidence: float
    terminology_source_status: TerminologySourceStatus
    explanation: str
    no_code_hallucinated: bool = True
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.safe_public_summary:
            self.safe_public_summary = {
                "candidate_safe_id": self.candidate_safe_id,
                "status": self.status.value if hasattr(self.status, "value") else self.status,
                "code_count": len(self.codes),
                "ambiguity_count": self.ambiguity_count,
                "confidence": self.confidence,
                "no_code_hallucinated": self.no_code_hallucinated,
                "preferred_code": (
                    self.preferred_code.safe_public_summary
                    if self.preferred_code else None
                ),
            }


@dataclass
class CodingValidationResult:
    valid: bool
    status: CodingStatus
    invalid_reasons: List[str]
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.safe_public_summary:
            self.safe_public_summary = {
                "valid": self.valid,
                "status": self.status.value if hasattr(self.status, "value") else self.status,
                "invalid_reason_count": len(self.invalid_reasons),
            }
