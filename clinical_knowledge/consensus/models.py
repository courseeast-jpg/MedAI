"""Consensus models and enums for CKA-B08."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

_SALT = "medai_cka_b08_consensus_v1"


def _safe_fact_id(fact_key: str) -> str:
    digest = hashlib.sha256(f"{_SALT}:fact:{fact_key}".encode()).hexdigest()[:16]
    return f"cka_cf_{digest}"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConsensusStatus(str, Enum):
    CONSENSUS_READY = "consensus_ready"
    INSUFFICIENT_RESPONSES = "insufficient_responses"
    CONTRADICTION_DETECTED = "contradiction_detected"
    TRUTH_RESOLUTION_REQUIRED = "truth_resolution_required"
    NO_CONSENSUS = "no_consensus"
    ALL_RESPONSES_DISCARDED = "all_responses_discarded"


class ConsensusFactStatus(str, Enum):
    AGREED = "agreed"
    SINGLE_SOURCE_PENALIZED = "single_source_penalized"
    CONTRADICTED = "contradicted"
    DISCARDED = "discarded"
    ROUTED_TRUTH_RESOLUTION = "routed_truth_resolution"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConsensusFact:
    fact_id: str                          # internal key (specialty:fact_type:entity_text)
    safe_fact_id: str                     # hash-based public ID
    fact_type: str
    entity_text: str
    structured: Dict[str, Any]
    specialty: str
    supporting_connectors: List[str]      # connector names
    contradicting_connectors: List[str]
    agreement_ratio: float                # supporting / total_successful
    confidence: float
    status: ConsensusFactStatus

    @property
    def safe_public_summary(self) -> Dict[str, Any]:
        return {
            "safe_fact_id": self.safe_fact_id,
            "fact_type": self.fact_type,
            "entity_text": self.entity_text,
            "specialty": self.specialty,
            "supporting_connector_count": len(self.supporting_connectors),
            "contradicting_connector_count": len(self.contradicting_connectors),
            "agreement_ratio": round(self.agreement_ratio, 4),
            "confidence": round(self.confidence, 4),
            "status": self.status.value,
        }


@dataclass
class ConsensusContradiction:
    fact_type: str
    entity_text: str
    specialty: str
    conflicting_structured_values: List[Dict[str, Any]]
    connector_names: List[str]
    safe_ids: List[str]
    is_medication_dose_conflict: bool = False

    @property
    def safe_public_summary(self) -> Dict[str, Any]:
        return {
            "fact_type": self.fact_type,
            "entity_text": self.entity_text,
            "specialty": self.specialty,
            "connector_count": len(self.connector_names),
            "is_medication_dose_conflict": self.is_medication_dose_conflict,
            "conflicting_value_count": len(self.conflicting_structured_values),
        }


@dataclass
class ConsensusResult:
    status: ConsensusStatus
    consensus_facts: List[ConsensusFact]
    contradictions: List[ConsensusContradiction]
    confidence_aggregate: float
    discarded_response_count: int
    truth_resolution_results: List[Dict[str, Any]]   # safe summaries only
    escalation_required: bool

    @property
    def safe_public_summary(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "consensus_fact_count": len(self.consensus_facts),
            "contradiction_count": len(self.contradictions),
            "confidence_aggregate": round(self.confidence_aggregate, 4),
            "discarded_response_count": self.discarded_response_count,
            "truth_resolution_count": len(self.truth_resolution_results),
            "escalation_required": self.escalation_required,
            "consensus_facts": [f.safe_public_summary for f in self.consensus_facts],
            "contradictions": [c.safe_public_summary for c in self.contradictions],
        }
