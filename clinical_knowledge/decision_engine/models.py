"""Data models for CKA-B03 Decision Engine."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QueryTaskType(str, Enum):
    MEDICATION = "medication"
    DIAGNOSIS = "diagnosis"
    SUMMARY = "summary"
    DOCUMENT = "document"
    GENERAL = "general"


class QuerySpecialty(str, Enum):
    EPILEPSY = "epilepsy"
    NEUROLOGY = "neurology"
    UNKNOWN = "unknown"


class ScoreBand(str, Enum):
    DISCARDED = "discarded"    # < 0.30
    LOW = "low"                # 0.30 – 0.49
    ACCEPTABLE = "acceptable"  # 0.50 – 0.74
    HIGH = "high"              # >= 0.75


@dataclass(frozen=True)
class QueryClassification:
    raw_query_hash: str            # SHA-256 of original query — never raw text
    specialty: QuerySpecialty
    task_type: QueryTaskType
    confidence: float              # 0.0 – 1.0
    requires_ddi_check: bool
    medication_terms_detected: List[str]
    clarification_required: bool
    refusal_reason: Optional[str]  # set if query must be refused outright


@dataclass(frozen=True)
class DecisionContext:
    query_hash: str
    mkb_records_found: int
    mkb_snippets: List[str]       # safe, redacted snippets only
    context_tiers: List[str]      # tiers of matched records


@dataclass(frozen=True)
class ConnectorRequest:
    connector_id: str
    query_hash: str
    specialty: str
    task_type: str
    privacy_cleared: bool


@dataclass(frozen=True)
class ConnectorResponse:
    connector_id: str
    success: bool
    content: str                  # stub text (no real API data)
    confidence: float
    citations: List[str]
    error: Optional[str] = None


@dataclass
class ScoredResponse:
    connector_id: str
    raw_content: str
    mkb_consistency_score: float      # weight 0.35
    internal_coherence_score: float   # weight 0.25
    citation_presence_score: float    # weight 0.20
    ddi_safety_score: float           # weight 0.20
    composite_score: float
    score_band: ScoreBand
    discarded: bool

    @staticmethod
    def compute_composite(
        mkb: float, coherence: float, citation: float, ddi: float
    ) -> float:
        return round(
            mkb * 0.35 + coherence * 0.25 + citation * 0.20 + ddi * 0.20, 4
        )


@dataclass
class SafeModeState:
    active: bool
    reason: str                       # human-readable reason
    prefix: str = "[SAFE MODE — MKB only, no external AI]"
    triggered_by_connector_failure: bool = False
    triggered_by_low_confidence: bool = False
    triggered_by_manual_flag: bool = False


@dataclass
class DecisionEngineResult:
    query_hash: str
    classification: QueryClassification
    context: DecisionContext
    connector_responses: List[ConnectorResponse]
    scored_responses: List[ScoredResponse]
    safe_mode: SafeModeState
    final_response: str
    refused: bool
    refusal_reason: Optional[str]
    external_api_used: bool = False   # always False in B03
    ddi_layer1_checked: bool = False
    raw_phi_in_query: bool = False
    phi_sanitized_before_connectors: bool = False
    ledger_events_written: int = 0
