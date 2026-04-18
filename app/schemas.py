"""
MedAI Platform v1.1 — All Pydantic Schemas
Single source of truth. Every module imports from here.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field
from uuid import uuid4


# ── Core MKB Record ───────────────────────────────────────────────────────────

class MKBRecord(BaseModel):
    id:                 str           = Field(default_factory=lambda: str(uuid4()))
    fact_type:          str           # diagnosis|medication|test_result|symptom|note|recommendation|relationship|event
    content:            str           # Human-readable fact string
    structured:         dict          = Field(default_factory=dict)
    specialty:          str           = "general"
    source_type:        str           # document|ai_response|manual|web|guideline
    source_name:        str           = ""
    source_url:         Optional[str] = None
    trust_level:        int           = 3  # 1–5
    confidence:         float         = 0.5
    status:             str           = "active"
    tier:               str           = "active"   # active|hypothesis|quarantined|superseded
    ddi_checked:        bool          = False
    ddi_status:         Optional[str] = None       # clear|low|medium|high_blocked|pending
    ddi_findings:       List[dict]    = Field(default_factory=list)
    extraction_method:  str           = "claude"   # claude|rules_based|manual
    resolution_id:      Optional[str] = None
    requires_review:    bool          = False
    first_recorded:     datetime      = Field(default_factory=datetime.utcnow)
    last_confirmed:     datetime      = Field(default_factory=datetime.utcnow)
    linked_to:          List[str]     = Field(default_factory=list)
    chunk_ids:          List[str]     = Field(default_factory=list)
    tags:               List[str]     = Field(default_factory=list)
    session_id:         str           = ""
    promotion_history:  List[dict]    = Field(default_factory=list)


# ── Anonymized Query Payload ──────────────────────────────────────────────────

class AnonymizedPayload(BaseModel):
    query_text:         str
    specialty:          str
    task_type:          str
    context_facts:      List[str]     = Field(default_factory=list)
    active_medications: List[str]     = Field(default_factory=list)
    requires_ddi_check: bool          = False
    session_id:         str           = Field(default_factory=lambda: str(uuid4()))


# ── Connector Response ────────────────────────────────────────────────────────

class ConnectorResponse(BaseModel):
    connector_name:     str
    content:            Optional[str] = None
    confidence:         Optional[float] = None
    citations:          List[str]     = Field(default_factory=list)
    raw_response:       dict          = Field(default_factory=dict)
    latency_ms:         int           = 0
    status:             str           = "ok"  # ok|timeout|error|stub


# ── Scored Response ───────────────────────────────────────────────────────────

class DDIFinding(BaseModel):
    drug_a:             str
    drug_b:             str
    severity:           str           # HIGH|MEDIUM|LOW
    mechanism:          Optional[str] = None
    management:         Optional[str] = None


class ScoredResponse(BaseModel):
    connector_name:     str
    content:            Optional[str] = None
    final_score:        float         = 0.0
    score_breakdown:    dict          = Field(default_factory=dict)
    ddi_findings:       List[DDIFinding] = Field(default_factory=list)
    discarded:          bool          = False
    discard_reason:     Optional[str] = None
    confidence_band:    str           = "discarded"  # high|acceptable|low|discarded


# ── Truth Resolution ──────────────────────────────────────────────────────────

class TruthResolutionInput(BaseModel):
    candidate_fact:     MKBRecord
    existing_fact:      MKBRecord
    conflict_type:      str           # value_conflict|status_conflict|date_conflict|source_conflict


class TruthResolutionOutput(BaseModel):
    resolution:         str           # keep_existing|replace_with_new|merge|quarantine
    winner:             MKBRecord
    loser_id:           str
    confidence:         float
    explanation:        str
    requires_review:    bool          = False
    rule_applied:       str           = ""


# ── Extraction Output ─────────────────────────────────────────────────────────

class ExtractedDiagnosis(BaseModel):
    name:               str
    icd_code:           Optional[str] = None
    date:               Optional[str] = None
    status:             str           = "active"
    notes:              Optional[str] = None


class ExtractedMedication(BaseModel):
    name:               str
    dose:               Optional[str] = None
    frequency:          Optional[str] = None
    route:              Optional[str] = None
    start_date:         Optional[str] = None
    end_date:           Optional[str] = None
    indication:         Optional[str] = None


class ExtractedTestResult(BaseModel):
    test_name:          str
    value:              Optional[str] = None
    unit:               Optional[str] = None
    date:               Optional[str] = None
    reference_range:    Optional[str] = None
    interpretation:     Optional[str] = None


class ExtractedSymptom(BaseModel):
    description:        str
    onset:              Optional[str] = None
    frequency:          Optional[str] = None
    severity:           Optional[str] = None
    triggers:           Optional[str] = None


class ExtractionOutput(BaseModel):
    diagnoses:          List[ExtractedDiagnosis]    = Field(default_factory=list)
    medications:        List[ExtractedMedication]   = Field(default_factory=list)
    test_results:       List[ExtractedTestResult]   = Field(default_factory=list)
    symptoms:           List[ExtractedSymptom]      = Field(default_factory=list)
    notes:              List[str]                   = Field(default_factory=list)
    recommendations:    List[str]                   = Field(default_factory=list)
    extraction_method:  str                         = "claude"
    confidence:         float                       = 0.5


# ── Decision Engine Query ─────────────────────────────────────────────────────

class ClassifiedQuery(BaseModel):
    original_query:     str
    specialty:          str
    task_type:          str
    confidence:         float
    requires_ddi_check: bool          = False
    session_id:         str           = Field(default_factory=lambda: str(uuid4()))


class MKBContext(BaseModel):
    structured_facts:   List[MKBRecord]  = Field(default_factory=list)
    semantic_chunks:    List[str]        = Field(default_factory=list)
    active_medications: List[MKBRecord]  = Field(default_factory=list)
    active_diagnoses:   List[MKBRecord]  = Field(default_factory=list)
    recent_conflicts:   List[MKBRecord]  = Field(default_factory=list)


# ── Ledger Event ──────────────────────────────────────────────────────────────

class LedgerEvent(BaseModel):
    id:                 Optional[int] = None
    event_type:         str
    record_id:          Optional[str] = None
    source_type:        Optional[str] = None
    previous_value:     Optional[dict] = None
    details:            dict          = Field(default_factory=dict)
    timestamp:          datetime      = Field(default_factory=datetime.utcnow)
    session_id:         str           = ""


# ── System State ──────────────────────────────────────────────────────────────

class SystemState(BaseModel):
    claude_available:   bool          = True
    safe_mode:          bool          = False
    safe_mode_reason:   Optional[str] = None
    active_connectors:  List[str]     = Field(default_factory=list)
    pending_enrichment: int           = 0


# ── Final Response ────────────────────────────────────────────────────────────

class UnifiedResponse(BaseModel):
    query:              str
    specialty:          str
    synthesis:          str
    confidence:         float
    confidence_band:    str
    sources_used:       List[str]     = Field(default_factory=list)
    mkb_facts_used:     List[MKBRecord] = Field(default_factory=list)
    hypothesis_facts:   List[MKBRecord] = Field(default_factory=list)
    ddi_findings:       List[DDIFinding] = Field(default_factory=list)
    safe_mode:          bool          = False
    discarded_responses: List[str]    = Field(default_factory=list)
    session_id:         str           = ""
    timestamp:          datetime      = Field(default_factory=datetime.utcnow)
