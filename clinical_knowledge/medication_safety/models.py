"""Data models for CKA-B05 Medication Safety / DDI Gate."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DDISeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNAVAILABLE = "unavailable"


class DDICheckStatus(str, Enum):
    CLEAR = "clear"
    LOW = "low"
    MEDIUM = "medium"
    HIGH_BLOCKED = "high_blocked"
    PENDING = "pending"
    UNAVAILABLE = "unavailable"


class MedicationSafetyAction(str, Enum):
    ALLOW = "allow"
    ALLOW_WITH_NOTE = "allow_with_note"
    WARN_REQUIRES_ACK = "warn_requires_ack"
    BLOCK_REQUIRES_CONFIRMATION = "block_requires_confirmation"
    QUEUE_PENDING_DDI = "queue_pending_ddi"


@dataclass(frozen=True)
class DDIFinding:
    drug_a: str
    drug_b: str
    severity: DDISeverity
    mechanism: str          # generic synthetic text only — no medical advice
    management_note: str    # generic synthetic text only — no prescribing advice
    source: str
    synthetic: bool = True
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DDICheckResult:
    checked: bool
    available: bool
    findings: List[DDIFinding]
    highest_severity: DDISeverity
    status: DDICheckStatus
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MedicationWriteGateResult:
    action: MedicationSafetyAction
    allowed_to_write: bool
    requires_user_confirmation: bool
    candidate_status: str
    ddi_checked: bool
    ddi_status: DDICheckStatus
    ddi_findings: List[DDIFinding]
    ledger_event_ready: bool
    explanation: str
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Layer1DDIScoreResult:
    base_score: float
    adjusted_score: float
    penalty_applied: float
    highest_severity: DDISeverity
    findings: List[DDIFinding]
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)
