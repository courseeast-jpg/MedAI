"""Connector models and enums for CKA-B08.

All models use safe IDs/hashes in public summaries.
external_api_used is always False in B08.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

_SALT = "medai_cka_b08_connector_v1"


def _hash(value: str) -> str:
    return hashlib.sha256(f"{_SALT}:{value}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConnectorKind(str, Enum):
    DXGPT_STUB = "dxgpt_stub"
    SAGE_EPILEPSY_STUB = "sage_epilepsy_stub"
    PATIENTNOTES_DDI_STUB = "patientnotes_ddi_stub"
    GENERIC_STUB = "generic_stub"


class ConnectorStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    BLOCKED_PRIVACY = "blocked_privacy"
    SKIPPED_SAFE_MODE = "skipped_safe_mode"
    DISABLED = "disabled"
    MALFORMED_RESPONSE = "malformed_response"


class ConnectorCapability(str, Enum):
    DIAGNOSIS_SUPPORT = "diagnosis_support"
    MEDICATION_SAFETY = "medication_safety"
    EPILEPSY_SUPPORT = "epilepsy_support"
    CITATION_SUPPORT = "citation_support"
    STRUCTURED_FACT_OUTPUT = "structured_fact_output"


class SimulationMode(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    MALFORMED_RESPONSE = "malformed_response"
    LOW_CONFIDENCE = "low_confidence"
    CONTRADICTION = "contradiction"
    PRIVACY_BLOCKED = "privacy_blocked"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConnectorSpec:
    name: str
    kind: ConnectorKind
    enabled: bool
    capabilities: List[ConnectorCapability]
    timeout_seconds: float = 5.0
    allow_external: bool = False      # always False in B08
    synthetic_only: bool = True       # always True in B08

    @property
    def safe_public_summary(self) -> Dict[str, Any]:
        return {
            "name_hash": _hash(self.name),
            "kind": self.kind.value,
            "enabled": self.enabled,
            "capabilities": [c.value for c in self.capabilities],
            "allow_external": self.allow_external,
            "synthetic_only": self.synthetic_only,
        }


@dataclass
class ConnectorExecutionRequest:
    connector_name: str
    query_hash: str
    sanitized_payload: Dict[str, Any]
    payload_hash: str
    purpose: str
    allow_external: bool = False

    @property
    def safe_public_summary(self) -> Dict[str, Any]:
        return {
            "connector_name_hash": _hash(self.connector_name),
            "query_hash": self.query_hash,
            "payload_hash": self.payload_hash,
            "purpose": self.purpose,
            "allow_external": self.allow_external,
        }


@dataclass
class ConnectorExecutionResult:
    connector_name: str
    status: ConnectorStatus
    normalized_response: Optional[Dict[str, Any]]
    latency_ms: float
    error_reason: Optional[str]
    external_api_used: bool = False   # always False in B08
    privacy_audit_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def safe_public_summary(self) -> Dict[str, Any]:
        return {
            "connector_name_hash": _hash(self.connector_name),
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "external_api_used": self.external_api_used,
            "has_normalized_response": self.normalized_response is not None,
            "error_reason": self.error_reason,
        }
