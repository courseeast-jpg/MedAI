"""MEDAI-V2-RUNTIME-CONTRACTS-01 — V2 runtime interface contracts.

Typing-only package. No runtime side effects on import. No concrete
adapter implementations. Standard-library imports only.

Importing this package must not:
    * call any external API,
    * open any file,
    * read any environment variable,
    * touch any DB,
    * import any project runtime module,
    * import Streamlit,
    * print anything,
    * mutate any global state.
"""
from __future__ import annotations

from clinical_knowledge.v2_contracts.runtime_contracts import (  # noqa: F401
    # Section 1 — Ingestion boundary
    V2SourceKind,
    V2DocumentSource,
    V2IngestionRequest,
    V2IngestionAdapterProtocol,
    # Section 2 — Text visibility / document quality
    V2VisibilityStatus,
    V2TextVisibilityProfile,
    V2DocumentQualityProtocol,
    # Section 3 — Extraction / OCR orchestration
    V2ExtractionMode,
    V2ExtractionRequest,
    V2ExtractionResult,
    V2ExtractionAdapterProtocol,
    # Section 4 — Document classification
    V2DocumentTypeCandidate,
    V2ClassificationResult,
    V2ClassifierProtocol,
    # Section 5 — Clinical knowledge / terminology
    V2TerminologyQuery,
    V2TerminologyMatchSummary,
    V2TerminologyLookupProtocol,
    # Section 6 — Review / HITL
    V2ReviewDisposition,
    V2ReviewItem,
    V2ReviewQueueProtocol,
    # Section 7 — Operator action
    V2OperatorActionKind,
    V2OperatorActionRequest,
    V2OperatorActionResult,
    V2OperatorActionProtocol,
    # Section 8 — Audit / observability
    V2AuditEventKind,
    V2AuditEvent,
    V2ObservabilitySinkProtocol,
    # Section 9 — Validation report
    V2ValidationStatus,
    V2ValidationReceipt,
    V2ValidationHarnessProtocol,
    # Section 10 — Runtime privacy / safety
    V2RuntimeSafetyProfile,
    V2PrivacyGateProtocol,
)

__all__ = (
    "V2SourceKind",
    "V2DocumentSource",
    "V2IngestionRequest",
    "V2IngestionAdapterProtocol",
    "V2VisibilityStatus",
    "V2TextVisibilityProfile",
    "V2DocumentQualityProtocol",
    "V2ExtractionMode",
    "V2ExtractionRequest",
    "V2ExtractionResult",
    "V2ExtractionAdapterProtocol",
    "V2DocumentTypeCandidate",
    "V2ClassificationResult",
    "V2ClassifierProtocol",
    "V2TerminologyQuery",
    "V2TerminologyMatchSummary",
    "V2TerminologyLookupProtocol",
    "V2ReviewDisposition",
    "V2ReviewItem",
    "V2ReviewQueueProtocol",
    "V2OperatorActionKind",
    "V2OperatorActionRequest",
    "V2OperatorActionResult",
    "V2OperatorActionProtocol",
    "V2AuditEventKind",
    "V2AuditEvent",
    "V2ObservabilitySinkProtocol",
    "V2ValidationStatus",
    "V2ValidationReceipt",
    "V2ValidationHarnessProtocol",
    "V2RuntimeSafetyProfile",
    "V2PrivacyGateProtocol",
)
