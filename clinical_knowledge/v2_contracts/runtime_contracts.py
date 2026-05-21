"""MEDAI-V2-RUNTIME-CONTRACTS-01 — Typing-only V2 runtime interface
contracts.

This module is the typing-only foundation for MedAI v2 runtime
boundaries. It defines `Protocol` interfaces, frozen `dataclass` DTOs,
and `Enum` value sets covering ten v2 boundaries: ingestion, document
quality, extraction / OCR orchestration, document classification,
clinical-knowledge / terminology lookup, review / HITL queue, operator
action, audit / observability, validation harness, and runtime privacy
gate.

This module:

* uses **only** the standard library (`dataclasses`, `enum`, `typing`,
  `datetime`);
* has **no runtime side effects on import** — no IO, no environment
  reads, no DB, no Streamlit, no project runtime imports, no external
  packages;
* defines **no concrete adapters**; every adapter shape is a
  `Protocol`;
* defaults the `V2RuntimeSafetyProfile` to local-only, review-bound,
  external-API-blocked, no-auto-accept.

This module is **not** imported from any production runtime path. The
V2 foundation doctrine (`MEDAI-V2-FOUNDATION-SPEC-02`) and the V2
architecture spec (`MEDAI-V2-ARCHITECTURE-SPEC-01`) gate every future
implementation that satisfies these contracts.

Notes:

* The terminology contract returns aggregate-only metadata. It must
  not expose licensed row content, codes, displays, synonyms, or
  definitions. The license-gated private adapter track remains parked
  at PARK-02 (`b9b19ad`).
* Cue expansion remains explicitly NOT recommended.
* No `from __future__ import annotations` consumer of this module is
  required to call any runtime adapter; the contracts are typing
  scaffolding only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)


# ── Type aliases ──────────────────────────────────────────────────────────

AnonymousId = str  # operator-curated anonymous identifier, e.g. "record_001"
ControlledVocabToken = str  # member of a fixed controlled-vocabulary set
CountInt = int  # aggregate count; no row content


# ── Section 1 — Ingestion boundary ────────────────────────────────────────


class V2SourceKind(Enum):
    LOCAL_PDF = "local_pdf"
    LOCAL_IMAGE = "local_image"
    LOCAL_TEXT = "local_text"
    LOCAL_OFFICE_DOC = "local_office_doc"
    LOCAL_OTHER = "local_other"


@dataclass(frozen=True)
class V2DocumentSource:
    """Privacy-safe description of a single document source.

    `anonymous_id` is operator-curated; no raw filenames, no private
    paths, no PHI. `byte_size_bucket` is a coarse aggregate bucket
    (e.g. "tiny", "small", "medium", "large"), never an exact byte
    count of a private file.
    """

    anonymous_id: AnonymousId
    source_kind: V2SourceKind
    byte_size_bucket: ControlledVocabToken
    operator_curated_label: Optional[ControlledVocabToken] = None


@dataclass(frozen=True)
class V2IngestionRequest:
    """A request to ingest a privacy-safe document source."""

    source: V2DocumentSource
    safety_profile_id: ControlledVocabToken


@runtime_checkable
class V2IngestionAdapterProtocol(Protocol):
    """Local-only ingestion adapter contract.

    A concrete implementation must:

    * never print raw filenames or private filesystem paths;
    * never read remote URLs;
    * never call external APIs;
    * never embed source content in any public report.
    """

    def supports(self, source: V2DocumentSource) -> bool: ...

    def describe(self, request: V2IngestionRequest) -> Mapping[str, ControlledVocabToken]:
        """Return aggregate-only descriptor metadata for the source.

        The descriptor must contain only controlled-vocabulary tokens
        and aggregate counts. No row content. No raw text. No private
        paths.
        """


# ── Section 2 — Text visibility / document quality boundary ───────────────


class V2VisibilityStatus(Enum):
    VISIBLE = "visible"
    PARTIALLY_VISIBLE = "partially_visible"
    NOT_VISIBLE = "not_visible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class V2TextVisibilityProfile:
    """Aggregate profile of text/layout visibility for a single source.

    Fields are controlled-vocabulary tokens or coarse aggregate
    buckets. No raw extracted text, no row content.
    """

    anonymous_id: AnonymousId
    pdf_text_layer_detected: ControlledVocabToken  # "yes" | "no" | "unknown"
    image_like_pdf: ControlledVocabToken  # "yes" | "no" | "unknown"
    alphabetic_content_bucket: ControlledVocabToken  # "none" | "low" | "medium" | "high"
    native_text_length_bucket: ControlledVocabToken  # "none" | "tiny" | "short" | "medium" | "long"
    table_like_structure_detected: ControlledVocabToken  # "yes" | "no" | "unknown"
    visibility_status: V2VisibilityStatus


@runtime_checkable
class V2DocumentQualityProtocol(Protocol):
    """Aggregate-only document-quality estimator contract.

    A concrete implementation must emit only aggregate buckets and
    controlled-vocabulary tokens. No raw OCR text, no row content,
    no clinical interpretation.
    """

    def profile(self, source: V2DocumentSource) -> V2TextVisibilityProfile: ...


# ── Section 3 — Extraction / OCR orchestration boundary ───────────────────


class V2ExtractionMode(Enum):
    NATIVE_TEXT = "native_text"
    OCR_DEFAULT = "ocr_default"
    OCR_FALLBACK = "ocr_fallback"
    DISABLED = "disabled"


@dataclass(frozen=True)
class V2ExtractionRequest:
    source: V2DocumentSource
    mode: V2ExtractionMode
    safety_profile_id: ControlledVocabToken


@dataclass(frozen=True)
class V2ExtractionResult:
    """Aggregate-only extraction result. NO raw extracted text fields.

    `text_length_bucket` is a coarse aggregate bucket. The concrete
    extracted text remains in a private store; it must never appear
    in a public report.
    """

    anonymous_id: AnonymousId
    mode_used: V2ExtractionMode
    text_length_bucket: ControlledVocabToken
    confidence_band: ControlledVocabToken  # "none" | "low" | "medium" | "high"
    fallback_used: bool = False
    private_store_reference: Optional[ControlledVocabToken] = None  # opaque token only


@runtime_checkable
class V2ExtractionAdapterProtocol(Protocol):
    """Extraction / OCR adapter contract.

    A concrete implementation must:

    * never embed the extracted text in the returned dataclass;
    * never write the extracted text into a public report;
    * never call external OCR APIs unless explicitly approved by a
      separate SPEC.
    """

    def extract(self, request: V2ExtractionRequest) -> V2ExtractionResult: ...


# ── Section 4 — Document classification boundary ──────────────────────────


@dataclass(frozen=True)
class V2DocumentTypeCandidate:
    """A single candidate document-type label."""

    label: ControlledVocabToken  # member of a fixed family-label set
    confidence_band: ControlledVocabToken  # "none" | "low" | "medium" | "high"
    reason_codes: Tuple[ControlledVocabToken, ...] = ()


@dataclass(frozen=True)
class V2ClassificationResult:
    """Aggregate-only classification result."""

    anonymous_id: AnonymousId
    candidates: Tuple[V2DocumentTypeCandidate, ...]
    review_required: bool = True
    auto_accept_allowed: bool = False
    cue_expansion_used: bool = False  # must remain false by contract


@runtime_checkable
class V2ClassifierProtocol(Protocol):
    """Document classifier contract.

    A concrete implementation must:

    * return `review_required=True` by default;
    * return `auto_accept_allowed=False` by default;
    * return `cue_expansion_used=False` by default (cue expansion
      remains explicitly NOT recommended);
    * never emit raw text or row content.
    """

    def classify(
        self,
        profile: V2TextVisibilityProfile,
        extraction: V2ExtractionResult,
    ) -> V2ClassificationResult: ...


# ── Section 5 — Clinical knowledge / terminology boundary ─────────────────


@dataclass(frozen=True)
class V2TerminologyQuery:
    """Operator-curated terminology lookup query.

    `candidate_text` is an operator-curated short token, not raw
    document text. The contract explicitly rejects raw OCR /
    document text as input.
    """

    candidate_text: ControlledVocabToken
    system_filter: Tuple[ControlledVocabToken, ...] = ()


@dataclass(frozen=True)
class V2TerminologyMatchSummary:
    """Aggregate-only terminology match summary.

    Contains only:

    * a controlled-vocabulary `match_family` token,
    * a controlled-vocabulary `terminology_system_family` token,
    * an aggregate `matches_count` (no row content),
    * explicit review-bound / refusal / privacy invariant flags.

    NEVER contains licensed row content: no codes, no displays, no
    synonyms, no definitions.
    """

    anonymous_id: AnonymousId
    match_family: ControlledVocabToken  # e.g. "exact_terminology_match"
    terminology_system_family: ControlledVocabToken  # e.g. "rxnorm" | "loinc"
    matches_count: CountInt
    review_required: bool = True
    auto_accept_allowed: bool = False
    licensed_row_content_included: bool = False
    public_report_safe: bool = True


@runtime_checkable
class V2TerminologyLookupProtocol(Protocol):
    """Aggregate-only terminology lookup contract.

    A concrete implementation must:

    * accept only operator-curated `V2TerminologyQuery` inputs;
    * return only `V2TerminologyMatchSummary` outputs (aggregate only);
    * never embed licensed row content in the return value;
    * fail closed (return `None`) when no audited adapter is available;
    * remain default-off behind an env gate at runtime.
    """

    def lookup(
        self,
        query: V2TerminologyQuery,
    ) -> Optional[V2TerminologyMatchSummary]: ...


# ── Section 6 — Review / HITL boundary ────────────────────────────────────


class V2ReviewDisposition(Enum):
    PENDING_REVIEW = "pending_review"
    OPERATOR_APPROVED = "operator_approved"
    OPERATOR_REJECTED = "operator_rejected"
    OPERATOR_DEFERRED = "operator_deferred"
    NEEDS_OPERATOR_CLARIFICATION = "needs_operator_clarification"


@dataclass(frozen=True)
class V2ReviewItem:
    """A single review-bound item in the HITL queue."""

    anonymous_id: AnonymousId
    classification: V2ClassificationResult
    terminology: Optional[V2TerminologyMatchSummary] = None
    disposition: V2ReviewDisposition = V2ReviewDisposition.PENDING_REVIEW
    review_required: bool = True
    auto_accept_allowed: bool = False


@runtime_checkable
class V2ReviewQueueProtocol(Protocol):
    """HITL review queue contract.

    A concrete implementation must:

    * keep every item review-bound by default;
    * never auto-accept;
    * never expose raw filenames or private paths in any public
      report.
    """

    def enqueue(self, item: V2ReviewItem) -> None: ...

    def pending_count(self) -> CountInt: ...

    def list_pending(self) -> Sequence[V2ReviewItem]: ...


# ── Section 7 — Operator action boundary ──────────────────────────────────


class V2OperatorActionKind(Enum):
    OPEN_RUN_AND_REVIEW = "open_run_and_review"
    OPEN_ADVANCED_TECHNICAL_DETAILS = "open_advanced_technical_details"
    REQUEST_RE_EXTRACTION = "request_re_extraction"
    DEFER_REVIEW = "defer_review"
    REJECT_REVIEW = "reject_review"
    APPROVE_REVIEW = "approve_review"
    REQUEST_ESCALATION = "request_escalation"


@dataclass(frozen=True)
class V2OperatorActionRequest:
    anonymous_id: AnonymousId
    action_kind: V2OperatorActionKind
    safety_profile_id: ControlledVocabToken


@dataclass(frozen=True)
class V2OperatorActionResult:
    anonymous_id: AnonymousId
    action_kind: V2OperatorActionKind
    accepted: bool
    review_required_after_action: bool = True
    state_mutation_performed: bool = False  # must remain False for non-state-changing actions


@runtime_checkable
class V2OperatorActionProtocol(Protocol):
    """Operator action contract.

    A concrete implementation must:

    * never auto-accept a clinical decision;
    * remain review-bound after every action;
    * never bypass the v2 privacy gate.
    """

    def submit(self, request: V2OperatorActionRequest) -> V2OperatorActionResult: ...


# ── Section 8 — Audit / observability boundary ────────────────────────────


class V2AuditEventKind(Enum):
    INGESTION_STARTED = "ingestion_started"
    QUALITY_PROFILED = "quality_profiled"
    EXTRACTION_PERFORMED = "extraction_performed"
    CLASSIFICATION_PERFORMED = "classification_performed"
    TERMINOLOGY_LOOKUP_PERFORMED = "terminology_lookup_performed"
    REVIEW_QUEUED = "review_queued"
    OPERATOR_ACTION_SUBMITTED = "operator_action_submitted"
    SAFETY_GATE_REFUSAL = "safety_gate_refusal"
    VALIDATION_RECEIPT_EMITTED = "validation_receipt_emitted"


@dataclass(frozen=True)
class V2AuditEvent:
    """Aggregate-only audit event.

    Carries only an anonymous identifier, event kind, controlled-
    vocabulary status token, and an aggregate timestamp. NEVER carries
    raw text, raw filenames, private paths, PHI, or secrets.
    """

    anonymous_id: AnonymousId
    kind: V2AuditEventKind
    status_token: ControlledVocabToken
    occurred_at: datetime
    review_required: bool = True


@runtime_checkable
class V2ObservabilitySinkProtocol(Protocol):
    """Observability sink contract.

    A concrete implementation must:

    * accept only `V2AuditEvent` instances;
    * never accept raw text / filename / private path fields;
    * never enable external telemetry by default.
    """

    def record(self, event: V2AuditEvent) -> None: ...

    def aggregate_counts_by_kind(self) -> Mapping[V2AuditEventKind, CountInt]: ...


# ── Section 9 — Validation report boundary ────────────────────────────────


class V2ValidationStatus(Enum):
    READY = "ready"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    REGRESSED = "regressed"


@dataclass(frozen=True)
class V2ValidationReceipt:
    """Aggregate-only validation receipt."""

    receipt_id: ControlledVocabToken
    validation_name: ControlledVocabToken  # e.g. "cka_final_mvp_release"
    status: V2ValidationStatus
    cases_total: CountInt
    cases_passed: CountInt
    cases_failed: CountInt
    external_api_used: bool = False
    cue_expansion_recommended: bool = False


@runtime_checkable
class V2ValidationHarnessProtocol(Protocol):
    """Validation harness contract.

    A concrete implementation must:

    * carry the v1 five-validation health-check set forward unchanged;
    * never overwrite existing v1 validation scripts;
    * emit only aggregate `V2ValidationReceipt` results;
    * never embed raw test output in a public report.
    """

    def run(
        self,
        validation_name: ControlledVocabToken,
    ) -> V2ValidationReceipt: ...

    def run_all(self) -> Sequence[V2ValidationReceipt]: ...


# ── Section 10 — Runtime privacy / safety boundary ────────────────────────


@dataclass(frozen=True)
class V2RuntimeSafetyProfile:
    """Canonical default runtime safety profile.

    Every concrete v2 adapter must accept (or be configured with) a
    `V2RuntimeSafetyProfile`. The default profile carries the
    contract-level invariants:

    * `local_only=True`
    * `review_bound=True`
    * `external_api_blocked=True`
    * `auto_accept_allowed=False`
    * `terminology_lookup_aggregate_only=True`
    * `private_adapter_implemented=False`
    * `cue_expansion_recommended=False`
    * `clinical_decision_expansion=False`
    """

    profile_id: ControlledVocabToken = "v2_default_safety_profile"
    local_only: bool = True
    review_bound: bool = True
    external_api_blocked: bool = True
    auto_accept_allowed: bool = False
    terminology_lookup_aggregate_only: bool = True
    private_adapter_implemented: bool = False
    cue_expansion_recommended: bool = False
    clinical_decision_expansion: bool = False
    notes: Tuple[ControlledVocabToken, ...] = field(default_factory=tuple)


@runtime_checkable
class V2PrivacyGateProtocol(Protocol):
    """Runtime privacy gate contract.

    A concrete implementation must:

    * refuse to emit any record that contains licensed row content,
      raw text, raw filenames, private paths, PHI, or secrets;
    * be consulted by every adapter before any public emission;
    * default to fail-closed on ambiguity.
    """

    def is_emission_allowed(
        self,
        candidate: Mapping[str, object],
        profile: V2RuntimeSafetyProfile,
    ) -> bool: ...

    def refusal_reason(
        self,
        candidate: Mapping[str, object],
        profile: V2RuntimeSafetyProfile,
    ) -> Optional[ControlledVocabToken]: ...


# ── Helper: enumerated contract names (for tests / audits) ────────────────

CONTRACT_NAMES: Tuple[str, ...] = (
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


__all__ = (
    "AnonymousId",
    "ControlledVocabToken",
    "CountInt",
    "CONTRACT_NAMES",
    *CONTRACT_NAMES,
)
