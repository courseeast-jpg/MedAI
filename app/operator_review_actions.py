"""Operator review actions for review-bound extracted MKB facts.

MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-03.

Streamlit-free. Deterministic. Local-only. No external API. No clinical
interpretation. No medication-safety bypass. No DDI bypass.

Actions:

* :func:`accept_after_source_comparison` — operator confirms the
  factual extraction matches the source document. Record moves to
  active / status=active / requires_review=False.
* :func:`reject_extracted_fact` — operator marks the extraction as not
  usable. Record moves to superseded / status=rejected_after_operator_review
  / requires_review=False.
* :func:`defer_extracted_fact` — operator keeps the record review-bound
  while noting the deferral. Tier remains quarantined, status becomes
  deferred_by_operator, requires_review stays True.

Every action returns an :class:`OperatorActionResult` containing only
public-safe fields. Raw source text, raw OCR text, raw filenames, and
private paths are **never** included in the result.

Each action also writes a :class:`LedgerEvent` via the SQLite ledger so
the audit trail is durable. Operator notes are accepted but **never**
echoed into the public ledger payload — the ledger records only
``operator_note_present`` (boolean) plus a short note length bucket.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from app.config import TIER_ACTIVE, TIER_QUARANTINED, TIER_SUPERSEDED
from app.schemas import LedgerEvent, MKBRecord

#: The set of fact_types this lab-review workflow is allowed to handle.
ALLOWED_FACT_TYPES = frozenset({"test_result", "observation"})

#: DDI states that block accept regardless of operator intent.
BLOCKED_DDI_STATES = frozenset(
    {
        "high_blocked",
        "pending_ddi",
        "pending_ddi_check",
        "pending_medication_review",
    }
)

#: Ledger event_type used for every operator review action.
LEDGER_EVENT_TYPE = "operator_review_action"
LEDGER_SOURCE = "operator_review_ux"

#: Safety disclaimer surfaced in result messages and required to be
#: visible to the operator before accept.
ACCEPT_DISCLAIMER = (
    "Accept only after comparing this value with the source document. "
    "This does not clinically interpret the result."
)

DEFER_DISCLAIMER = (
    "Deferred. The record stays review-bound. No clinical interpretation occurred."
)

REJECT_DISCLAIMER = (
    "Rejected. The record will not be used as a source-derived fact."
)


@dataclass
class OperatorActionResult:
    """Public-safe result of one operator action.

    Never carries: raw source text, raw OCR text, raw filenames,
    private paths, or PHI. The operator note text itself is **not**
    included; only ``operator_note_present`` is reported.
    """

    success: bool
    action: str
    record_id: str
    previous_tier: Optional[str] = None
    previous_status: Optional[str] = None
    previous_requires_review: Optional[bool] = None
    new_tier: Optional[str] = None
    new_status: Optional[str] = None
    new_requires_review: Optional[bool] = None
    operator_note_present: bool = False
    safe_message: str = ""
    error_code: Optional[str] = None
    ledger_event_id: Optional[int] = None
    timestamp: Optional[str] = None

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _note_length_bucket(operator_note: Optional[str]) -> str:
    """Return a coarse, non-identifying note-length bucket.

    The note text itself is never written to the ledger. The bucket
    helps an audit reviewer answer "did the operator type anything" at
    a glance without exposing content.
    """
    if not operator_note:
        return "none"
    n = len(str(operator_note))
    if n <= 16:
        return "short"
    if n <= 80:
        return "medium"
    return "long"


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _is_review_bound(record: MKBRecord) -> bool:
    return bool(record.requires_review) or record.tier == TIER_QUARANTINED


def _is_already_active(record: MKBRecord) -> bool:
    return (
        record.tier == TIER_ACTIVE
        and (record.requires_review is False)
        and record.status in {"active", "accepted_after_operator_review"}
    )


def _is_ddi_blocked(record: MKBRecord) -> bool:
    ddi = (record.ddi_status or "").strip()
    return ddi in BLOCKED_DDI_STATES


def _write_action_ledger(
    sql_store: Any,
    *,
    action: str,
    record: MKBRecord,
    previous_snapshot: dict[str, Any],
    new_snapshot: dict[str, Any],
    operator_note: Optional[str],
    session_id: str,
) -> Optional[int]:
    """Append a ledger event. Never emits the note text itself."""
    if sql_store is None or not hasattr(sql_store, "write_ledger"):
        return None
    details: dict[str, Any] = {
        "action": action,
        "fact_type": record.fact_type,
        "operator_note_present": bool(operator_note),
        "operator_note_length_bucket": _note_length_bucket(operator_note),
        "old": previous_snapshot,
        "new": new_snapshot,
        "source": LEDGER_SOURCE,
    }
    event = LedgerEvent(
        event_type=LEDGER_EVENT_TYPE,
        record_id=record.id,
        source_type="operator_review",
        previous_value=previous_snapshot,
        details=details,
        session_id=session_id or record.session_id or "",
    )
    try:
        return int(sql_store.write_ledger(event))
    except Exception:
        return None


def _persist_record_transition(
    sql_store: Any,
    *,
    record: MKBRecord,
    new_tier: str,
    new_status: str,
    new_requires_review: bool,
    operator_note: Optional[str],
    action: str,
) -> tuple[dict[str, Any], dict[str, Any], MKBRecord]:
    """Apply the transition to ``record`` and persist it.

    Mutates ``record`` in place; returns the old and new snapshots plus
    the mutated record for the caller to inspect or return downstream.
    """
    previous_snapshot = {
        "tier": record.tier,
        "status": record.status,
        "requires_review": bool(record.requires_review),
    }
    record.tier = new_tier
    record.status = new_status
    record.requires_review = bool(new_requires_review)
    record.last_confirmed = datetime.utcnow()
    structured = dict(record.structured or {})
    structured["operator_review_status"] = new_status
    structured["operator_reviewed_at"] = _now_iso()
    structured["operator_action"] = action
    structured["operator_note_present"] = bool(operator_note)
    # Replace the structured payload but never embed the note text.
    record.structured = structured
    if sql_store is not None and hasattr(sql_store, "write_record"):
        sql_store.write_record(record, session_id=record.session_id or "")
    new_snapshot = {
        "tier": record.tier,
        "status": record.status,
        "requires_review": bool(record.requires_review),
    }
    return previous_snapshot, new_snapshot, record


def _failure(
    *,
    action: str,
    record_id: str,
    error_code: str,
    safe_message: str,
    previous: Optional[MKBRecord] = None,
) -> OperatorActionResult:
    prev_tier = previous.tier if previous is not None else None
    prev_status = previous.status if previous is not None else None
    prev_review = bool(previous.requires_review) if previous is not None else None
    return OperatorActionResult(
        success=False,
        action=action,
        record_id=str(record_id),
        previous_tier=prev_tier,
        previous_status=prev_status,
        previous_requires_review=prev_review,
        error_code=error_code,
        safe_message=safe_message,
        timestamp=_now_iso(),
    )


def _guard_acceptable(record: MKBRecord, *, action: str) -> Optional[OperatorActionResult]:
    """Shared guards for accept / reject paths."""
    if record.fact_type == "medication":
        return _failure(
            action=action,
            record_id=record.id,
            error_code="medication_safety_workflow_required",
            safe_message=(
                "Medication facts require the medication safety workflow, "
                "not the lab review workflow."
            ),
            previous=record,
        )
    if _is_ddi_blocked(record):
        return _failure(
            action=action,
            record_id=record.id,
            error_code="ddi_blocked_or_pending",
            safe_message="DDI status is blocked or pending; lab review accept is not allowed.",
            previous=record,
        )
    if record.fact_type not in ALLOWED_FACT_TYPES:
        return _failure(
            action=action,
            record_id=record.id,
            error_code="fact_type_not_supported_by_lab_review",
            safe_message=(
                "This fact type is not supported by the lab review workflow."
            ),
            previous=record,
        )
    return None


def accept_after_source_comparison(
    sql_store: Any,
    record_id: str,
    *,
    operator_note: Optional[str] = None,
    session_id: str = "",
) -> OperatorActionResult:
    """Operator accepts a review-bound extracted fact after source comparison.

    Allowed only for review-bound factual records (``test_result``,
    ``observation``). Medication facts must use the medication safety
    workflow. DDI-blocked records cannot be accepted via this path.
    """
    action = "accept_after_source_comparison"
    record = sql_store.get_record(record_id) if sql_store is not None else None
    if record is None:
        return _failure(
            action=action,
            record_id=record_id,
            error_code="record_not_found",
            safe_message="Record not found in MKB.",
        )
    if _is_already_active(record):
        return _failure(
            action=action,
            record_id=record.id,
            error_code="record_already_active",
            safe_message="Record is already active and accepted; nothing to do.",
            previous=record,
        )
    guard = _guard_acceptable(record, action=action)
    if guard is not None:
        return guard
    if not _is_review_bound(record):
        return _failure(
            action=action,
            record_id=record.id,
            error_code="record_not_review_bound",
            safe_message="Record is not in a review-bound state.",
            previous=record,
        )

    previous_snapshot, new_snapshot, mutated = _persist_record_transition(
        sql_store,
        record=record,
        new_tier=TIER_ACTIVE,
        new_status="accepted_after_operator_review",
        new_requires_review=False,
        operator_note=operator_note,
        action=action,
    )
    ledger_event_id = _write_action_ledger(
        sql_store,
        action=action,
        record=mutated,
        previous_snapshot=previous_snapshot,
        new_snapshot=new_snapshot,
        operator_note=operator_note,
        session_id=session_id,
    )
    return OperatorActionResult(
        success=True,
        action=action,
        record_id=mutated.id,
        previous_tier=previous_snapshot["tier"],
        previous_status=previous_snapshot["status"],
        previous_requires_review=previous_snapshot["requires_review"],
        new_tier=new_snapshot["tier"],
        new_status=new_snapshot["status"],
        new_requires_review=new_snapshot["requires_review"],
        operator_note_present=bool(operator_note),
        safe_message=ACCEPT_DISCLAIMER,
        ledger_event_id=ledger_event_id,
        timestamp=_now_iso(),
    )


def reject_extracted_fact(
    sql_store: Any,
    record_id: str,
    *,
    operator_note: Optional[str] = None,
    session_id: str = "",
) -> OperatorActionResult:
    """Operator rejects an extracted fact.

    Record moves to superseded / status=rejected_after_operator_review.
    The record is preserved for audit but is no longer a source-derived
    active fact.
    """
    action = "reject_extracted_fact"
    record = sql_store.get_record(record_id) if sql_store is not None else None
    if record is None:
        return _failure(
            action=action,
            record_id=record_id,
            error_code="record_not_found",
            safe_message="Record not found in MKB.",
        )
    guard = _guard_acceptable(record, action=action)
    if guard is not None:
        return guard

    previous_snapshot, new_snapshot, mutated = _persist_record_transition(
        sql_store,
        record=record,
        new_tier=TIER_SUPERSEDED,
        new_status="rejected_after_operator_review",
        new_requires_review=False,
        operator_note=operator_note,
        action=action,
    )
    ledger_event_id = _write_action_ledger(
        sql_store,
        action=action,
        record=mutated,
        previous_snapshot=previous_snapshot,
        new_snapshot=new_snapshot,
        operator_note=operator_note,
        session_id=session_id,
    )
    return OperatorActionResult(
        success=True,
        action=action,
        record_id=mutated.id,
        previous_tier=previous_snapshot["tier"],
        previous_status=previous_snapshot["status"],
        previous_requires_review=previous_snapshot["requires_review"],
        new_tier=new_snapshot["tier"],
        new_status=new_snapshot["status"],
        new_requires_review=new_snapshot["requires_review"],
        operator_note_present=bool(operator_note),
        safe_message=REJECT_DISCLAIMER,
        ledger_event_id=ledger_event_id,
        timestamp=_now_iso(),
    )


def defer_extracted_fact(
    sql_store: Any,
    record_id: str,
    *,
    operator_note: Optional[str] = None,
    session_id: str = "",
) -> OperatorActionResult:
    """Operator defers the decision; record stays review-bound.

    Defer is a no-op-with-audit transition. Tier stays
    ``quarantined``; status becomes ``deferred_by_operator``;
    ``requires_review`` stays True. A ledger event is written so the
    deferral is durable.
    """
    action = "defer_extracted_fact"
    record = sql_store.get_record(record_id) if sql_store is not None else None
    if record is None:
        return _failure(
            action=action,
            record_id=record_id,
            error_code="record_not_found",
            safe_message="Record not found in MKB.",
        )
    if record.fact_type not in ALLOWED_FACT_TYPES:
        return _failure(
            action=action,
            record_id=record.id,
            error_code="fact_type_not_supported_by_lab_review",
            safe_message="This fact type is not supported by the lab review workflow.",
            previous=record,
        )
    # Defer is permitted even when the record is already active — it
    # explicitly re-enters review with operator intent. But auto-accept
    # is never enabled by this path.
    previous_snapshot, new_snapshot, mutated = _persist_record_transition(
        sql_store,
        record=record,
        new_tier=TIER_QUARANTINED,
        new_status="deferred_by_operator",
        new_requires_review=True,
        operator_note=operator_note,
        action=action,
    )
    ledger_event_id = _write_action_ledger(
        sql_store,
        action=action,
        record=mutated,
        previous_snapshot=previous_snapshot,
        new_snapshot=new_snapshot,
        operator_note=operator_note,
        session_id=session_id,
    )
    return OperatorActionResult(
        success=True,
        action=action,
        record_id=mutated.id,
        previous_tier=previous_snapshot["tier"],
        previous_status=previous_snapshot["status"],
        previous_requires_review=previous_snapshot["requires_review"],
        new_tier=new_snapshot["tier"],
        new_status=new_snapshot["status"],
        new_requires_review=new_snapshot["requires_review"],
        operator_note_present=bool(operator_note),
        safe_message=DEFER_DISCLAIMER,
        ledger_event_id=ledger_event_id,
        timestamp=_now_iso(),
    )


def render_action_affordances_plan(record_id_or_state: dict[str, Any] | str) -> dict[str, Any]:
    """Build a Streamlit-free render plan for the per-row action panel.

    The caller passes either:

    * a record-id string (for the simplest case), or
    * a dict carrying ``record_id``, ``fact_type``, ``tier``,
      ``status``, ``requires_review`` (the typical case when rendering
      next to the preview table).

    The plan describes which actions are enabled, why others are
    disabled, and the disclaimer text. The plan does **not** contain
    raw source text, raw OCR text, raw filenames, or PHI.
    """
    if isinstance(record_id_or_state, str):
        state = {"record_id": record_id_or_state}
    else:
        state = dict(record_id_or_state or {})
    record_id = str(state.get("record_id") or "")
    fact_type = str(state.get("fact_type") or "test_result")
    tier = str(state.get("tier") or "")
    status = str(state.get("status") or "")
    requires_review = bool(state.get("requires_review", True))
    ddi_status = str(state.get("ddi_status") or "")

    accept_enabled = True
    accept_reason = "ready"
    if fact_type == "medication":
        accept_enabled = False
        accept_reason = "medication_safety_workflow_required"
    elif ddi_status in BLOCKED_DDI_STATES:
        accept_enabled = False
        accept_reason = "ddi_blocked_or_pending"
    elif fact_type not in ALLOWED_FACT_TYPES:
        accept_enabled = False
        accept_reason = "fact_type_not_supported_by_lab_review"
    elif tier == TIER_ACTIVE and not requires_review and status in {
        "active",
        "accepted_after_operator_review",
    }:
        accept_enabled = False
        accept_reason = "record_already_active"
    elif not (requires_review or tier == TIER_QUARANTINED):
        accept_enabled = False
        accept_reason = "record_not_review_bound"

    reject_enabled = accept_reason not in {"medication_safety_workflow_required", "ddi_blocked_or_pending"} and fact_type in ALLOWED_FACT_TYPES
    defer_enabled = fact_type in ALLOWED_FACT_TYPES

    return {
        "record_id": record_id,
        "fact_type": fact_type,
        "actions": [
            {
                "key": "accept_after_source_comparison",
                "label": "Accept after source comparison",
                "enabled": accept_enabled,
                "reason": accept_reason,
                "disclaimer": ACCEPT_DISCLAIMER,
            },
            {
                "key": "reject_extracted_fact",
                "label": "Reject",
                "enabled": reject_enabled,
                "reason": "ready" if reject_enabled else accept_reason,
                "disclaimer": REJECT_DISCLAIMER,
            },
            {
                "key": "defer_extracted_fact",
                "label": "Defer",
                "enabled": defer_enabled,
                "reason": "ready" if defer_enabled else "fact_type_not_supported_by_lab_review",
                "disclaimer": DEFER_DISCLAIMER,
            },
        ],
        "auto_accept_allowed": False,
        "review_required": bool(requires_review),
    }


__all__ = [
    "ALLOWED_FACT_TYPES",
    "BLOCKED_DDI_STATES",
    "LEDGER_EVENT_TYPE",
    "LEDGER_SOURCE",
    "ACCEPT_DISCLAIMER",
    "REJECT_DISCLAIMER",
    "DEFER_DISCLAIMER",
    "OperatorActionResult",
    "accept_after_source_comparison",
    "reject_extracted_fact",
    "defer_extracted_fact",
    "render_action_affordances_plan",
]
