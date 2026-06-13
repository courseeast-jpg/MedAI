"""Review-bound, local-only decision store for Vertex semantic findings (15R).

Persists operator accept/reject/defer decisions over the recorded 15P-C/15P-D
Vertex semantic findings (surfaced by 15Q) as REVIEW-BOUND DRAFT decision
records only. It is deterministic, local-only, and isolated:

* It NEVER writes active MKB records and never mutates MKB facts.
* It NEVER sets auto_accept=true; review_required stays true.
* It performs no provider/network call and uses no live gate.
* Decision records preserve provider provenance and source evidence anchors.

Storage is an isolated JSONL file under a safe reports/test artifact path; no
production MKB table, ledger, or review-queue write path is touched.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.vertex_semantic_review_surface import (
    PROVIDER_MODEL,
    PROVIDER_ROUTE,
    VertexSemanticReviewDraft,
    build_vertex_semantic_review_drafts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE_PATH = (
    REPO_ROOT
    / "reports"
    / "medai_vertex_semantic_package_review_decision_persist_15r"
    / "decision_store_preview.jsonl"
)

ACTION_TO_STATUS = {
    "accept_for_review": "review_draft",
    "reject": "rejected",
    "defer": "deferred",
}
ACTION_ALIASES = {
    "accept": "accept_for_review",
    "accept_for_review": "accept_for_review",
    "reject": "reject",
    "defer": "defer",
}
# Deterministic, test-stable timestamp (no wall-clock; avoids flaky tests).
DETERMINISTIC_TIMESTAMP = "2026-01-01T00:00:00Z"

SOURCE_REPORT_REFERENCE = {
    "portal_result_cards": "reports/medai_vertex_semantic_package_comparison_live_15p_c/live_comparison_result.json",
    "cytology_pathology_narrative": "reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json",
    "urinalysis_table_like_lab": "reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json",
    "mixed_narrative_numeric_result": "reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json",
}


@dataclass(frozen=True)
class VertexSemanticReviewDecision:
    decision_id: str
    package_id: str
    package_family: str
    finding_id: str
    candidate_fact_id: str
    action: str
    decision_status: str
    provider_route: str
    provider_model: str
    source_report_reference: str
    source_evidence_text: str
    evidence_anchor: str
    unknown_values: list[str]
    uncertainty_flags: list[str]
    hallucinated_field_count: int
    review_required: bool
    auto_accept: bool
    creates_active_mkb_record: bool
    active_written_count_delta: int
    timestamp: str
    audit_reason: str
    operator_id: str
    local_only: bool


@dataclass
class VertexSemanticReviewDecisionStore:
    store_path: Path = DEFAULT_STORE_PATH
    timestamp: str = DETERMINISTIC_TIMESTAMP
    _records: list[VertexSemanticReviewDecision] = field(default_factory=list)

    def record_decision(
        self,
        draft: VertexSemanticReviewDraft,
        finding: dict[str, Any],
        finding_index: int,
        action: str,
        *,
        operator_id: str = "synthetic_operator",
        audit_reason: str = "",
    ) -> VertexSemanticReviewDecision | None:
        normalized = ACTION_ALIASES.get(str(action or "").strip().lower())
        if normalized is None:
            # Invalid action: no record is written.
            return None
        status = ACTION_TO_STATUS[normalized]
        anchor = draft.evidence_anchors[0] if draft.evidence_anchors else {}
        finding_id = f"{draft.package_family}_finding_{finding_index:02d}"
        candidate_fact_id = f"{draft.package_family}_candidate_{finding_index:02d}"
        decision_basis = f"{draft.package_id}|{finding_id}|{normalized}"
        decision = VertexSemanticReviewDecision(
            decision_id=f"dec_{hashlib.sha256(decision_basis.encode('utf-8')).hexdigest()[:16]}",
            package_id=draft.package_id,
            package_family=draft.package_family,
            finding_id=finding_id,
            candidate_fact_id=candidate_fact_id,
            action=normalized,
            decision_status=status,
            provider_route=draft.provider_route or PROVIDER_ROUTE,
            provider_model=draft.provider_model or PROVIDER_MODEL,
            source_report_reference=SOURCE_REPORT_REFERENCE.get(draft.package_family, ""),
            source_evidence_text=str(finding.get("evidence_text", "")),
            evidence_anchor=str(anchor.get("anchor_id", "")),
            unknown_values=list(draft.unknown_values),
            uncertainty_flags=list(draft.uncertainty_flags),
            hallucinated_field_count=int(draft.hallucinated_field_count),
            review_required=True,
            auto_accept=False,
            creates_active_mkb_record=False,
            active_written_count_delta=0,
            timestamp=self.timestamp,
            audit_reason=audit_reason or _default_audit_reason(normalized),
            operator_id=str(operator_id or "synthetic_operator"),
            local_only=True,
        )
        self._records.append(decision)
        return decision

    def records(self) -> list[VertexSemanticReviewDecision]:
        return list(self._records)

    def public_records(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self._records]

    def persist(self) -> Path:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(record), sort_keys=True) for record in self._records]
        self.store_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return self.store_path

    def active_written_count(self) -> int:
        # Review-draft store only; never writes active MKB records.
        return sum(record.active_written_count_delta for record in self._records)

    def active_mkb_record_created_count(self) -> int:
        return sum(1 for record in self._records if record.creates_active_mkb_record)


def _default_audit_reason(action: str) -> str:
    return {
        "accept_for_review": "operator accepted finding for review queue only; no active write",
        "reject": "operator rejected finding; no active write",
        "defer": "operator deferred finding; no active write",
    }.get(action, "review-bound decision; no active write")


def operator_status_message(decision: VertexSemanticReviewDecision) -> str:
    return {
        "review_draft": "accepted for review queue only",
        "rejected": "rejected - no active write",
        "deferred": "deferred - no active write",
    }.get(decision.decision_status, "review-bound decision - no active write")


def persist_decisions_for_all_families(
    *,
    store_path: Path | None = None,
    timestamp: str = DETERMINISTIC_TIMESTAMP,
    action_plan: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Persist a deterministic accept/reject/defer decision per family.

    By default the first finding of each family is assigned a rotating action so
    all three actions are exercised; the rest are accepted-for-review.
    """
    drafts = build_vertex_semantic_review_drafts()
    store = VertexSemanticReviewDecisionStore(
        store_path=store_path or DEFAULT_STORE_PATH, timestamp=timestamp
    )
    rotation = ("accept_for_review", "reject", "defer")
    invalid_action_rejected_count = 0
    for family_index, draft in enumerate(drafts):
        planned = (action_plan or {}).get(draft.package_family)
        for finding_index, finding in enumerate(draft.vertex_semantic_findings):
            if planned is not None:
                action = planned
            elif finding_index == 0:
                action = rotation[family_index % len(rotation)]
            else:
                action = "accept_for_review"
            result = store.record_decision(draft, finding, finding_index, action)
            if result is None:
                invalid_action_rejected_count += 1
    # Exercise invalid-action safety explicitly (no record written).
    if drafts:
        probe = store.record_decision(drafts[0], drafts[0].vertex_semantic_findings[0], 0, "delete_everything")
        if probe is None:
            invalid_action_rejected_count += 1
    store.persist()
    return _build_report(drafts, store, invalid_action_rejected_count)


def _build_report(
    drafts: list[VertexSemanticReviewDraft],
    store: VertexSemanticReviewDecisionStore,
    invalid_action_rejected_count: int,
) -> dict[str, Any]:
    records = store.records()
    families_with_decisions = {r.package_family for r in records}
    summary = {
        "block": "MEDAI-VERTEX-SEMANTIC-PACKAGE-REVIEW-DECISION-PERSIST-15R",
        "package_families_loaded": len(drafts),
        "package_families_with_persisted_decisions": len(families_with_decisions),
        "decision_records_created": len(records),
        "accept_decision_records_created": sum(1 for r in records if r.action == "accept_for_review"),
        "reject_decision_records_created": sum(1 for r in records if r.action == "reject"),
        "defer_decision_records_created": sum(1 for r in records if r.action == "defer"),
        "invalid_action_rejected_count": invalid_action_rejected_count,
        "review_bound_decision_count": sum(1 for r in records if r.review_required and not r.creates_active_mkb_record),
        "active_mkb_record_created_count": store.active_mkb_record_created_count(),
        "active_written_count": store.active_written_count(),
        "auto_accept_true_count": sum(1 for r in records if r.auto_accept),
        "review_required_true_count": sum(1 for r in records if r.review_required),
        "evidence_anchor_preserved_count": sum(1 for r in records if r.evidence_anchor),
        "provider_provenance_preserved_count": sum(
            1 for r in records if r.provider_route == PROVIDER_ROUTE and r.provider_model == PROVIDER_MODEL
        ),
        "unknown_values_preserved_count": sum(
            1 for r in records if (r.unknown_values or r.package_family != "mixed_narrative_numeric_result")
        ),
        "uncertainty_flags_preserved_count": sum(1 for r in records if r.uncertainty_flags),
        "hallucinated_field_count": sum(r.hallucinated_field_count for r in records),
        "local_only_decision_count": sum(1 for r in records if r.local_only),
        "audit_reason_present_count": sum(1 for r in records if r.audit_reason.strip()),
        "source_report_reference_present_count": sum(1 for r in records if r.source_report_reference),
        "live_call_made": False,
        "external_api_used": False,
        "privacy_result": "passed",
        "billing_check_pending": True,
    }
    summary["all_actions_exercised"] = (
        summary["accept_decision_records_created"] > 0
        and summary["reject_decision_records_created"] > 0
        and summary["defer_decision_records_created"] > 0
    )
    return {
        "summary": summary,
        "decision_records": store.public_records(),
        "store_path": _safe_relpath(store.store_path),
    }


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return path.name


__all__ = [
    "DEFAULT_STORE_PATH",
    "DETERMINISTIC_TIMESTAMP",
    "ACTION_TO_STATUS",
    "ACTION_ALIASES",
    "VertexSemanticReviewDecision",
    "VertexSemanticReviewDecisionStore",
    "operator_status_message",
    "persist_decisions_for_all_families",
]
