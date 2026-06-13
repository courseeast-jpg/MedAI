#!/usr/bin/env python3
"""MEDAI-VERTEX-SEMANTIC-PACKAGE-REVIEW-DECISION-PERSIST-15R (no-live).

Persists operator accept/reject/defer decisions over the recorded 15P-C/15P-D
Vertex semantic findings into an isolated, review-bound, local-only decision
store, and generates audit reports. No provider call; no active MKB writes; no
auto-accept; no live gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.vertex_semantic_review_decision_store import (
    DEFAULT_STORE_PATH,
    persist_decisions_for_all_families,
)

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_package_review_decision_persist_15r"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CASES_JSON = REPORT_DIR / "decision_cases.json"
MATRIX_MD = REPORT_DIR / "decision_matrix.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_TOKENS = ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "DOB", "MRN", "Accession")


def _privacy_scan(payload: Any) -> bool:
    blob = json.dumps(payload, default=str)
    return not any(token in blob for token in FORBIDDEN_TOKENS)


def _matrix_markdown(summary: dict[str, Any]) -> str:
    keys = [
        "package_families_loaded",
        "package_families_with_persisted_decisions",
        "decision_records_created",
        "accept_decision_records_created",
        "reject_decision_records_created",
        "defer_decision_records_created",
        "invalid_action_rejected_count",
        "review_bound_decision_count",
        "active_mkb_record_created_count",
        "active_written_count",
        "auto_accept_true_count",
        "review_required_true_count",
        "evidence_anchor_preserved_count",
        "provider_provenance_preserved_count",
        "unknown_values_preserved_count",
        "uncertainty_flags_preserved_count",
        "hallucinated_field_count",
        "local_only_decision_count",
        "audit_reason_present_count",
        "live_call_made",
        "external_api_used",
        "privacy_result",
        "billing_check_pending",
    ]
    return "\n".join(
        [
            "# 15R Vertex semantic review-decision persistence matrix",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *[f"| {key} | `{summary[key]}` |" for key in keys],
            "",
            "Review-bound draft decisions only; no active MKB writes; no auto-accept; no live call.",
            "",
        ]
    )


def _implementation_markdown(summary: dict[str, Any], store_path: str) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-SEMANTIC-PACKAGE-REVIEW-DECISION-PERSIST-15R",
            "",
            f"- Decision store (isolated, local-only): `{store_path}`",
            f"- Package families loaded / with decisions: `{summary['package_families_loaded']}` / `{summary['package_families_with_persisted_decisions']}`",
            f"- Decision records created: `{summary['decision_records_created']}` "
            f"(accept `{summary['accept_decision_records_created']}`, reject `{summary['reject_decision_records_created']}`, defer `{summary['defer_decision_records_created']}`)",
            f"- Invalid actions rejected (no record written): `{summary['invalid_action_rejected_count']}`",
            f"- Review-bound decision count: `{summary['review_bound_decision_count']}`",
            f"- Active MKB records created: `{summary['active_mkb_record_created_count']}` | Active written count: `{summary['active_written_count']}`",
            f"- Auto-accept true count: `{summary['auto_accept_true_count']}` | Review required true count: `{summary['review_required_true_count']}`",
            f"- Evidence anchors / provider provenance preserved: `{summary['evidence_anchor_preserved_count']}` / `{summary['provider_provenance_preserved_count']}`",
            f"- Unknown / uncertainty preserved: `{summary['unknown_values_preserved_count']}` / `{summary['uncertainty_flags_preserved_count']}`",
            f"- Hallucinated field count: `{summary['hallucinated_field_count']}`",
            f"- live_call_made: `{summary['live_call_made']}` | external_api_used: `{summary['external_api_used']}`",
            f"- privacy_result: `{summary['privacy_result']}` | billing_check_pending: `{summary['billing_check_pending']}`",
            "",
            "## Safety",
            "",
            "- Decisions persist to an isolated JSONL review-draft store; no production MKB/ledger/queue write path is touched.",
            "- Accept = 'accepted for review queue only'; Reject = 'rejected - no active write'; Defer = 'deferred - no active write'.",
            "- creates_active_mkb_record=false and active_written_count_delta=0 on every record; auto_accept stays false; review_required stays true.",
            "- Evidence anchors and Vertex provider provenance preserved on every record. No provider call; no live gate.",
            "",
        ]
    )


def main() -> int:
    report = persist_decisions_for_all_families()
    summary = report["summary"]
    records = report["decision_records"]
    store_path = report["store_path"]

    privacy_ok = _privacy_scan(summary) and _privacy_scan(records)
    summary["privacy_result"] = "passed" if privacy_ok else "failed"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    CASES_JSON.write_text(json.dumps({"decision_records": records, "store_path": store_path}, indent=2), encoding="utf-8")
    MATRIX_MD.write_text(_matrix_markdown(summary), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation_markdown(summary, store_path), encoding="utf-8")

    ready = all(
        [
            summary["package_families_loaded"] == 4,
            summary["package_families_with_persisted_decisions"] == 4,
            summary["decision_records_created"] > 0,
            summary["all_actions_exercised"] is True,
            summary["invalid_action_rejected_count"] >= 1,
            summary["review_bound_decision_count"] == summary["decision_records_created"],
            summary["active_mkb_record_created_count"] == 0,
            summary["active_written_count"] == 0,
            summary["auto_accept_true_count"] == 0,
            summary["review_required_true_count"] == summary["decision_records_created"],
            summary["evidence_anchor_preserved_count"] == summary["decision_records_created"],
            summary["provider_provenance_preserved_count"] == summary["decision_records_created"],
            summary["uncertainty_flags_preserved_count"] == summary["decision_records_created"],
            summary["audit_reason_present_count"] == summary["decision_records_created"],
            summary["local_only_decision_count"] == summary["decision_records_created"],
            summary["hallucinated_field_count"] == 0,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["privacy_result"] == "passed",
            DEFAULT_STORE_PATH.exists(),
        ]
    )
    print("medai_vertex_semantic_package_review_decision_persist_15r_ready" if ready else "medai_vertex_semantic_package_review_decision_persist_15r_not_ready")
    print(
        json.dumps(
            {
                "decision_records_created": summary["decision_records_created"],
                "accept": summary["accept_decision_records_created"],
                "reject": summary["reject_decision_records_created"],
                "defer": summary["defer_decision_records_created"],
                "invalid_action_rejected_count": summary["invalid_action_rejected_count"],
                "active_mkb_record_created_count": summary["active_mkb_record_created_count"],
                "active_written_count": summary["active_written_count"],
                "auto_accept_true_count": summary["auto_accept_true_count"],
                "hallucinated_field_count": summary["hallucinated_field_count"],
                "live_call_made": summary["live_call_made"],
                "external_api_used": summary["external_api_used"],
                "privacy_result": summary["privacy_result"],
                "billing_check_pending": summary["billing_check_pending"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
