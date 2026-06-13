#!/usr/bin/env python3
"""MEDAI-VERTEX-SEMANTIC-REVIEW-DECISION-AUDIT-EXPORT-15S (no-live, read-only).

Builds a read-only operator audit/export over the isolated 15R review-decision
store and writes JSON / CSV / markdown audit artifacts. The 15R store is read
only and never mutated; no provider call; no active MKB writes; no auto-accept;
no live gate.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.vertex_semantic_review_decision_audit_export import (
    DEFAULT_DECISION_STORE_PATH,
    build_audit_export,
)

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_review_decision_audit_export_15s"
SUMMARY_JSON = REPORT_DIR / "summary.json"
EXPORT_JSON = REPORT_DIR / "decision_audit_export.json"
EXPORT_CSV = REPORT_DIR / "decision_audit_export.csv"
SUMMARY_MD = REPORT_DIR / "decision_audit_summary.md"
MATRIX_MD = REPORT_DIR / "decision_audit_matrix.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_TOKENS = ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "DOB", "MRN", "Accession")


def _privacy_scan(payload: Any) -> bool:
    blob = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return not any(token in blob for token in FORBIDDEN_TOKENS)


def _store_fingerprint() -> str:
    if not DEFAULT_DECISION_STORE_PATH.exists():
        return ""
    return hashlib.sha256(DEFAULT_DECISION_STORE_PATH.read_bytes()).hexdigest()[:16]


def _implementation_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-SEMANTIC-REVIEW-DECISION-AUDIT-EXPORT-15S",
            "",
            f"- Decision records loaded (read-only from 15R store): `{summary['decision_records_loaded']}`",
            f"- Exported JSON / CSV: `{summary['decision_records_exported_json']}` / `{summary['decision_records_exported_csv']}`",
            f"- Per-family: `{summary['package_family_breakdown']}`",
            f"- Per-action: accepted_for_review `{summary['accepted_for_review_count']}`, "
            f"rejected `{summary['rejected_count']}`, deferred `{summary['deferred_count']}`",
            f"- Provider provenance visible: `{summary['provider_provenance_visible_count']}` (vertex / gemini-2.5-flash-lite)",
            f"- Evidence anchors / unknowns / uncertainty / source refs / audit reasons visible: "
            f"`{summary['evidence_anchor_visible_count']}` / `{summary['unknown_values_visible_count']}` / "
            f"`{summary['uncertainty_flags_visible_count']}` / `{summary['source_report_reference_visible_count']}` / "
            f"`{summary['audit_reason_visible_count']}`",
            f"- Hallucinated field count: `{summary['hallucinated_field_count']}`",
            f"- export_read_only: `{summary['export_read_only']}` | active_mkb_record_created_count: `{summary['active_mkb_record_created_count']}` | "
            f"active_written_count: `{summary['active_written_count']}` | auto_accept_true_count: `{summary['auto_accept_true_count']}`",
            f"- review_required_true_count: `{summary['review_required_true_count']}` | "
            f"live_call_made: `{summary['live_call_made']}` | external_api_used: `{summary['external_api_used']}`",
            f"- privacy_result: `{summary['privacy_result']}` | billing_check_pending: `{summary['billing_check_pending']}`",
            "",
            "## Safety",
            "",
            "- Read-only audit/export; the 15R decision store JSONL is read and never mutated.",
            "- Exports (JSON/CSV/markdown) are written only to the separate 15S report directory.",
            "- No active MKB writes; no decision_status change; no auto-accept; no provider/network call; no live gate.",
            "",
        ]
    )


def main() -> int:
    fingerprint_before = _store_fingerprint()
    export = build_audit_export()
    summary = export.summary

    privacy_ok = (
        _privacy_scan(summary)
        and _privacy_scan(export.json_export)
        and _privacy_scan(export.csv_export)
        and _privacy_scan(export.audit_summary_markdown)
    )
    summary["privacy_result"] = "passed" if privacy_ok else "failed"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    EXPORT_JSON.write_text(json.dumps({"decision_records": export.json_export}, indent=2), encoding="utf-8")
    EXPORT_CSV.write_text(export.csv_export, encoding="utf-8")
    SUMMARY_MD.write_text(export.audit_summary_markdown, encoding="utf-8")
    MATRIX_MD.write_text(export.audit_matrix_markdown, encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation_markdown(summary), encoding="utf-8")

    # Confirm the 15R store was not mutated by the export.
    fingerprint_after = _store_fingerprint()
    store_unchanged = fingerprint_before == fingerprint_after

    ready = all(
        [
            summary["decision_records_loaded"] > 0,
            summary["decision_records_exported_json"] == summary["decision_records_loaded"],
            summary["decision_records_exported_csv"] == summary["decision_records_loaded"],
            summary["package_family_count"] == 4,
            summary["package_family_breakdown_present"] is True,
            summary["action_breakdown_present"] is True,
            summary["provider_provenance_visible_count"] == summary["decision_records_loaded"],
            summary["evidence_anchor_visible_count"] == summary["decision_records_loaded"],
            summary["uncertainty_flags_visible_count"] == summary["decision_records_loaded"],
            summary["source_report_reference_visible_count"] == summary["decision_records_loaded"],
            summary["audit_reason_visible_count"] == summary["decision_records_loaded"],
            summary["review_required_true_count"] == summary["decision_records_loaded"],
            summary["hallucinated_field_count"] == 0,
            summary["active_mkb_record_created_count"] == 0,
            summary["active_written_count"] == 0,
            summary["auto_accept_true_count"] == 0,
            summary["export_read_only"] is True,
            summary["all_records_review_bound"] is True,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["privacy_result"] == "passed",
            store_unchanged,
        ]
    )
    print("medai_vertex_semantic_review_decision_audit_export_15s_ready" if ready else "medai_vertex_semantic_review_decision_audit_export_15s_not_ready")
    print(
        json.dumps(
            {
                "decision_records_loaded": summary["decision_records_loaded"],
                "exported_json": summary["decision_records_exported_json"],
                "exported_csv": summary["decision_records_exported_csv"],
                "accepted_for_review_count": summary["accepted_for_review_count"],
                "rejected_count": summary["rejected_count"],
                "deferred_count": summary["deferred_count"],
                "hallucinated_field_count": summary["hallucinated_field_count"],
                "active_mkb_record_created_count": summary["active_mkb_record_created_count"],
                "active_written_count": summary["active_written_count"],
                "auto_accept_true_count": summary["auto_accept_true_count"],
                "export_read_only": summary["export_read_only"],
                "decision_store_unchanged": store_unchanged,
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
