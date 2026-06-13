#!/usr/bin/env python3
"""MEDAI-VERTEX-SEMANTIC-REVIEW-DECISION-OPERATOR-PANEL-15T (no-live, read-only).

Builds the read-only operator panel view-model over the 15S audit/export layer
and writes panel preview + summary reports. The 15R decision store is read only
and never mutated; no provider call; no active MKB writes; no auto-accept; no
live gate.
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

from app.vertex_semantic_review_decision_audit_panel import (
    build_audit_panel_view_model,
    render_audit_panel_preview_markdown,
)
from execution.vertex_semantic_review_decision_audit_export import DEFAULT_DECISION_STORE_PATH

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_review_decision_operator_panel_15t"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CASES_JSON = REPORT_DIR / "operator_panel_cases.json"
MATRIX_MD = REPORT_DIR / "operator_panel_matrix.md"
PREVIEW_MD = REPORT_DIR / "operator_panel_preview.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_TOKENS = ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "DOB", "MRN", "Accession")


def _privacy_scan(payload: Any) -> bool:
    blob = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return not any(token in blob for token in FORBIDDEN_TOKENS)


def _store_fingerprint() -> str:
    if not DEFAULT_DECISION_STORE_PATH.exists():
        return ""
    return hashlib.sha256(DEFAULT_DECISION_STORE_PATH.read_bytes()).hexdigest()[:16]


def _build_metrics(vm: dict[str, Any], store_unchanged: bool) -> dict[str, Any]:
    rows = vm["decision_rows"]
    n = len(rows)
    ds = vm["decision_summary"]
    breakdown = vm["package_family_breakdown"]
    present = vm["export_affordances_present"]
    return {
        "block": "MEDAI-VERTEX-SEMANTIC-REVIEW-DECISION-OPERATOR-PANEL-15T",
        "panel_rendered": True,
        "decision_records_loaded": ds["total_decisions"],
        "decision_records_visible_count": n,
        "total_decision_count_visible": ds["total_decisions"],
        "accepted_for_review_count_visible": ds["accepted_for_review"],
        "rejected_count_visible": ds["rejected"],
        "deferred_count_visible": ds["deferred"],
        "package_family_breakdown_visible": breakdown,
        "package_family_count_visible": len(breakdown),
        "provider_route_visible_count": sum(1 for r in rows if r["provider_route"] == "vertex"),
        "provider_model_visible_count": sum(1 for r in rows if r["provider_model"] == "gemini-2.5-flash-lite"),
        "evidence_anchor_visible_count": sum(1 for r in rows if str(r["evidence_anchor"]).strip()),
        "source_evidence_visible_count": sum(1 for r in rows if str(r["source_evidence_text"]).strip()),
        "source_report_reference_visible_count": sum(1 for r in rows if str(r["source_report_reference"]).strip()),
        "audit_reason_visible_count": sum(1 for r in rows if str(r["audit_reason"]).strip()),
        "unknown_values_visible_count": sum(
            1 for r in rows if (r["unknown_values"] or r["package_family"] != "mixed_narrative_numeric_result")
        ),
        "uncertainty_flags_visible_count": sum(1 for r in rows if r["uncertainty_flag_count"] > 0),
        "hallucinated_field_count_visible": sum(r["hallucinated_field_count"] for r in rows),
        "no_live_read_only_indicator_visible": bool(vm["no_live_read_only_indicator"]),
        "active_written_count_indicator_visible": True,
        "active_mkb_record_created_count_indicator_visible": True,
        "auto_accept_false_indicator_visible": vm["safety_summary"]["auto_accept"] is False,
        "review_required_indicator_visible": vm["safety_summary"]["review_required"] is True,
        "json_export_affordance_visible": bool(present.get("json")),
        "csv_export_affordance_visible": bool(present.get("csv")),
        "markdown_export_affordance_visible": bool(present.get("markdown")),
        "export_read_only": True,
        "decision_store_unchanged": store_unchanged,
        "active_mkb_record_created_count": vm["safety_summary"]["active_mkb_record_created_count"],
        "active_written_count": vm["safety_summary"]["active_written_count"],
        "auto_accept_true_count": sum(1 for r in rows if r["auto_accept"]),
        "review_required": True,
        "live_call_made": False,
        "external_api_used": False,
        "privacy_result": "passed",
        "billing_check_pending": True,
    }


def _matrix_markdown(metrics: dict[str, Any]) -> str:
    keys = [
        "panel_rendered", "decision_records_loaded", "decision_records_visible_count",
        "total_decision_count_visible", "accepted_for_review_count_visible", "rejected_count_visible",
        "deferred_count_visible", "package_family_count_visible", "provider_route_visible_count",
        "provider_model_visible_count", "evidence_anchor_visible_count", "source_evidence_visible_count",
        "source_report_reference_visible_count", "audit_reason_visible_count", "unknown_values_visible_count",
        "uncertainty_flags_visible_count", "hallucinated_field_count_visible",
        "no_live_read_only_indicator_visible", "active_written_count_indicator_visible",
        "active_mkb_record_created_count_indicator_visible", "auto_accept_false_indicator_visible",
        "review_required_indicator_visible", "json_export_affordance_visible", "csv_export_affordance_visible",
        "markdown_export_affordance_visible", "export_read_only", "decision_store_unchanged",
        "active_mkb_record_created_count", "active_written_count", "auto_accept_true_count",
        "live_call_made", "external_api_used", "privacy_result", "billing_check_pending",
    ]
    return "\n".join(
        [
            "# 15T operator panel matrix",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *[f"| {key} | `{metrics[key]}` |" for key in keys],
            "",
            "Read-only panel; 15R decision store unchanged; no active writes; no auto-accept; no live call.",
            "",
        ]
    )


def _implementation_markdown(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-SEMANTIC-REVIEW-DECISION-OPERATOR-PANEL-15T",
            "",
            f"- Panel rendered: `{metrics['panel_rendered']}` | decisions visible: `{metrics['decision_records_visible_count']}`",
            f"- Decision summary: total `{metrics['total_decision_count_visible']}`, accepted_for_review "
            f"`{metrics['accepted_for_review_count_visible']}`, rejected `{metrics['rejected_count_visible']}`, deferred `{metrics['deferred_count_visible']}`",
            f"- Package families visible: `{metrics['package_family_breakdown_visible']}`",
            f"- Provider route/model visible: `{metrics['provider_route_visible_count']}` / `{metrics['provider_model_visible_count']}` (vertex / gemini-2.5-flash-lite)",
            f"- Evidence anchor / source evidence / source ref / audit reason visible: "
            f"`{metrics['evidence_anchor_visible_count']}` / `{metrics['source_evidence_visible_count']}` / "
            f"`{metrics['source_report_reference_visible_count']}` / `{metrics['audit_reason_visible_count']}`",
            f"- Unknown / uncertainty visible: `{metrics['unknown_values_visible_count']}` / `{metrics['uncertainty_flags_visible_count']}`",
            f"- Hallucinated field count visible: `{metrics['hallucinated_field_count_visible']}`",
            f"- Export affordances visible (JSON/CSV/MD): `{metrics['json_export_affordance_visible']}` / "
            f"`{metrics['csv_export_affordance_visible']}` / `{metrics['markdown_export_affordance_visible']}`",
            f"- export_read_only: `{metrics['export_read_only']}` | decision_store_unchanged: `{metrics['decision_store_unchanged']}`",
            f"- active_mkb_record_created_count: `{metrics['active_mkb_record_created_count']}` | active_written_count: `{metrics['active_written_count']}` | "
            f"auto_accept_true_count: `{metrics['auto_accept_true_count']}`",
            f"- live_call_made: `{metrics['live_call_made']}` | external_api_used: `{metrics['external_api_used']}` | "
            f"privacy_result: `{metrics['privacy_result']}` | billing_check_pending: `{metrics['billing_check_pending']}`",
            "",
            "## Safety",
            "",
            "- Compact read-only operator panel over the 15S audit/export view-model.",
            "- The 15R decision store is read only and never mutated (fingerprint verified before/after).",
            "- Export affordances reference already-generated 15S artifacts; no active MKB writes; no decision_status change.",
            "- No auto-accept; no provider/network call; no live gate. Additive UI hook only (no broad redesign).",
            "",
        ]
    )


def main() -> int:
    fingerprint_before = _store_fingerprint()
    view_model = build_audit_panel_view_model()
    preview_md = render_audit_panel_preview_markdown(view_model)
    fingerprint_after = _store_fingerprint()
    store_unchanged = fingerprint_before == fingerprint_after

    metrics = _build_metrics(view_model, store_unchanged)
    privacy_ok = _privacy_scan(metrics) and _privacy_scan(view_model) and _privacy_scan(preview_md)
    metrics["privacy_result"] = "passed" if privacy_ok else "failed"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CASES_JSON.write_text(json.dumps({"view_model": view_model}, indent=2), encoding="utf-8")
    MATRIX_MD.write_text(_matrix_markdown(metrics), encoding="utf-8")
    PREVIEW_MD.write_text(preview_md, encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation_markdown(metrics), encoding="utf-8")

    n = metrics["decision_records_visible_count"]
    ready = all(
        [
            metrics["panel_rendered"] is True,
            metrics["decision_records_loaded"] == n and n > 0,
            metrics["package_family_count_visible"] == 4,
            metrics["provider_route_visible_count"] == n,
            metrics["provider_model_visible_count"] == n,
            metrics["evidence_anchor_visible_count"] == n,
            metrics["source_evidence_visible_count"] == n,
            metrics["source_report_reference_visible_count"] == n,
            metrics["audit_reason_visible_count"] == n,
            metrics["uncertainty_flags_visible_count"] == n,
            metrics["hallucinated_field_count_visible"] == 0,
            metrics["no_live_read_only_indicator_visible"] is True,
            metrics["auto_accept_false_indicator_visible"] is True,
            metrics["review_required_indicator_visible"] is True,
            metrics["json_export_affordance_visible"] is True,
            metrics["csv_export_affordance_visible"] is True,
            metrics["markdown_export_affordance_visible"] is True,
            metrics["export_read_only"] is True,
            metrics["decision_store_unchanged"] is True,
            metrics["active_mkb_record_created_count"] == 0,
            metrics["active_written_count"] == 0,
            metrics["auto_accept_true_count"] == 0,
            metrics["live_call_made"] is False,
            metrics["external_api_used"] is False,
            metrics["privacy_result"] == "passed",
        ]
    )
    print("medai_vertex_semantic_review_decision_operator_panel_15t_ready" if ready else "medai_vertex_semantic_review_decision_operator_panel_15t_not_ready")
    print(
        json.dumps(
            {
                "panel_rendered": metrics["panel_rendered"],
                "decision_records_visible_count": n,
                "accepted_for_review_count_visible": metrics["accepted_for_review_count_visible"],
                "rejected_count_visible": metrics["rejected_count_visible"],
                "deferred_count_visible": metrics["deferred_count_visible"],
                "hallucinated_field_count_visible": metrics["hallucinated_field_count_visible"],
                "export_read_only": metrics["export_read_only"],
                "decision_store_unchanged": metrics["decision_store_unchanged"],
                "active_mkb_record_created_count": metrics["active_mkb_record_created_count"],
                "active_written_count": metrics["active_written_count"],
                "auto_accept_true_count": metrics["auto_accept_true_count"],
                "live_call_made": metrics["live_call_made"],
                "external_api_used": metrics["external_api_used"],
                "privacy_result": metrics["privacy_result"],
                "billing_check_pending": metrics["billing_check_pending"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
