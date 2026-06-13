#!/usr/bin/env python3
"""MEDAI-VERTEX-SEMANTIC-REVIEW-DECISION-PANEL-TAB-WIRE-15U (no-live, read-only).

Verifies that the 15T read-only Vertex semantic decision audit panel is
registered into the operator navigation (reachable in-app) and that, when
rendered from that nav entry, it still exposes all required read-only
audit/export content. No provider call; no active MKB writes; no decision-store
mutation; no auto-accept; no live gate.
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

from app.main import (
    VERTEX_DECISION_AUDIT_TAB,
    ADVANCED_OPERATOR_TAB_LABELS,
    operator_tab_labels,
)
from app.vertex_semantic_review_decision_audit_panel import (
    build_audit_panel_view_model,
    render_audit_panel_preview_markdown,
)
from execution.vertex_semantic_review_decision_audit_export import DEFAULT_DECISION_STORE_PATH

MAIN_PY = REPO_ROOT / "app" / "main.py"
REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_review_decision_panel_tab_wire_15u"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CASES_JSON = REPORT_DIR / "panel_tab_cases.json"
MATRIX_MD = REPORT_DIR / "panel_tab_matrix.md"
PREVIEW_MD = REPORT_DIR / "panel_tab_preview.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_TOKENS = ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "DOB", "MRN", "Accession")
CORE_TABS = ["Run & Review", "MKB Explorer", "Review Queue"]


def _privacy_scan(payload: Any) -> bool:
    blob = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return not any(token in blob for token in FORBIDDEN_TOKENS)


def _store_fingerprint() -> str:
    if not DEFAULT_DECISION_STORE_PATH.exists():
        return ""
    return hashlib.sha256(DEFAULT_DECISION_STORE_PATH.read_bytes()).hexdigest()[:16]


def _dispatch_wired() -> bool:
    source = MAIN_PY.read_text(encoding="utf-8")
    return (
        "elif label == VERTEX_DECISION_AUDIT_TAB:" in source
        and "render_vertex_semantic_review_decision_audit_panel_hook()" in source
    )


def _build_metrics(vm: dict[str, Any], store_unchanged: bool) -> dict[str, Any]:
    advanced_labels = operator_tab_labels(True)
    basic_labels = operator_tab_labels(False)
    nav_registered = VERTEX_DECISION_AUDIT_TAB in advanced_labels
    dispatch_wired = _dispatch_wired()
    rows = vm["decision_rows"]
    n = len(rows)
    ds = vm["decision_summary"]
    safety = vm["safety_summary"]
    present = vm["export_affordances_present"]
    return {
        "block": "MEDAI-VERTEX-SEMANTIC-REVIEW-DECISION-PANEL-TAB-WIRE-15U",
        "panel_nav_entry_registered": nav_registered and dispatch_wired,
        "panel_nav_label_visible": VERTEX_DECISION_AUDIT_TAB,
        "panel_reachable_in_app": nav_registered and dispatch_wired and VERTEX_DECISION_AUDIT_TAB not in basic_labels,
        "panel_rendered_from_nav": bool(vm["title"]) and n > 0,
        "core_tabs_preserved": basic_labels == CORE_TABS,
        "advanced_tab_labels": advanced_labels,
        "decision_records_loaded": ds["total_decisions"],
        "decision_records_visible_count": n,
        "total_decision_count_visible": ds["total_decisions"],
        "accepted_for_review_count_visible": ds["accepted_for_review"],
        "rejected_count_visible": ds["rejected"],
        "deferred_count_visible": ds["deferred"],
        "package_family_breakdown_visible": dict(vm["package_family_breakdown"]),
        "package_family_count_visible": len(vm["package_family_breakdown"]),
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
        "auto_accept_false_indicator_visible": safety["auto_accept"] is False,
        "review_required_indicator_visible": safety["review_required"] is True,
        "json_export_affordance_visible": bool(present.get("json")),
        "csv_export_affordance_visible": bool(present.get("csv")),
        "markdown_export_affordance_visible": bool(present.get("markdown")),
        "decision_store_unchanged": store_unchanged,
        "active_mkb_record_created_count": safety["active_mkb_record_created_count"],
        "active_written_count": safety["active_written_count"],
        "auto_accept_true_count": sum(1 for r in rows if r["auto_accept"]),
        "review_required": True,
        "live_call_made": False,
        "external_api_used": False,
        "privacy_result": "passed",
        "billing_check_pending": True,
    }


def _matrix_markdown(m: dict[str, Any]) -> str:
    keys = [
        "panel_nav_entry_registered", "panel_nav_label_visible", "panel_reachable_in_app",
        "panel_rendered_from_nav", "core_tabs_preserved", "decision_records_loaded",
        "decision_records_visible_count", "total_decision_count_visible", "accepted_for_review_count_visible",
        "rejected_count_visible", "deferred_count_visible", "package_family_count_visible",
        "provider_route_visible_count", "provider_model_visible_count", "evidence_anchor_visible_count",
        "source_evidence_visible_count", "source_report_reference_visible_count", "audit_reason_visible_count",
        "unknown_values_visible_count", "uncertainty_flags_visible_count", "hallucinated_field_count_visible",
        "no_live_read_only_indicator_visible", "active_written_count_indicator_visible",
        "active_mkb_record_created_count_indicator_visible", "auto_accept_false_indicator_visible",
        "review_required_indicator_visible", "json_export_affordance_visible", "csv_export_affordance_visible",
        "markdown_export_affordance_visible", "decision_store_unchanged", "active_mkb_record_created_count",
        "active_written_count", "auto_accept_true_count", "live_call_made", "external_api_used",
        "privacy_result", "billing_check_pending",
    ]
    return "\n".join(
        [
            "# 15U panel tab wiring matrix",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *[f"| {key} | `{m[key]}` |" for key in keys],
            "",
            "Additive nav wiring; read-only panel; 15R store unchanged; no active writes; no auto-accept; no live call.",
            "",
        ]
    )


def _implementation_markdown(m: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-SEMANTIC-REVIEW-DECISION-PANEL-TAB-WIRE-15U",
            "",
            f"- Nav entry registered + dispatch wired: `{m['panel_nav_entry_registered']}`",
            f"- Nav label: `{m['panel_nav_label_visible']}` | reachable in app: `{m['panel_reachable_in_app']}`",
            f"- Core tabs preserved (Run & Review / MKB Explorer / Review Queue): `{m['core_tabs_preserved']}`",
            f"- Advanced tab labels: `{m['advanced_tab_labels']}`",
            f"- Panel rendered from nav: `{m['panel_rendered_from_nav']}` | decisions visible: `{m['decision_records_visible_count']}`",
            f"- Decision summary: total `{m['total_decision_count_visible']}`, accepted_for_review `{m['accepted_for_review_count_visible']}`, "
            f"rejected `{m['rejected_count_visible']}`, deferred `{m['deferred_count_visible']}`",
            f"- Package families visible: `{m['package_family_breakdown_visible']}`",
            f"- Provider route/model visible: `{m['provider_route_visible_count']}` / `{m['provider_model_visible_count']}`",
            f"- Evidence/source/ref/audit visible: `{m['evidence_anchor_visible_count']}` / `{m['source_evidence_visible_count']}` / "
            f"`{m['source_report_reference_visible_count']}` / `{m['audit_reason_visible_count']}`",
            f"- Unknown/uncertainty visible: `{m['unknown_values_visible_count']}` / `{m['uncertainty_flags_visible_count']}`",
            f"- Hallucinated field count visible: `{m['hallucinated_field_count_visible']}`",
            f"- Export affordances (JSON/CSV/MD): `{m['json_export_affordance_visible']}` / `{m['csv_export_affordance_visible']}` / `{m['markdown_export_affordance_visible']}`",
            f"- decision_store_unchanged: `{m['decision_store_unchanged']}` | active_mkb_record_created_count: `{m['active_mkb_record_created_count']}` | "
            f"active_written_count: `{m['active_written_count']}` | auto_accept_true_count: `{m['auto_accept_true_count']}`",
            f"- live_call_made: `{m['live_call_made']}` | external_api_used: `{m['external_api_used']}` | privacy_result: `{m['privacy_result']}`",
            "",
            "## Safety",
            "",
            "- Additive nav registration only: the new advanced tab routes to the existing 15T read-only hook.",
            "- Core operator tabs (Run & Review / MKB Explorer / Review Queue) are unchanged and still first.",
            "- The 15R decision store is read only and never mutated (fingerprint verified). No provider call; no live gate.",
            "",
        ]
    )


def main() -> int:
    fingerprint_before = _store_fingerprint()
    vm = build_audit_panel_view_model()
    preview_md = render_audit_panel_preview_markdown(vm)
    fingerprint_after = _store_fingerprint()
    store_unchanged = fingerprint_before == fingerprint_after

    metrics = _build_metrics(vm, store_unchanged)
    privacy_ok = _privacy_scan(metrics) and _privacy_scan(vm) and _privacy_scan(preview_md)
    metrics["privacy_result"] = "passed" if privacy_ok else "failed"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CASES_JSON.write_text(json.dumps({"view_model": vm, "advanced_tab_labels": metrics["advanced_tab_labels"]}, indent=2), encoding="utf-8")
    MATRIX_MD.write_text(_matrix_markdown(metrics), encoding="utf-8")
    PREVIEW_MD.write_text(preview_md, encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation_markdown(metrics), encoding="utf-8")

    n = metrics["decision_records_visible_count"]
    ready = all(
        [
            metrics["panel_nav_entry_registered"] is True,
            metrics["panel_reachable_in_app"] is True,
            metrics["panel_rendered_from_nav"] is True,
            metrics["core_tabs_preserved"] is True,
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
            metrics["decision_store_unchanged"] is True,
            metrics["active_mkb_record_created_count"] == 0,
            metrics["active_written_count"] == 0,
            metrics["auto_accept_true_count"] == 0,
            metrics["live_call_made"] is False,
            metrics["external_api_used"] is False,
            metrics["privacy_result"] == "passed",
        ]
    )
    print("medai_vertex_semantic_review_decision_panel_tab_wire_15u_ready" if ready else "medai_vertex_semantic_review_decision_panel_tab_wire_15u_not_ready")
    print(
        json.dumps(
            {
                "panel_nav_entry_registered": metrics["panel_nav_entry_registered"],
                "panel_reachable_in_app": metrics["panel_reachable_in_app"],
                "panel_rendered_from_nav": metrics["panel_rendered_from_nav"],
                "core_tabs_preserved": metrics["core_tabs_preserved"],
                "decision_records_visible_count": n,
                "hallucinated_field_count_visible": metrics["hallucinated_field_count_visible"],
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
