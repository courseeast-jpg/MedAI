#!/usr/bin/env python3
"""MEDAI-VERTEX-SEMANTIC-PACKAGE-OPERATOR-REVIEW-SURFACE-15Q (no-live).

Replays the recorded, sanitized 15P-C / 15P-D live-comparison artifacts into
review-bound operator draft packages and generates operator review reports.
Makes NO provider call and sets NO live gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.vertex_semantic_review_surface import (
    build_operator_review_report,
    build_vertex_semantic_review_drafts,
    render_operator_review_preview,
)

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_package_operator_review_surface_15q"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CASES_JSON = REPORT_DIR / "operator_review_cases.json"
MATRIX_MD = REPORT_DIR / "operator_review_matrix.md"
PREVIEW_MD = REPORT_DIR / "operator_review_preview.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_TOKENS = ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "DOB", "MRN", "Accession")


def _privacy_scan(payload: Any) -> bool:
    blob = json.dumps(payload, default=str)
    return not any(token in blob for token in FORBIDDEN_TOKENS)


def _matrix_markdown(summary: dict[str, Any]) -> str:
    rows = [
        ("package_families_loaded", summary["package_families_loaded"]),
        ("package_families_rendered", summary["package_families_rendered"]),
        ("provider_route_visible_count", summary["provider_route_visible_count"]),
        ("provider_model_visible_count", summary["provider_model_visible_count"]),
        ("source_visible_body_present_count", summary["source_visible_body_present_count"]),
        ("evidence_anchor_present_count", summary["evidence_anchor_present_count"]),
        ("vertex_semantic_findings_separated_count", summary["vertex_semantic_findings_separated_count"]),
        ("candidate_facts_separated_count", summary["candidate_facts_separated_count"]),
        ("unknown_values_explicit_count", summary["unknown_values_explicit_count"]),
        ("uncertainty_flags_visible_count", summary["uncertainty_flags_visible_count"]),
        ("hallucinated_field_count", summary["hallucinated_field_count"]),
        ("no_live_replay_indicator_visible_count", summary["no_live_replay_indicator_visible_count"]),
        ("active_written_count_indicator_visible_count", summary["active_written_count_indicator_visible_count"]),
        ("auto_accept_false_indicator_visible_count", summary["auto_accept_false_indicator_visible_count"]),
        ("accept_control_visible_count", summary["accept_control_visible_count"]),
        ("reject_control_visible_count", summary["reject_control_visible_count"]),
        ("defer_control_visible_count", summary["defer_control_visible_count"]),
        ("accept_simulation_review_bound_count", summary["accept_simulation_review_bound_count"]),
        ("reject_simulation_local_only_count", summary["reject_simulation_local_only_count"]),
        ("defer_simulation_local_only_count", summary["defer_simulation_local_only_count"]),
        ("active_written_count", summary["active_written_count"]),
        ("auto_accept", summary["auto_accept"]),
        ("review_required", summary["review_required"]),
        ("live_call_made", summary["live_call_made"]),
        ("external_api_used", summary["external_api_used"]),
        ("privacy_result", summary["privacy_result"]),
        ("billing_check_pending", summary["billing_check_pending"]),
    ]
    return "\n".join(
        [
            "# 15Q operator review surface matrix",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *[f"| {name} | `{value}` |" for name, value in rows],
            "",
            "Replay-only (no live call); review-bound; no active writes; no auto-accept.",
            "",
        ]
    )


def _implementation_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-SEMANTIC-PACKAGE-OPERATOR-REVIEW-SURFACE-15Q",
            "",
            f"- Package families loaded: `{summary['package_families_loaded']}` | rendered: `{summary['package_families_rendered']}`",
            f"- Provider route/model visible: `{summary['provider_route_visible_count']}` / `{summary['provider_model_visible_count']}`",
            f"- Source body / evidence anchors present: `{summary['source_visible_body_present_count']}` / `{summary['evidence_anchor_present_count']}`",
            f"- Vertex findings / candidate facts separated: `{summary['vertex_semantic_findings_separated_count']}` / `{summary['candidate_facts_separated_count']}`",
            f"- Unknown explicit / uncertainty visible: `{summary['unknown_values_explicit_count']}` / `{summary['uncertainty_flags_visible_count']}`",
            f"- Hallucinated field count: `{summary['hallucinated_field_count']}`",
            f"- Accept/Reject/Defer controls visible: `{summary['accept_control_visible_count']}` / `{summary['reject_control_visible_count']}` / `{summary['defer_control_visible_count']}`",
            f"- Accept review-bound / Reject local / Defer local: `{summary['accept_simulation_review_bound_count']}` / `{summary['reject_simulation_local_only_count']}` / `{summary['defer_simulation_local_only_count']}`",
            f"- live_call_made: `{summary['live_call_made']}` | external_api_used: `{summary['external_api_used']}`",
            f"- active_written_count: `{summary['active_written_count']}` | auto_accept: `{summary['auto_accept']}` | review_required: `{summary['review_required']}`",
            f"- privacy_result: `{summary['privacy_result']}` | billing_check_pending: `{summary['billing_check_pending']}`",
            "",
            "## Safety",
            "",
            "- No live provider call; replays recorded 15P-C/15P-D sanitized artifacts only.",
            "- Operator accept/reject/defer are deterministic local simulations; no active MKB writes; no auto-accept.",
            "- Source body, sections, evidence anchors, unknowns, uncertainty, and review-required are all operator-visible.",
            "- No live provider gates are used or required.",
            "",
        ]
    )


def main() -> int:
    report = build_operator_review_report()
    summary = report["summary"]
    cases = report["cases"]
    action_sims = report["action_simulations"]
    preview_md = render_operator_review_preview(build_vertex_semantic_review_drafts())

    privacy_ok = _privacy_scan(summary) and _privacy_scan(cases) and _privacy_scan(action_sims) and _privacy_scan(preview_md)
    summary["privacy_result"] = "passed" if privacy_ok else "failed"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    CASES_JSON.write_text(json.dumps({"cases": cases, "action_simulations": action_sims}, indent=2), encoding="utf-8")
    MATRIX_MD.write_text(_matrix_markdown(summary), encoding="utf-8")
    PREVIEW_MD.write_text(preview_md, encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation_markdown(summary), encoding="utf-8")

    ready = all(
        [
            summary["package_families_rendered"] == 4,
            summary["all_required_families_rendered"] is True,
            summary["provider_route_visible_count"] == 4,
            summary["provider_model_visible_count"] == 4,
            summary["source_visible_body_present_count"] == 4,
            summary["evidence_anchor_present_count"] == 4,
            summary["vertex_semantic_findings_separated_count"] == 4,
            summary["candidate_facts_separated_count"] == 4,
            summary["uncertainty_flags_visible_count"] == 4,
            summary["hallucinated_field_count"] == 0,
            summary["accept_control_visible_count"] == 4,
            summary["reject_control_visible_count"] == 4,
            summary["defer_control_visible_count"] == 4,
            summary["accept_simulation_review_bound_count"] == 4,
            summary["reject_simulation_local_only_count"] == 4,
            summary["defer_simulation_local_only_count"] == 4,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_required"] is True,
            summary["privacy_result"] == "passed",
        ]
    )
    print("medai_vertex_semantic_package_operator_review_surface_15q_ready" if ready else "medai_vertex_semantic_package_operator_review_surface_15q_not_ready")
    print(
        json.dumps(
            {
                "package_families_loaded": summary["package_families_loaded"],
                "package_families_rendered": summary["package_families_rendered"],
                "hallucinated_field_count": summary["hallucinated_field_count"],
                "accept_simulation_review_bound_count": summary["accept_simulation_review_bound_count"],
                "live_call_made": summary["live_call_made"],
                "external_api_used": summary["external_api_used"],
                "active_written_count": summary["active_written_count"],
                "auto_accept": summary["auto_accept"],
                "review_required": summary["review_required"],
                "privacy_result": summary["privacy_result"],
                "billing_check_pending": summary["billing_check_pending"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
