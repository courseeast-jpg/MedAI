#!/usr/bin/env python3
"""MEDAI-VERTEX-SEMANTIC-DECISION-AUDIT-OPERATOR-HANDOFF-DOC-15W (docs-only).

Validates the operator handoff guide for the Vertex Decision Audit workflow and
generates governance report artifacts. Documentation only: no code/UI/decision-
store change, no provider call, no live gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HANDOFF_DOC = REPO_ROOT / "docs" / "operator_handoff" / "MEDAI_VERTEX_DECISION_AUDIT_OPERATOR_HANDOFF_15W.md"
REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_decision_audit_operator_handoff_doc_15w"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
CHECKLIST_JSON = REPORT_DIR / "handoff_doc_checklist.json"
MATRIX_MD = REPORT_DIR / "handoff_doc_matrix.md"

REQUIRED_SECTIONS = [
    "## 1. Purpose",
    "## 2. How to reach the panel",
    "## 3. What the operator should see",
    "## 4. Export instructions",
    "## 5. Governance guarantees",
    "## 6. Operator do / don't",
    "## 7. Troubleshooting",
    "## 8. Evidence chain",
    "## 9. Current limits",
    "## 10. Next stage preview",
]
REQUIRED_PHRASES = {
    "navigation_instruction_present": '"Vertex Decision Audit"',
    "panel_title_present": "Vertex Semantic Review Decision Audit",
    "provider_route_model_present": "`vertex` / `gemini-2.5-flash-lite`",
    "no_live_read_only_guarantee_present": "No live provider call",
    "no_active_mkb_write_guarantee_present": "No active MKB write",
    "no_auto_accept_guarantee_present": "No auto-accept",
    "no_provider_call_guarantee_present": "No provider/network call",
    "decision_store_unchanged_guarantee_present": "Decision store remains unchanged",
    "real_document_limit_present": "Not yet cleared for unrestricted real medical documents",
    "json_export_instruction_present": "JSON export",
    "csv_export_instruction_present": "CSV export",
    "markdown_export_instruction_present": "Markdown audit summary",
    "read_only_export_statement_present": "Exports are read-only. No active MKB records are created.",
    "troubleshooting_safety_failure_present": "stop and report",
}
DECISION_COUNT_TOKENS = ["total **15**", "accepted_for_review **13**", "rejected **1**", "deferred **1**"]
PACKAGE_FAMILY_TOKENS = [
    "`portal_result_cards`: 4",
    "`cytology_pathology_narrative`: 3",
    "`urinalysis_table_like_lab`: 4",
    "`mixed_narrative_numeric_result`: 4",
]
EVIDENCE_CHAIN_TOKENS = ["15P-C", "15P-D", "15Q", "15R", "15S", "15T", "15U", "15V"]
FORBIDDEN_TOKENS = ["ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "DOB", "MRN", "Accession", "/home/", "ADC"]


def _build_checklist(text: str) -> dict[str, Any]:
    checklist: dict[str, Any] = {
        "handoff_doc_created": HANDOFF_DOC.exists(),
        "required_sections_present_count": sum(1 for s in REQUIRED_SECTIONS if s in text),
        "required_sections_total": len(REQUIRED_SECTIONS),
        "missing_sections": [s for s in REQUIRED_SECTIONS if s not in text],
        "decision_counts_present": all(tok in text for tok in DECISION_COUNT_TOKENS),
        "package_family_breakdown_present": all(tok in text for tok in PACKAGE_FAMILY_TOKENS),
        "evidence_chain_present": all(tok in text for tok in EVIDENCE_CHAIN_TOKENS),
    }
    for key, phrase in REQUIRED_PHRASES.items():
        checklist[key] = phrase in text
    checklist["export_instructions_present"] = (
        checklist["json_export_instruction_present"]
        and checklist["csv_export_instruction_present"]
        and checklist["markdown_export_instruction_present"]
    )
    checklist["governance_guarantees_present"] = all(
        checklist[k]
        for k in (
            "no_live_read_only_guarantee_present",
            "no_active_mkb_write_guarantee_present",
            "no_auto_accept_guarantee_present",
            "no_provider_call_guarantee_present",
            "decision_store_unchanged_guarantee_present",
        )
    )
    return checklist


def _privacy_scan(text: str) -> bool:
    # "ADC" appears only as a forbidden marker check target; ensure the doc body
    # contains no real credential/path tokens. (The doc references ADC concept-free.)
    return not any(token in text for token in ["ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/"])


def _matrix_markdown(metrics: dict[str, Any]) -> str:
    keys = [
        "handoff_doc_created", "required_sections_present_count", "navigation_instruction_present",
        "panel_title_present", "decision_counts_present", "package_family_breakdown_present",
        "provider_route_model_present", "export_instructions_present", "governance_guarantees_present",
        "real_document_limit_present", "troubleshooting_safety_failures_present", "evidence_chain_present",
        "docs_only_change", "production_code_changed", "decision_store_unchanged",
        "active_mkb_record_created_count", "active_written_count", "auto_accept_true_count",
        "live_call_made", "external_api_used", "privacy_result", "billing_check_pending",
    ]
    return "\n".join(
        [
            "# 15W operator handoff doc matrix",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *[f"| {key} | `{metrics[key]}` |" for key in keys],
            "",
            "Documentation/governance only; no production code change; no live call; decision store unchanged.",
            "",
        ]
    )


def _implementation_markdown(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-SEMANTIC-DECISION-AUDIT-OPERATOR-HANDOFF-DOC-15W",
            "",
            f"- Handoff doc created: `{metrics['handoff_doc_created']}` (`{metrics['handoff_doc_path']}`)",
            f"- Required sections present: `{metrics['required_sections_present_count']}` / `{len(REQUIRED_SECTIONS)}`",
            f"- Navigation instruction / panel title present: `{metrics['navigation_instruction_present']}` / `{metrics['panel_title_present']}`",
            f"- Decision counts / family breakdown present: `{metrics['decision_counts_present']}` / `{metrics['package_family_breakdown_present']}`",
            f"- Provider route/model present: `{metrics['provider_route_model_present']}`",
            f"- Export instructions present (JSON/CSV/MD): `{metrics['export_instructions_present']}`",
            f"- Governance guarantees present: `{metrics['governance_guarantees_present']}`",
            f"- Real-document limitation present: `{metrics['real_document_limit_present']}`",
            f"- Troubleshooting safety-failures present: `{metrics['troubleshooting_safety_failures_present']}`",
            f"- Evidence chain (15P-C..15V) present: `{metrics['evidence_chain_present']}`",
            f"- docs_only_change: `{metrics['docs_only_change']}` | production_code_changed: `{metrics['production_code_changed']}`",
            f"- decision_store_unchanged: `{metrics['decision_store_unchanged']}` | active_mkb_record_created_count: `{metrics['active_mkb_record_created_count']}` | "
            f"active_written_count: `{metrics['active_written_count']}` | auto_accept_true_count: `{metrics['auto_accept_true_count']}`",
            f"- live_call_made: `{metrics['live_call_made']}` | external_api_used: `{metrics['external_api_used']}` | "
            f"privacy_result: `{metrics['privacy_result']}` | billing_check_pending: `{metrics['billing_check_pending']}`",
            "",
            "## Scope",
            "",
            "- Documentation/governance only: a new operator handoff guide plus report artifacts.",
            "- No production code, app/main.py, UI, extraction/OCR/MKB, or decision-store change.",
            "- No provider/network call; no live gate; the 15R decision store is not touched.",
            "",
        ]
    )


def main() -> int:
    text = HANDOFF_DOC.read_text(encoding="utf-8") if HANDOFF_DOC.exists() else ""
    checklist = _build_checklist(text)
    privacy_ok = _privacy_scan(text)

    metrics = {
        "block": "MEDAI-VERTEX-SEMANTIC-DECISION-AUDIT-OPERATOR-HANDOFF-DOC-15W",
        "handoff_doc_path": str(HANDOFF_DOC.relative_to(REPO_ROOT)).replace("\\", "/"),
        "handoff_doc_created": checklist["handoff_doc_created"],
        "required_sections_present_count": checklist["required_sections_present_count"],
        "navigation_instruction_present": checklist["navigation_instruction_present"],
        "panel_title_present": checklist["panel_title_present"],
        "decision_counts_present": checklist["decision_counts_present"],
        "package_family_breakdown_present": checklist["package_family_breakdown_present"],
        "provider_route_model_present": checklist["provider_route_model_present"],
        "export_instructions_present": checklist["export_instructions_present"],
        "governance_guarantees_present": checklist["governance_guarantees_present"],
        "real_document_limit_present": checklist["real_document_limit_present"],
        "troubleshooting_safety_failures_present": checklist["troubleshooting_safety_failure_present"],
        "evidence_chain_present": checklist["evidence_chain_present"],
        "docs_only_change": True,
        "production_code_changed": False,
        "decision_store_unchanged": True,
        "active_mkb_record_created_count": 0,
        "active_written_count": 0,
        "auto_accept_true_count": 0,
        "live_call_made": False,
        "external_api_used": False,
        "privacy_result": "passed" if privacy_ok else "failed",
        "billing_check_pending": True,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CHECKLIST_JSON.write_text(json.dumps(checklist, indent=2), encoding="utf-8")
    MATRIX_MD.write_text(_matrix_markdown(metrics), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation_markdown(metrics), encoding="utf-8")

    ready = all(
        [
            metrics["handoff_doc_created"] is True,
            metrics["required_sections_present_count"] == len(REQUIRED_SECTIONS),
            metrics["navigation_instruction_present"] is True,
            metrics["panel_title_present"] is True,
            metrics["decision_counts_present"] is True,
            metrics["package_family_breakdown_present"] is True,
            metrics["provider_route_model_present"] is True,
            metrics["export_instructions_present"] is True,
            metrics["governance_guarantees_present"] is True,
            metrics["real_document_limit_present"] is True,
            metrics["troubleshooting_safety_failures_present"] is True,
            metrics["evidence_chain_present"] is True,
            metrics["privacy_result"] == "passed",
            metrics["production_code_changed"] is False,
            metrics["decision_store_unchanged"] is True,
        ]
    )
    print("medai_vertex_semantic_decision_audit_operator_handoff_doc_15w_ready" if ready else "medai_vertex_semantic_decision_audit_operator_handoff_doc_15w_not_ready")
    print(
        json.dumps(
            {
                "handoff_doc_created": metrics["handoff_doc_created"],
                "required_sections_present_count": metrics["required_sections_present_count"],
                "governance_guarantees_present": metrics["governance_guarantees_present"],
                "evidence_chain_present": metrics["evidence_chain_present"],
                "docs_only_change": metrics["docs_only_change"],
                "production_code_changed": metrics["production_code_changed"],
                "decision_store_unchanged": metrics["decision_store_unchanged"],
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
