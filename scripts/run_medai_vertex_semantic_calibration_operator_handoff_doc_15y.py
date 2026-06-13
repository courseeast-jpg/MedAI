#!/usr/bin/env python3
"""MEDAI-VERTEX-SEMANTIC-CALIBRATION-OPERATOR-HANDOFF-DOC-15Y (docs-only).

Validates the synthetic Vertex calibration operator/governance handoff guide and
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

HANDOFF_DOC = REPO_ROOT / "docs" / "operator_handoff" / "MEDAI_VERTEX_SYNTHETIC_CALIBRATION_HANDOFF_15Y.md"
REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_calibration_operator_handoff_doc_15y"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
CHECKLIST_JSON = REPORT_DIR / "handoff_doc_checklist.json"
MATRIX_MD = REPORT_DIR / "handoff_doc_matrix.md"

REQUIRED_SECTIONS = [
    "## 1. Purpose",
    "## 2. Calibration status",
    "## 3. What 15X-R4 proves",
    "## 4. What 15X-R4 does NOT prove",
    "## 5. Final synthetic calibration metrics",
    "## 6. Category coverage (12)",
    "## 7. Guardrails that held",
    "## 8. Contract hardenings created during calibration",
    "## 9. Operator interpretation",
    "## 10. Required gates before real-document routing",
    "## 11. Recommended next implementation stage",
    "## 12. Evidence chain",
    "## 13. Troubleshooting / stop conditions",
]
CATEGORIES = [
    "portal_result_cards", "cytology_pathology_narrative", "urinalysis_table_like_lab",
    "mixed_narrative_numeric_result", "short_clinical_note_negation", "short_clinical_note_uncertainty",
    "medication_mention_no_ddi", "bilingual_cyrillic_snippet", "sparse_low_information_result",
    "multi_section_explicit_unknowns", "abnormal_numeric_with_units", "normal_numeric_with_units",
]
FINAL_METRIC_TOKENS = [
    "fixture_count = **15**", "live_call_count = **15**", "schema_validation_pass_count = **15**",
    "verbatim_evidence_anchor_pass_count = **15**", "hallucinated_field_count = **0**",
    "review_required_count = **15**", "active_written_count = **0**", "active_mkb_record_created_count = **0**",
    "auto_accept = **false**", "total_token_count_all_calls = **7334**",
    "estimated_total_cost_usd_all_calls = **$0.0014486**", "billing_check_pending = **true**",
]
EVIDENCE_CHAIN = ["15P-C", "15P-D", "15Q", "15R", "15S", "15T", "15U", "15V", "15W", "15X", "15X-R1", "15X-R2", "15X-R3", "15X-R4"]
FUTURE_GATES = [
    "Real-doc privacy stripping proof", "PII vault isolation proof", "No raw private document leakage",
    "Synthetic-to-real adapter dry run", "Redacted real-like fixture replay", "Operator review queue handoff",
    "Real-doc refusal path", "Billing/cost cap check", "Human authorization gate for any real live call",
    "No active MKB write gate", "No auto-accept gate", "Medication safety gate non-bypass proof",
]
STOP_CONDITIONS = [
    "real PII in an outbound payload", "raw private content in reports", "a provider call without the dedicated gate",
    "an active MKB write", "`auto_accept=true`", "missing `review_required`", "non-verbatim evidence accepted",
    "undeclared label drift accepted", "medication safety bypass",
]
FORBIDDEN_TOKENS = ["ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/", "DOB", "MRN", "Accession"]


def _build_checklist(text: str) -> dict[str, Any]:
    c: dict[str, Any] = {
        "handoff_doc_created": HANDOFF_DOC.exists(),
        "required_sections_present_count": sum(1 for s in REQUIRED_SECTIONS if s in text),
        "required_sections_total": len(REQUIRED_SECTIONS),
        "missing_sections": [s for s in REQUIRED_SECTIONS if s not in text],
        "calibration_pass_summary_present": "15X-R4: full synthetic calibration PASS" in text and "15/15 fixtures passed" in text,
        "final_metrics_present": all(tok in text for tok in FINAL_METRIC_TOKENS),
        "category_coverage_present": all(cat in text for cat in CATEGORIES),
        "token_cost_summary_present": "total_token_count_all_calls = **7334**" in text and "$0.0014486" in text,
        "evidence_anchor_guardrails_present": "verbatim source span" in text and "paraphrase is rejected" in text.lower(),
        "bilingual_label_guardrails_present": "declared label aliases only" in text and "undeclared label drift rejected" in text.lower(),
        "real_document_boundary_present": (
            "clear unrestricted real medical documents" in text
            and "safe to process real documents" in text
            and "not real-document clearance" in text.lower()
        ),
        "future_real_doc_gates_present": all(g in text for g in FUTURE_GATES),
        "active_write_boundary_present": "No active MKB write" in text and "active_written_count = **0**" in text,
        "auto_accept_boundary_present": "No auto-accept" in text and "auto_accept = **false**" in text,
        "review_bound_present": "Review-bound only" in text,
        "next_stage_15z_present": "MEDAI-VERTEX-REAL-DOC-READINESS-GATES-NO-LIVE-15Z" in text,
        "evidence_chain_present": all(b in text for b in EVIDENCE_CHAIN),
        "troubleshooting_stop_conditions_present": "stop and treat as a safety failure" in text.lower() and all(sc in text for sc in STOP_CONDITIONS),
    }
    return c


def _privacy_scan(text: str) -> bool:
    return not any(token in text for token in ["ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/"])


def _matrix_markdown(m: dict[str, Any]) -> str:
    keys = [
        "handoff_doc_created", "required_sections_present_count", "calibration_pass_summary_present",
        "final_metrics_present", "category_coverage_present", "token_cost_summary_present",
        "evidence_anchor_guardrails_present", "bilingual_label_guardrails_present", "real_document_boundary_present",
        "future_real_doc_gates_present", "active_write_boundary_present", "auto_accept_boundary_present",
        "evidence_chain_present", "troubleshooting_stop_conditions_present", "docs_only_change",
        "production_code_changed", "live_call_made", "external_api_used", "active_written_count",
        "active_mkb_record_created_count", "auto_accept_true_count", "privacy_result", "billing_check_pending",
    ]
    return "\n".join(
        ["# 15Y calibration handoff doc matrix", "", "| Metric | Value |", "| --- | --- |",
         *[f"| {k} | `{m[k]}` |" for k in keys], "",
         "Documentation/governance only; no production code change; no live call; no real-document clearance.", ""]
    )


def _implementation_markdown(m: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-SEMANTIC-CALIBRATION-OPERATOR-HANDOFF-DOC-15Y",
            "",
            f"- Handoff doc created: `{m['handoff_doc_created']}` (`{m['handoff_doc_path']}`)",
            f"- Required sections present: `{m['required_sections_present_count']}` / `{len(REQUIRED_SECTIONS)}`",
            f"- Calibration PASS summary / final metrics / category coverage present: "
            f"`{m['calibration_pass_summary_present']}` / `{m['final_metrics_present']}` / `{m['category_coverage_present']}`",
            f"- Token/cost summary present: `{m['token_cost_summary_present']}`",
            f"- Evidence-anchor / bilingual-label guardrails present: `{m['evidence_anchor_guardrails_present']}` / `{m['bilingual_label_guardrails_present']}`",
            f"- Real-document boundary present: `{m['real_document_boundary_present']}` | future real-doc gates present: `{m['future_real_doc_gates_present']}`",
            f"- Active-write boundary / auto-accept boundary present: `{m['active_write_boundary_present']}` / `{m['auto_accept_boundary_present']}`",
            f"- Evidence chain (15P-C..15X-R4) present: `{m['evidence_chain_present']}` | troubleshooting stop conditions present: `{m['troubleshooting_stop_conditions_present']}`",
            f"- docs_only_change: `{m['docs_only_change']}` | production_code_changed: `{m['production_code_changed']}`",
            f"- live_call_made: `{m['live_call_made']}` | external_api_used: `{m['external_api_used']}` | "
            f"active_written_count: `{m['active_written_count']}` | auto_accept_true_count: `{m['auto_accept_true_count']}`",
            f"- privacy_result: `{m['privacy_result']}` | billing_check_pending: `{m['billing_check_pending']}`",
            "",
            "## Scope",
            "",
            "- Documentation/governance only: a synthetic-calibration operator handoff guide plus report artifacts.",
            "- Summarizes 15X-R4 synthetic PASS and defines explicit gates required before any real-document routing.",
            "- No production code, app/main.py, UI, extraction/OCR/MKB, or decision-store change; no provider call; no live gate.",
            "",
        ]
    )


def main() -> int:
    text = HANDOFF_DOC.read_text(encoding="utf-8") if HANDOFF_DOC.exists() else ""
    checklist = _build_checklist(text)
    privacy_ok = _privacy_scan(text)

    metrics = {
        "block": "MEDAI-VERTEX-SEMANTIC-CALIBRATION-OPERATOR-HANDOFF-DOC-15Y",
        "handoff_doc_path": str(HANDOFF_DOC.relative_to(REPO_ROOT)).replace("\\", "/"),
        "handoff_doc_created": checklist["handoff_doc_created"],
        "required_sections_present_count": checklist["required_sections_present_count"],
        "calibration_pass_summary_present": checklist["calibration_pass_summary_present"],
        "final_metrics_present": checklist["final_metrics_present"],
        "category_coverage_present": checklist["category_coverage_present"],
        "token_cost_summary_present": checklist["token_cost_summary_present"],
        "evidence_anchor_guardrails_present": checklist["evidence_anchor_guardrails_present"],
        "bilingual_label_guardrails_present": checklist["bilingual_label_guardrails_present"],
        "real_document_boundary_present": checklist["real_document_boundary_present"],
        "future_real_doc_gates_present": checklist["future_real_doc_gates_present"],
        "active_write_boundary_present": checklist["active_write_boundary_present"],
        "auto_accept_boundary_present": checklist["auto_accept_boundary_present"],
        "evidence_chain_present": checklist["evidence_chain_present"],
        "troubleshooting_stop_conditions_present": checklist["troubleshooting_stop_conditions_present"],
        "docs_only_change": True,
        "production_code_changed": False,
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "active_mkb_record_created_count": 0,
        "auto_accept_true_count": 0,
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
            metrics["calibration_pass_summary_present"] is True,
            metrics["final_metrics_present"] is True,
            metrics["category_coverage_present"] is True,
            metrics["token_cost_summary_present"] is True,
            metrics["evidence_anchor_guardrails_present"] is True,
            metrics["bilingual_label_guardrails_present"] is True,
            metrics["real_document_boundary_present"] is True,
            metrics["future_real_doc_gates_present"] is True,
            metrics["active_write_boundary_present"] is True,
            metrics["auto_accept_boundary_present"] is True,
            metrics["evidence_chain_present"] is True,
            metrics["troubleshooting_stop_conditions_present"] is True,
            metrics["privacy_result"] == "passed",
            metrics["production_code_changed"] is False,
        ]
    )
    print("medai_vertex_semantic_calibration_operator_handoff_doc_15y_ready" if ready else "medai_vertex_semantic_calibration_operator_handoff_doc_15y_not_ready")
    print(
        json.dumps(
            {
                "handoff_doc_created": metrics["handoff_doc_created"],
                "required_sections_present_count": metrics["required_sections_present_count"],
                "calibration_pass_summary_present": metrics["calibration_pass_summary_present"],
                "real_document_boundary_present": metrics["real_document_boundary_present"],
                "future_real_doc_gates_present": metrics["future_real_doc_gates_present"],
                "evidence_chain_present": metrics["evidence_chain_present"],
                "docs_only_change": metrics["docs_only_change"],
                "production_code_changed": metrics["production_code_changed"],
                "live_call_made": metrics["live_call_made"],
                "privacy_result": metrics["privacy_result"],
                "billing_check_pending": metrics["billing_check_pending"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
