"""Phase70 full-corpus post-diagnostics decision audit.

Report-only consolidation of Phases 58-69. This script does not inspect raw
documents, private mappings, OCR text, extracted text, or PHI. It reads public
reports, summarizes branch status, and makes a deterministic next-action
recommendation.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["MEDAI_LOCAL_ONLY"] = "true"
os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ["MEDAI_REQUIRE_PII_SCRUB"] = "true"
os.environ["MEDAI_PRIVACY_AUDIT"] = "true"

import app.config as app_config
import privacy.outbound_gate as outbound_gate
from privacy.privacy_audit import phi_artifact_tracking_status, write_json


REPORT_DIR = ROOT / "reports" / "phase70_post_diagnostics_decision_audit"
JSON_REPORT = REPORT_DIR / "phase70_post_diagnostics_decision_audit_report.json"
MD_REPORT = REPORT_DIR / "phase70_post_diagnostics_decision_audit_report.md"

REPORT_INPUTS: dict[str, Path] = {
    "phase54_operator_review_feedback": ROOT / "reports" / "phase54_operator_review_feedback" / "phase54_operator_review_feedback_report.json",
    "phase58_stratified_problem_fix_plan": ROOT / "reports" / "phase58_stratified_problem_fix_plan" / "phase58_stratified_problem_fix_plan.json",
    "phase59_empty_extraction_forensics": ROOT / "reports" / "phase59_empty_extraction_forensics" / "phase59_empty_extraction_forensics_report.json",
    "phase60_text_extraction_gap_diagnostic": ROOT / "reports" / "phase60_text_extraction_gap_diagnostic" / "phase60_text_extraction_gap_diagnostic_report.json",
    "phase61_header_label_inference_diagnostic": ROOT / "reports" / "phase61_header_label_inference_diagnostic" / "phase61_header_label_inference_diagnostic_report.json",
    "phase62_table_geometry_header_inference_prototype": ROOT / "reports" / "phase62_table_geometry_header_inference_prototype" / "phase62_table_geometry_header_inference_prototype_report.json",
    "phase63_unsupported_extension_triage": ROOT / "reports" / "phase63_unsupported_extension_triage" / "phase63_unsupported_extension_triage_report.json",
    "phase64_rtf_local_text_parser": ROOT / "reports" / "phase64_rtf_local_text_parser" / "phase64_rtf_local_text_parser_report.json",
    "phase65_full_corpus_delta_after_rtf": ROOT / "reports" / "phase65_full_corpus_delta_after_rtf" / "phase65_full_corpus_delta_after_rtf_report.json",
    "phase66_pdf_ocr_low_quality_diagnostic": ROOT / "reports" / "phase66_pdf_ocr_low_quality_diagnostic" / "phase66_pdf_ocr_low_quality_diagnostic_report.json",
    "phase67_ocr_preprocessing_comparison": ROOT / "reports" / "phase67_ocr_preprocessing_comparison" / "phase67_ocr_preprocessing_comparison_report.json",
    "phase68_image_ocr_low_quality_diagnostic": ROOT / "reports" / "phase68_image_ocr_low_quality_diagnostic" / "phase68_image_ocr_low_quality_diagnostic_report.json",
    "phase69_image_ocr_preprocessing_comparison": ROOT / "reports" / "phase69_image_ocr_preprocessing_comparison" / "phase69_image_ocr_preprocessing_comparison_report.json",
}
REQUIRED_REPORT_KEY = "phase69_image_ocr_preprocessing_comparison"


def force_local_only_runtime() -> None:
    os.environ["MEDAI_LOCAL_ONLY"] = "true"
    os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
    os.environ["MEDAI_REQUIRE_PII_SCRUB"] = "true"
    os.environ["MEDAI_PRIVACY_AUDIT"] = "true"
    app_config.MEDAI_LOCAL_ONLY = True
    app_config.MEDAI_ALLOW_EXTERNAL_API = False
    app_config.MEDAI_REQUIRE_PII_SCRUB = True
    app_config.MEDAI_PRIVACY_AUDIT = True
    outbound_gate.MEDAI_LOCAL_ONLY = True
    outbound_gate.MEDAI_ALLOW_EXTERNAL_API = False
    outbound_gate.MEDAI_REQUIRE_PII_SCRUB = True


def load_public_reports(report_inputs: dict[str, Path]) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    reports: dict[str, dict[str, Any]] = {}
    reports_read: list[str] = []
    reports_missing: list[str] = []
    for key, path in report_inputs.items():
        if not path.exists():
            reports_missing.append(key)
            continue
        try:
            reports[key] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            if key == REQUIRED_REPORT_KEY:
                raise RuntimeError(f"Required Phase69 report is unreadable: {exc.__class__.__name__}") from exc
            reports_missing.append(key)
            continue
        reports_read.append(key)
    if REQUIRED_REPORT_KEY not in reports:
        raise FileNotFoundError("Required Phase69 report is missing: phase69_image_ocr_preprocessing_comparison")
    return reports, reports_read, reports_missing


def run_audit(
    *,
    report_inputs: dict[str, Path] = REPORT_INPUTS,
    report_dir: Path = REPORT_DIR,
) -> dict[str, Any]:
    force_local_only_runtime()
    report_dir.mkdir(parents=True, exist_ok=True)
    reports, reports_read, reports_missing = load_public_reports(report_inputs)
    closed = build_closed_branches(reports)
    open_branches = build_open_branches(reports)
    deferred = build_deferred_branches(reports)
    matrix = build_decision_matrix(reports)
    selected = max(matrix, key=lambda item: (item["score"], -item["tie_breaker"]))
    phi_artifacts = phi_artifact_tracking_status()
    report: dict[str, Any] = {
        "phase": 70,
        "phase_name": "Full Corpus Post-Diagnostics Decision Audit",
        "generated_at": datetime.now(UTC).isoformat(),
        "conclusion": "post_diagnostics_decision_audit_complete",
        "recommended_next_phase": selected["recommended_phase"],
        "recommended_next_action": selected["candidate"],
        "production_extractor_should_change_yet": False,
        "production_ocr_should_change_yet": False,
        "safety_gates_should_change_yet": False,
        "manual_review_boundary_retained": True,
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "private_filename_path_leaks": 0,
        "reports_read": reports_read,
        "reports_missing": reports_missing,
        "closed_branches": closed,
        "open_branches": open_branches,
        "deferred_branches": deferred,
        "decision_matrix": matrix,
        "rationale": build_rationale(reports, selected),
        "validation_commands": [
            "python -m pytest tests",
            "python scripts/run_phase70_post_diagnostics_decision_audit.py",
            "python scripts/run_phase47_final_regression_hardening.py",
            "python scripts/run_phase48_release_snapshot_validation.py",
            "python scripts/run_phase49_privacy_gate_validation.py",
        ],
        "privacy_self_check": {
            "raw_filenames_written": False,
            "raw_paths_written": False,
            "ocr_text_written": False,
            "extracted_text_written": False,
            "phi_written": False,
            "public_report_identifiers": "aggregate_branch_names_only",
            "phi_artifact_check_passed": bool(phi_artifacts.get("passed", False)),
        },
    }
    write_json(report_dir / JSON_REPORT.name, report)
    (report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
    if public_report_contains_forbidden_tokens(report_dir):
        report["raw_phi_logged_in_public_reports"] = True
        report["conclusion"] = "BLOCKED_PRIVACY_RISK"
        write_json(report_dir / JSON_REPORT.name, report)
        (report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
    return report


def build_closed_branches(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    closed: list[dict[str, Any]] = []
    phase62 = reports.get("phase62_table_geometry_header_inference_prototype", {})
    if phase62.get("production_extractor_should_change_yet") is False:
        closed.append(
            {
                "branch": "pdf_geometry_header_inference",
                "status": "closed_to_manual_review_boundary",
                "evidence": "Phase62 prototype did not justify production extractor changes.",
            }
        )
    phase67 = reports.get("phase67_ocr_preprocessing_comparison", {})
    if phase67.get("phase68_controlled_ocr_fallback_sandbox_recommended") is False:
        closed.append(
            {
                "branch": "pdf_ocr_preprocessing",
                "status": "closed_to_manual_review_boundary",
                "evidence": "Phase67 comparison retained manual-review boundary.",
            }
        )
    phase69 = reports["phase69_image_ocr_preprocessing_comparison"]
    if phase69.get("phase70_controlled_image_ocr_fallback_sandbox_recommended") is False:
        closed.append(
            {
                "branch": "image_ocr_preprocessing",
                "status": "closed_to_manual_review_boundary",
                "evidence": "Phase69 found improvement in too few files to justify a controlled fallback sandbox.",
            }
        )
    phase65 = reports.get("phase65_full_corpus_delta_after_rtf", {})
    if phase65.get("production_extractor_should_change_yet") is False:
        closed.append(
            {
                "branch": "rtf_narrow_format_support",
                "status": "completed_no_safety_regression",
                "evidence": "Phase64/65 completed RTF local support measurement without production safety regression.",
            }
        )
    return closed


def build_open_branches(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    phase54 = reports.get("phase54_operator_review_feedback", {})
    summary = phase54.get("global_summary") or {}
    reviewed = int(summary.get("reviewed_files") or 0)
    not_reviewed = int(summary.get("not_reviewed_files") or 0)
    open_branches = [
        {
            "branch": "operator_feedback_completion",
            "status": "open",
            "evidence": f"Phase54 reviewed_files={reviewed}; not_reviewed_files={not_reviewed}.",
        },
        {
            "branch": "manual_review_package_improvement",
            "status": "open",
            "evidence": "Diagnostics repeatedly retain review/manual-review boundaries; operator workflow quality now limits useful next decisions.",
        },
        {
            "branch": "document_class_classifier_improvement",
            "status": "open",
            "evidence": "Full-corpus diagnostics still leave broad empty-extraction/document-class ambiguity unresolved.",
        },
    ]
    return open_branches


def build_deferred_branches(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    deferred = [
        {
            "branch": "docx_support_triage_or_prototype",
            "status": "deferred",
            "evidence": "Phase63/65 leave DOCX as later narrow format work; no evidence shows it outranks operator feedback.",
        },
        {
            "branch": "another_ocr_sandbox",
            "status": "deferred",
            "evidence": "Phase67 and Phase69 both retained manual-review boundary.",
        },
        {
            "branch": "production_ocr_or_extractor_change",
            "status": "deferred_blocked_by_evidence",
            "evidence": "No completed diagnostic justifies changing OCR routing, extraction logic, thresholds, or safety gates.",
        },
    ]
    return deferred


def build_decision_matrix(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    phase54 = reports.get("phase54_operator_review_feedback", {})
    summary = phase54.get("global_summary") or {}
    reviewed = int(summary.get("reviewed_files") or 0)
    feedback_incomplete = reviewed == 0
    phase69 = reports["phase69_image_ocr_preprocessing_comparison"]
    phase69_sandbox = bool(phase69.get("phase70_controlled_image_ocr_fallback_sandbox_recommended"))
    candidates = [
        {
            "candidate": "Operator feedback completion / review capture",
            "recommended_phase": "Phase71 Operator Feedback Completion and Review Prioritization",
            "evidence_strength": 5 if feedback_incomplete else 3,
            "safety_risk": 1,
            "privacy_risk": 1,
            "expected_value": 5,
            "implementation_scope": 2,
            "dependency_on_operator_feedback": 0,
            "production_change_required": False,
            "tie_breaker": 1,
        },
        {
            "candidate": "Manual-review package improvements",
            "recommended_phase": "Phase71 Manual Review Package Improvement",
            "evidence_strength": 4,
            "safety_risk": 1,
            "privacy_risk": 1,
            "expected_value": 4,
            "implementation_scope": 2,
            "dependency_on_operator_feedback": 1,
            "production_change_required": False,
            "tie_breaker": 2,
        },
        {
            "candidate": "DOCX support triage/prototype",
            "recommended_phase": "Phase71 DOCX Support Triage",
            "evidence_strength": 2,
            "safety_risk": 2,
            "privacy_risk": 2,
            "expected_value": 2,
            "implementation_scope": 3,
            "dependency_on_operator_feedback": 2,
            "production_change_required": False,
            "tie_breaker": 3,
        },
        {
            "candidate": "Document-class classifier improvement",
            "recommended_phase": "Phase71 Document Class Review-Driven Diagnostic",
            "evidence_strength": 3,
            "safety_risk": 3,
            "privacy_risk": 2,
            "expected_value": 4,
            "implementation_scope": 4,
            "dependency_on_operator_feedback": 4,
            "production_change_required": False,
            "tie_breaker": 4,
        },
        {
            "candidate": "Another OCR sandbox",
            "recommended_phase": "Deferred OCR Sandbox",
            "evidence_strength": 1 if not phase69_sandbox else 3,
            "safety_risk": 4,
            "privacy_risk": 2,
            "expected_value": 1 if not phase69_sandbox else 3,
            "implementation_scope": 4,
            "dependency_on_operator_feedback": 3,
            "production_change_required": False,
            "tie_breaker": 5,
        },
        {
            "candidate": "Production OCR change",
            "recommended_phase": "Not Recommended",
            "evidence_strength": 0,
            "safety_risk": 5,
            "privacy_risk": 3,
            "expected_value": 1,
            "implementation_scope": 5,
            "dependency_on_operator_feedback": 5,
            "production_change_required": True,
            "tie_breaker": 6,
        },
    ]
    for item in candidates:
        item["score"] = score_candidate(item)
    return sorted(candidates, key=lambda item: (-item["score"], item["tie_breaker"]))


def score_candidate(item: dict[str, Any]) -> int:
    score = (
        int(item["evidence_strength"]) * 3
        + int(item["expected_value"]) * 3
        - int(item["safety_risk"]) * 3
        - int(item["privacy_risk"]) * 2
        - int(item["implementation_scope"])
        - int(item["dependency_on_operator_feedback"])
    )
    if item["production_change_required"]:
        score -= 20
    return score


def build_rationale(reports: dict[str, dict[str, Any]], selected: dict[str, Any]) -> list[str]:
    phase67 = reports.get("phase67_ocr_preprocessing_comparison", {})
    phase69 = reports["phase69_image_ocr_preprocessing_comparison"]
    phase54 = reports.get("phase54_operator_review_feedback", {})
    reviewed = int((phase54.get("global_summary") or {}).get("reviewed_files") or 0)
    return [
        "Phase67 retained the PDF OCR manual-review boundary.",
        f"Phase69 retained the image OCR manual-review boundary after {phase69.get('meaningful_improvement_file_count')} meaningful improvements out of {phase69.get('candidate_count')} candidates.",
        f"Phase54 operator feedback remains incomplete with reviewed_files={reviewed}.",
        "Production OCR, extraction, routing, threshold, and safety-gate changes are not currently justified.",
        f"The highest-scoring safe next action is: {selected['candidate']}.",
        f"Phase67 recommended_next_action={phase67.get('recommended_next_action', 'unknown')}.",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase70 Full Corpus Post-Diagnostics Decision Audit",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Recommended next phase: `{report['recommended_next_phase']}`",
        f"- Recommended next action: `{report['recommended_next_action']}`",
        f"- Production extractor should change yet: `{report['production_extractor_should_change_yet']}`",
        f"- Production OCR should change yet: `{report['production_ocr_should_change_yet']}`",
        f"- Safety gates should change yet: `{report['safety_gates_should_change_yet']}`",
        f"- Manual-review boundary retained: `{report['manual_review_boundary_retained']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Raw PHI logged in public reports: `{report['raw_phi_logged_in_public_reports']}`",
        "",
        "## Branch Status",
        "",
        "### Closed",
    ]
    for item in report["closed_branches"]:
        lines.append(f"- `{item['branch']}`: `{item['status']}` - {item['evidence']}")
    lines.extend(["", "### Open"])
    for item in report["open_branches"]:
        lines.append(f"- `{item['branch']}`: `{item['status']}` - {item['evidence']}")
    lines.extend(["", "### Deferred"])
    for item in report["deferred_branches"]:
        lines.append(f"- `{item['branch']}`: `{item['status']}` - {item['evidence']}")
    lines.extend(
        [
            "",
            "## Decision Matrix",
            "",
            "| Candidate | Score | Evidence | Safety Risk | Privacy Risk | Expected Value | Production Change Required |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in report["decision_matrix"]:
        lines.append(
            " | ".join(
                [
                    f"| `{item['candidate']}`",
                    str(item["score"]),
                    str(item["evidence_strength"]),
                    str(item["safety_risk"]),
                    str(item["privacy_risk"]),
                    str(item["expected_value"]),
                    f"`{item['production_change_required']}` |",
                ]
            )
        )
    lines.extend(["", "## Rationale", ""])
    for item in report["rationale"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Phase70 is a decision audit only.",
            "- No production OCR routing, extraction logic, thresholds, safety gates, privacy gates, or acceptance behavior changed.",
            "- Public reports contain aggregate branch names and report labels only.",
        ]
    )
    return "\n".join(lines) + "\n"


def public_report_contains_forbidden_tokens(report_dir: Path) -> bool:
    forbidden = [
        "full_corpus_input",
        "real_validation_input",
        "local_filename_mapping_PRIVATE",
        "operator_feedback_PRIVATE",
        "original_filename",
        "original_relative_path",
        "filename_hash",
        "content_hash",
        "OCR TEXT",
        "EXTRACTED TEXT",
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        "DOB",
        "MRN",
        "Patient ",
    ]
    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in [report_dir / JSON_REPORT.name, report_dir / MD_REPORT.name]
        if path.exists()
    )
    return any(token in public_text for token in forbidden)


def main() -> int:
    try:
        report = run_audit()
    except Exception as exc:  # noqa: BLE001
        print(f"MedAI Phase70 post-diagnostics decision audit failed: {exc}", file=sys.stderr)
        return 1
    print("MedAI Phase70 post-diagnostics decision audit complete.")
    print(f"conclusion: {report['conclusion']}")
    print(f"recommended_next_phase: {report['recommended_next_phase']}")
    print(f"recommended_next_action: {report['recommended_next_action']}")
    print(f"production_extractor_should_change_yet: {report['production_extractor_should_change_yet']}")
    print(f"production_ocr_should_change_yet: {report['production_ocr_should_change_yet']}")
    print(f"safety_gates_should_change_yet: {report['safety_gates_should_change_yet']}")
    print(f"manual_review_boundary_retained: {report['manual_review_boundary_retained']}")
    print(f"external_api_used: {report['external_api_used']}")
    print(f"raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}")
    print(f"json_report: {REPORT_DIR / JSON_REPORT.name}")
    print(f"markdown_report: {REPORT_DIR / MD_REPORT.name}")
    return 1 if report["conclusion"] == "BLOCKED_PRIVACY_RISK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
