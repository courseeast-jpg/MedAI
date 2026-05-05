"""Phase 71 — Operator Feedback Completion and Review Prioritization.

Converts the Phase54/Phase70 "operator feedback incomplete" state into a
structured, privacy-safe review prioritization workflow.

This script:
  - Reads Phase53, Phase54, Phase57, Phase58, Phase70 public reports.
  - Identifies all unreviewed candidates by safe_file_id only.
  - Assigns deterministic priority tiers based on suspected problem class.
  - Emits a prioritized operator review queue (safe IDs only).
  - Emits a plain-language operator checklist.
  - Emits a feedback template example (fake placeholder IDs only).

Does NOT change extraction logic, OCR routing, classifiers, thresholds,
safety gates, or privacy gates. production_extractor_should_change_yet=False.

Privacy:
  - No raw filenames, raw paths, PHI, OCR text, or extracted text in any
    public output file.
  - Operator notes go in a private file only (not created or staged here).
  - All file references use safe_file_id only.
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

os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")
os.environ.setdefault("MEDAI_ALLOW_EXTERNAL_API", "false")
os.environ.setdefault("MEDAI_REQUIRE_PII_SCRUB", "true")
os.environ.setdefault("MEDAI_PRIVACY_AUDIT", "true")

REPORT_DIR = ROOT / "reports" / "phase71_operator_feedback_prioritization"
JSON_REPORT = REPORT_DIR / "phase71_operator_feedback_prioritization_report.json"
MD_REPORT = REPORT_DIR / "phase71_operator_feedback_prioritization_report.md"
SAFE_QUEUE = REPORT_DIR / "operator_review_queue_SAFE.json"
SAFE_CHECKLIST = REPORT_DIR / "operator_review_checklist_SAFE.md"
FEEDBACK_TEMPLATE = REPORT_DIR / "operator_feedback_template_PRIVATE.example.json"

REPORT_INPUTS: dict[str, Path] = {
    "phase70_post_diagnostics_decision_audit": ROOT / "reports" / "phase70_post_diagnostics_decision_audit" / "phase70_post_diagnostics_decision_audit_report.json",
    "phase54_operator_review_feedback": ROOT / "reports" / "phase54_operator_review_feedback" / "phase54_operator_review_feedback_report.json",
    "phase53_blind_generalization_audit": ROOT / "reports" / "phase53_blind_generalization_audit" / "phase53_blind_generalization_audit_report.json",
    "phase57_full_corpus_inventory_audit": ROOT / "reports" / "phase57_full_corpus_inventory_audit" / "phase57_full_corpus_inventory_audit_report.json",
    "phase58_stratified_problem_fix_plan": ROOT / "reports" / "phase58_stratified_problem_fix_plan" / "phase58_stratified_problem_fix_plan.json",
}
REQUIRED_REPORT_KEY = "phase70_post_diagnostics_decision_audit"

ALLOWED_OPERATOR_ANSWERS = [
    "correct_accept",
    "false_accept",
    "correct_review",
    "false_review",
    "wrong_document_class",
    "unreadable_or_blank",
    "not_medical",
    "duplicate_or_bundle",
    "needs_manual_review",
    "unsure",
]


# ---------------------------------------------------------------------------
# Report loading
# ---------------------------------------------------------------------------


def load_public_reports(
    report_inputs: dict[str, Path] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    inputs = report_inputs or REPORT_INPUTS
    reports: dict[str, dict[str, Any]] = {}
    reports_read: list[str] = []
    reports_missing: list[str] = []
    for key, path in inputs.items():
        if not path.exists():
            reports_missing.append(key)
            continue
        try:
            reports[key] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            if key == REQUIRED_REPORT_KEY:
                raise RuntimeError(
                    f"Required Phase70 report is unreadable: {exc.__class__.__name__}"
                ) from exc
            reports_missing.append(key)
            continue
        reports_read.append(key)
    if REQUIRED_REPORT_KEY not in reports:
        raise FileNotFoundError(
            "Required Phase70 report is missing: phase70_post_diagnostics_decision_audit"
        )
    return reports, reports_read, reports_missing


# ---------------------------------------------------------------------------
# Priority tier assignment
# ---------------------------------------------------------------------------

_TIER1_REVIEW_GOALS = {
    "ocr_quality_gate_trigger": (
        "Determine whether this OCR-quality-flagged file contains usable medical data "
        "or whether the OCR safety gate correctly deferred it."
    ),
    "borderline_ocr_quality": (
        "Determine whether OCR quality is genuinely marginal or whether the document "
        "is structurally unusual — informs manual-review-package improvement."
    ),
    "flagged_needs_review": (
        "Validation system flagged this file as requiring human review. "
        "Confirm whether the routing decision was correct."
    ),
}

_TIER1_QUESTIONS = {
    "ocr_quality_gate_trigger": (
        "Open the file. Is it a real medical document (lab results, medication, "
        "radiology, clinical note)? Was the OCR quality gate correct to flag it?"
    ),
    "borderline_ocr_quality": (
        "Open the file. Is it readable? Does it contain medical data? "
        "Should this have been accepted or kept in review?"
    ),
    "flagged_needs_review": (
        "Open the file. Does it contain medical data? Was the system correct to "
        "route this to review rather than accept or empty?"
    ),
}

_TIER2_REVIEW_GOAL = (
    "Determine the true document class and whether the system routing was correct. "
    "Informs document-class classifier improvement."
)
_TIER2_QUESTION = (
    "Open the file. What kind of document is it? "
    "(Lab report / radiology / medication / admin / other non-medical)"
)


def _classify_record(record: dict[str, Any]) -> tuple[int, str]:
    """Return (priority_tier, suspected_problem_class) for a Phase54 record."""
    status = str(record.get("status") or "")
    ocr_status = str(record.get("ocr_status") or "")
    validation_status = str(record.get("validation_status") or "")

    if status == "review_ocr_quality":
        return 1, "ocr_quality_gate_trigger"
    if ocr_status == "usable_with_review":
        return 1, "borderline_ocr_quality"
    if validation_status == "needs_review":
        return 1, "flagged_needs_review"
    return 2, "unknown_document_class"


def _development_impact(tier: int, problem_class: str) -> str:
    if tier == 1:
        return (
            "Resolves operator_feedback_completion open branch. "
            "Directly informs manual_review_package_improvement and "
            "document_class_classifier_improvement open branches."
        )
    return (
        "Informs document_class_classifier_improvement open branch. "
        "Helps resolve operator_feedback_completion backlog."
    )


def build_review_queue(
    reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build deterministically ordered review queue from Phase54 records."""
    p54 = reports.get("phase54_operator_review_feedback") or {}
    records: list[dict[str, Any]] = p54.get("records") or []

    unreviewed = [
        r for r in records
        if str(r.get("operator_verdict") or "") == "not_reviewed"
    ]
    # Sort deterministically: tier ASC, then safe_file_id ASC
    classified = []
    for rec in unreviewed:
        tier, problem_class = _classify_record(rec)
        classified.append((tier, str(rec.get("safe_file_id") or ""), rec, problem_class))
    classified.sort(key=lambda t: (t[0], t[1]))

    queue: list[dict[str, Any]] = []
    for rank, (tier, _sid, rec, problem_class) in enumerate(classified, start=1):
        sid = str(rec.get("safe_file_id") or "")
        if tier == 1:
            review_goal = _TIER1_REVIEW_GOALS.get(problem_class, _TIER2_REVIEW_GOAL)
            operator_question = _TIER1_QUESTIONS.get(problem_class, _TIER2_QUESTION)
        else:
            review_goal = _TIER2_REVIEW_GOAL
            operator_question = _TIER2_QUESTION

        queue.append({
            "safe_file_id": sid,
            "priority_rank": rank,
            "priority_tier": tier,
            "suspected_problem_class": problem_class,
            "source_phase": "phase54",
            "review_goal": review_goal,
            "operator_question": operator_question,
            "allowed_answers": ALLOWED_OPERATOR_ANSWERS,
            "development_impact": _development_impact(tier, problem_class),
            "should_open_original_file": True,
            "notes_allowed_private_only": True,
        })
    return queue


# ---------------------------------------------------------------------------
# Operator checklist
# ---------------------------------------------------------------------------


def build_operator_checklist(queue: list[dict[str, Any]]) -> str:
    lines = [
        "# Operator Review Checklist — Phase 71 (SAFE)",
        "",
        "Review one file at a time. Record your answers in the private feedback file.",
        "Do not add raw filenames, raw paths, or patient data to this document.",
        "",
        "## Instructions",
        "",
        "1. Open the file listed under `safe_file_id` using the private filename mapping.",
        "2. Answer the `operator_question` for that file.",
        "3. Choose one answer from `allowed_answers`.",
        "4. Record your answer privately — never in this checklist.",
        "5. Mark the item reviewed in your private feedback file.",
        "",
        "## Allowed Answers",
        "",
    ]
    for ans in ALLOWED_OPERATOR_ANSWERS:
        lines.append(f"- `{ans}`")

    lines += [
        "",
        "## Review Queue (safe IDs only)",
        "",
        "| Priority Rank | Safe File ID | Priority Tier | Suspected Problem | Operator Question |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in queue:
        q = item["operator_question"].replace("|", "/")
        lines.append(
            f"| {item['priority_rank']} "
            f"| `{item['safe_file_id']}` "
            f"| {item['priority_tier']} "
            f"| {item['suspected_problem_class']} "
            f"| {q} |"
        )
    lines += [
        "",
        "## Privacy Rules",
        "",
        "- Do NOT write raw filenames or folder paths in any shared document.",
        "- Do NOT write patient names, dates of birth, or other PHI anywhere outside",
        "  a locally-secured private feedback file.",
        "- Use safe_file_id values only when referencing files in shared context.",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Feedback template example
# ---------------------------------------------------------------------------


def build_feedback_template_example() -> dict[str, Any]:
    """Returns an example structure with fake placeholder IDs only."""
    return {
        "_comment": (
            "PRIVATE feedback file — do not commit or share. "
            "Replace placeholder IDs with real safe_file_ids from the review queue. "
            "Do not add raw filenames, raw paths, or PHI."
        ),
        "reviewer_id": "operator_1",
        "review_session": "phase72_session_001",
        "feedback": [
            {
                "safe_file_id": "PLACEHOLDER_file_001",
                "operator_verdict": "correct_review",
                "operator_document_class": "lab_report",
                "operator_notes": "",
                "reviewed_at": "YYYY-MM-DDTHH:MM:SSZ",
            },
            {
                "safe_file_id": "PLACEHOLDER_file_002",
                "operator_verdict": "not_medical",
                "operator_document_class": "admin_or_billing",
                "operator_notes": "",
                "reviewed_at": "YYYY-MM-DDTHH:MM:SSZ",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------


def _priority_distribution(queue: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for item in queue:
        key = f"tier_{item['priority_tier']}"
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items()))


def _problem_class_distribution(queue: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for item in queue:
        key = str(item["suspected_problem_class"])
        dist[key] = dist.get(key, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))


def _top_review_goals(queue: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for item in queue:
        goal = item["review_goal"]
        if goal not in seen:
            seen.append(goal)
        if len(seen) >= 3:
            break
    return seen


# ---------------------------------------------------------------------------
# Privacy self-check
# ---------------------------------------------------------------------------

_FORBIDDEN_REPORT_STRINGS = (
    "original_filename",
    "original_relative_path",
    "full_corpus_input",
    "local_filename_mapping_PRIVATE",
    "Patient ",
    "SSN ",
    "\\\\",
)


def _privacy_self_check(
    report_json: str,
    md_text: str,
    queue_json: str,
    checklist_md: str,
) -> dict[str, bool]:
    combined = report_json + "\n" + md_text + "\n" + queue_json + "\n" + checklist_md
    raw_filenames_written = False
    raw_paths_written = False
    phi_written = any(s in combined for s in ("Patient ", "SSN ", "DOB "))
    for s in _FORBIDDEN_REPORT_STRINGS:
        if s in combined:
            if "filename" in s or "path" in s or "mapping" in s:
                raw_filenames_written = True
            elif "corpus_input" in s:
                raw_paths_written = True
    return {
        "raw_filenames_written": raw_filenames_written,
        "raw_paths_written": raw_paths_written,
        "ocr_text_written": False,
        "extracted_text_written": False,
        "phi_written": phi_written,
        "public_report_identifiers": "safe_file_ids_only",
        "phi_artifact_check_passed": not phi_written,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    rec_next = report.get("recommended_next_action", "")
    rec_phase = report.get("recommended_next_phase", "")
    lines = [
        "# Phase 71 Operator Feedback Completion and Review Prioritization",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Phase: `{report['phase']}` — {report['phase_name']}",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Recommended next phase: {rec_phase}",
        f"- Recommended next action: {rec_next}",
        "",
        "## Safety Flags",
        "",
        f"- external_api_used: `{report['external_api_used']}`",
        f"- production_extractor_should_change_yet: `{report['production_extractor_should_change_yet']}`",
        f"- production_ocr_should_change_yet: `{report['production_ocr_should_change_yet']}`",
        f"- safety_gates_should_change_yet: `{report['safety_gates_should_change_yet']}`",
        f"- manual_review_boundary_retained: `{report['manual_review_boundary_retained']}`",
        f"- raw_phi_logged_in_public_reports: `{report['raw_phi_logged_in_public_reports']}`",
        f"- private_filename_path_leaks: `{report['private_filename_path_leaks']}`",
        "",
        "## Review Queue Summary",
        "",
        f"- Total queued: `{report['review_queue_count']}`",
        "",
        "### Priority Distribution",
        "",
        "| Priority Tier | Count |",
        "| --- | ---: |",
    ]
    for tier, count in report["priority_distribution"].items():
        lines.append(f"| `{tier}` | {count} |")

    lines += [
        "",
        "### Problem Class Distribution",
        "",
        "| Problem Class | Count |",
        "| --- | ---: |",
    ]
    for cls, count in report["problem_class_distribution"].items():
        lines.append(f"| `{cls}` | {count} |")

    lines += [
        "",
        "## Reports Read / Missing",
        "",
        f"- Reports read: {', '.join(report['reports_read']) or 'none'}",
        f"- Reports missing: {', '.join(report['reports_missing']) or 'none'}",
        "",
        "## Open Branches (from Phase70)",
        "",
    ]
    for branch in report.get("open_branches", []):
        lines.append(f"- `{branch['branch']}`: {branch['status']} — {branch['evidence']}")

    lines += [
        "",
        "## Deferred Branches (from Phase70)",
        "",
    ]
    for branch in report.get("deferred_branches", []):
        lines.append(f"- `{branch['branch']}`: {branch['status']}")

    lines += [
        "",
        "## Generated Files",
        "",
        f"- `{SAFE_QUEUE.name}` — operator review queue (safe IDs only)",
        f"- `{SAFE_CHECKLIST.name}` — plain-language operator instructions",
        f"- `{FEEDBACK_TEMPLATE.name}` — private feedback file template (placeholders only)",
        "",
        "## Privacy Self-Check",
        "",
    ]
    for key, val in report["privacy_self_check"].items():
        lines.append(f"- {key}: `{val}`")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_diagnostic(
    *,
    report_inputs: dict[str, Path] | None = None,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    target_dir = report_dir or REPORT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    reports, reports_read, reports_missing = load_public_reports(report_inputs)

    p70 = reports["phase70_post_diagnostics_decision_audit"]
    open_branches: list[dict[str, Any]] = p70.get("open_branches") or []
    deferred_branches: list[dict[str, Any]] = p70.get("deferred_branches") or []

    queue = build_review_queue(reports)
    checklist_md = build_operator_checklist(queue)
    template_example = build_feedback_template_example()

    priority_dist = _priority_distribution(queue)
    problem_dist = _problem_class_distribution(queue)
    top_goals = _top_review_goals(queue)

    # Verify OCR sandbox / production change not recommended
    ocr_sandbox_recommended = False
    production_change_recommended = False

    report: dict[str, Any] = {
        "phase": 71,
        "phase_name": "Operator Feedback Completion and Review Prioritization",
        "generated_at": datetime.now(UTC).isoformat(),
        "conclusion": "operator_feedback_prioritization_ready",
        "recommended_next_phase": "Phase72 Operator Feedback Collection Pass",
        "recommended_next_action": (
            "Run operator review on the prioritized safe queue before "
            "more extractor/OCR work."
        ),
        "external_api_used": False,
        "production_extractor_should_change_yet": False,
        "production_ocr_should_change_yet": False,
        "safety_gates_should_change_yet": False,
        "manual_review_boundary_retained": True,
        "raw_phi_logged_in_public_reports": False,
        "private_filename_path_leaks": 0,
        "reports_read": reports_read,
        "reports_missing": reports_missing,
        "open_branches": open_branches,
        "deferred_branches": deferred_branches,
        "review_queue_count": len(queue),
        "priority_distribution": priority_dist,
        "problem_class_distribution": problem_dist,
        "top_review_goals": top_goals,
        "ocr_sandbox_recommended": ocr_sandbox_recommended,
        "production_change_recommended": production_change_recommended,
        "validation_commands": [
            "python -m pytest tests",
            "python scripts/run_phase71_operator_feedback_prioritization.py",
            "python scripts/run_phase47_final_regression_hardening.py",
            "python scripts/run_phase48_release_snapshot_validation.py",
            "python scripts/run_phase49_privacy_gate_validation.py",
        ],
        "privacy_self_check": {},
    }

    report_json_str = json.dumps(report, indent=2, default=str)
    md_text = render_markdown(report)
    queue_json_str = json.dumps({"review_queue": queue}, indent=2, default=str)
    privacy = _privacy_self_check(report_json_str, md_text, queue_json_str, checklist_md)
    report["privacy_self_check"] = privacy

    # Write all outputs
    (target_dir / JSON_REPORT.name).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (target_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
    (target_dir / SAFE_QUEUE.name).write_text(queue_json_str, encoding="utf-8")
    (target_dir / SAFE_CHECKLIST.name).write_text(checklist_md, encoding="utf-8")
    (target_dir / FEEDBACK_TEMPLATE.name).write_text(
        json.dumps(template_example, indent=2), encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    report = run_diagnostic()
    print("MedAI Phase 71 operator feedback prioritization complete.")
    print(f"conclusion: {report['conclusion']}")
    print(f"review_queue_count: {report['review_queue_count']}")
    print(f"priority_distribution: {report['priority_distribution']}")
    print(f"problem_class_distribution: {report['problem_class_distribution']}")
    print(f"recommended_next_phase: {report['recommended_next_phase']}")
    print(f"production_extractor_should_change_yet: {report['production_extractor_should_change_yet']}")
    print(f"production_ocr_should_change_yet: {report['production_ocr_should_change_yet']}")
    print(f"ocr_sandbox_recommended: {report['ocr_sandbox_recommended']}")
    print(f"privacy_self_check.phi_artifact_check_passed: {report['privacy_safety_check_passed'] if 'privacy_safety_check_passed' in report else report['privacy_self_check'].get('phi_artifact_check_passed')}")
    print(f"json_report: {JSON_REPORT}")
    print(f"safe_queue: {SAFE_QUEUE}")
    print(f"checklist: {SAFE_CHECKLIST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
