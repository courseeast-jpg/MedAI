"""Summarize operator review feedback for Phase53 blind audit outputs."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
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

PHASE53_REPORT = ROOT / "reports" / "phase53_blind_generalization_audit" / "phase53_blind_generalization_audit_report.json"
REPORT_DIR = ROOT / "reports" / "phase54_operator_review_feedback"
JSON_REPORT = REPORT_DIR / "phase54_operator_review_feedback_report.json"
MD_REPORT = REPORT_DIR / "phase54_operator_review_feedback_report.md"
CLASS_SUMMARY = REPORT_DIR / "phase54_operator_review_feedback_class_summary.json"
PRIVATE_FEEDBACK = REPORT_DIR / "operator_feedback_PRIVATE.json"

VERDICTS = {"correct", "incorrect", "uncertain", "not_reviewed"}
DOCUMENT_CLASSES = {
    "lab_report",
    "ecg",
    "prescription",
    "microbiology_pcr",
    "imaging",
    "visit_note",
    "insurance_admin",
    "unknown_other",
}
REASONS = {
    "extraction_good",
    "extraction_incomplete",
    "wrong_document_class",
    "ocr_quality_issue",
    "empty_expected",
    "empty_unexpected",
    "false_accept",
    "false_review",
    "needs_new_document_class",
    "other",
}


def run_summary(
    *,
    phase53_report_path: Path = PHASE53_REPORT,
    report_dir: Path = REPORT_DIR,
    private_feedback_path: Path | None = None,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    private_path = private_feedback_path or (report_dir / PRIVATE_FEEDBACK.name)
    phase53 = load_json(phase53_report_path)
    if not phase53:
        report = missing_phase53_report(phase53_report_path)
        write_public_reports(report, report_dir)
        return report

    feedback = load_feedback(private_path)
    merged = merge_results_with_feedback(phase53.get("results", []), feedback)
    class_summary = summarize_by_class(merged)
    global_summary = summarize_global(merged, class_summary)
    recommendations = recommendations_for(global_summary, class_summary)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "phase": "Phase 54 Operator Review Feedback Capture + Class-Level Audit Summary",
        "source_phase53_report": "reports/phase53_blind_generalization_audit/phase53_blind_generalization_audit_report.json",
        "phase53_run_id": phase53.get("run_id"),
        "phase53_total_files": phase53.get("total_files", 0),
        "local_only_mode": True,
        "external_api_used": False,
        "phase53_report_modified": False,
        "private_feedback_file": "reports/phase54_operator_review_feedback/operator_feedback_PRIVATE.json",
        "private_feedback_loaded": private_path.exists(),
        "public_report_contains_operator_notes": False,
        "raw_filenames_in_public_report": False,
        "records": public_records(merged),
        "class_summary": class_summary,
        "global_summary": global_summary,
        "recommendations": recommendations,
        "conclusion": conclusion_for(global_summary),
    }
    write_public_reports(report, report_dir)
    return report


def missing_phase53_report(path: Path) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "phase": "Phase 54 Operator Review Feedback Capture + Class-Level Audit Summary",
        "source_phase53_report": "reports/phase53_blind_generalization_audit/phase53_blind_generalization_audit_report.json",
        "phase53_report_exists": False,
        "local_only_mode": True,
        "external_api_used": False,
        "phase53_report_modified": False,
        "public_report_contains_operator_notes": False,
        "raw_filenames_in_public_report": False,
        "records": [],
        "class_summary": {},
        "global_summary": {
            "total_files": 0,
            "reviewed_files": 0,
            "not_reviewed_files": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "uncertain_count": 0,
            "false_accept_count": 0,
            "false_review_count": 0,
            "classes_seen": [],
        },
        "recommendations": ["Run Phase53 before generating Phase54 operator feedback summaries."],
        "conclusion": "phase53_report_missing",
    }


def merge_results_with_feedback(results: list[dict[str, Any]], feedback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feedback_by_key = {
        (str(row.get("safe_file_id") or row.get("file_id")), str(row.get("filename_hash"))): normalize_feedback(row)
        for row in feedback
    }
    merged: list[dict[str, Any]] = []
    for item in results:
        key = (str(item.get("file_id")), str(item.get("filename_hash")))
        row = feedback_by_key.get(key) or default_feedback(item)
        merged.append(
            {
                "safe_file_id": str(item.get("file_id")),
                "filename_hash": str(item.get("filename_hash")),
                "status": str(item.get("status") or "unknown"),
                "validation_status": item.get("validation_status"),
                "ocr_status": item.get("ocr_status") or item.get("ocr_quality_band"),
                "operator_verdict": row["operator_verdict"],
                "operator_document_class": row["operator_document_class"],
                "operator_reason": row["operator_reason"],
                "reviewed_at": row.get("reviewed_at"),
            }
        )
    return merged


def default_feedback(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "safe_file_id": item.get("file_id"),
        "filename_hash": item.get("filename_hash"),
        "operator_verdict": "not_reviewed",
        "operator_document_class": "unknown_other",
        "operator_reason": "other",
        "reviewed_at": None,
    }


def normalize_feedback(row: dict[str, Any]) -> dict[str, Any]:
    verdict = str(row.get("operator_verdict") or "not_reviewed")
    doc_class = str(row.get("operator_document_class") or "unknown_other")
    reason = str(row.get("operator_reason") or "other")
    return {
        "safe_file_id": str(row.get("safe_file_id") or row.get("file_id") or ""),
        "filename_hash": str(row.get("filename_hash") or ""),
        "operator_verdict": verdict if verdict in VERDICTS else "not_reviewed",
        "operator_document_class": doc_class if doc_class in DOCUMENT_CLASSES else "unknown_other",
        "operator_reason": reason if reason in REASONS else "other",
        "reviewed_at": row.get("reviewed_at"),
    }


def summarize_by_class(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        doc_class = record["operator_document_class"]
        bucket = summary.setdefault(doc_class, empty_class_summary())
        bucket["total"] += 1
        status = record["status"]
        if status == "accepted":
            bucket["accepted"] += 1
        elif status == "review_ocr_quality":
            bucket["review"] += 1
            bucket["review_ocr_quality"] += 1
        elif status == "empty":
            bucket["review"] += 1
            bucket["empty"] += 1
        elif status == "error":
            bucket["errors"] += 1
        else:
            bucket["review"] += 1
        if record.get("ocr_status") == "empty" or status == "empty":
            bucket["empty"] += 1 if status != "empty" else 0

        verdict = record["operator_verdict"]
        bucket[f"{verdict}_count"] += 1
        reason = record["operator_reason"]
        if reason == "false_accept":
            bucket["false_accept_count"] += 1
        if reason == "false_review":
            bucket["false_review_count"] += 1
        if reason == "ocr_quality_issue":
            bucket["ocr_quality_issue_count"] += 1
    return dict(sorted(summary.items()))


def empty_class_summary() -> dict[str, int]:
    return {
        "total": 0,
        "accepted": 0,
        "review": 0,
        "review_ocr_quality": 0,
        "empty": 0,
        "errors": 0,
        "correct_count": 0,
        "incorrect_count": 0,
        "uncertain_count": 0,
        "not_reviewed_count": 0,
        "false_accept_count": 0,
        "false_review_count": 0,
        "ocr_quality_issue_count": 0,
    }


def summarize_global(records: list[dict[str, Any]], class_summary: dict[str, dict[str, int]]) -> dict[str, Any]:
    totals = defaultdict(int)
    for record in records:
        totals["total_files"] += 1
        verdict = record["operator_verdict"]
        if verdict == "not_reviewed":
            totals["not_reviewed_files"] += 1
        else:
            totals["reviewed_files"] += 1
        totals[f"{verdict}_count"] += 1
        if record["operator_reason"] == "false_accept":
            totals["false_accept_count"] += 1
        if record["operator_reason"] == "false_review":
            totals["false_review_count"] += 1
    return {
        "total_files": totals["total_files"],
        "reviewed_files": totals["reviewed_files"],
        "not_reviewed_files": totals["not_reviewed_files"],
        "correct_count": totals["correct_count"],
        "incorrect_count": totals["incorrect_count"],
        "uncertain_count": totals["uncertain_count"],
        "false_accept_count": totals["false_accept_count"],
        "false_review_count": totals["false_review_count"],
        "classes_seen": sorted(class_summary),
    }


def recommendations_for(global_summary: dict[str, Any], class_summary: dict[str, dict[str, int]]) -> list[str]:
    recommendations: list[str] = []
    if global_summary["false_accept_count"] > 0:
        recommendations.append("Safety investigation required before automation expansion because false accepts were reported.")
    for doc_class, stats in class_summary.items():
        if stats["review_ocr_quality"] > 0 and stats["review_ocr_quality"] >= max(1, stats["total"] // 2):
            recommendations.append(f"Investigate OCR/layout quality for `{doc_class}` before parser tuning.")
    unknown = class_summary.get("unknown_other", {})
    if unknown.get("total", 0) >= max(3, global_summary["total_files"] // 3):
        recommendations.append("Consider document classification expansion because many files remain `unknown_other`.")
    if global_summary["not_reviewed_files"] > 0:
        recommendations.append("Complete operator review before parser tuning or class-specific changes.")
    if not recommendations:
        recommendations.append("No safety-blocking operator feedback trends detected.")
    return recommendations


def conclusion_for(global_summary: dict[str, Any]) -> str:
    if global_summary["false_accept_count"] > 0:
        return "blocked_by_false_accept_feedback"
    if global_summary["not_reviewed_files"] > 0:
        return "review_feedback_incomplete"
    if global_summary["incorrect_count"] > 0 or global_summary["uncertain_count"] > 0:
        return "review_feedback_collected_with_findings"
    return "review_feedback_summary_ready"


def public_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "safe_file_id": row["safe_file_id"],
            "filename_hash": row["filename_hash"],
            "status": row["status"],
            "validation_status": row["validation_status"],
            "ocr_status": row["ocr_status"],
            "operator_verdict": row["operator_verdict"],
            "operator_document_class": row["operator_document_class"],
            "operator_reason": row["operator_reason"],
            "reviewed_at": row["reviewed_at"],
        }
        for row in records
    ]


def write_public_reports(report: dict[str, Any], report_dir: Path) -> None:
    write_json(report_dir / JSON_REPORT.name, report)
    write_json(report_dir / CLASS_SUMMARY.name, report.get("class_summary", {}))
    (report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    global_summary = report.get("global_summary", {})
    lines = [
        "# Phase54 Operator Review Feedback Summary",
        "",
        f"- Timestamp: `{report['timestamp']}`",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Total files: `{global_summary.get('total_files', 0)}`",
        f"- Reviewed files: `{global_summary.get('reviewed_files', 0)}`",
        f"- Not reviewed files: `{global_summary.get('not_reviewed_files', 0)}`",
        f"- Correct: `{global_summary.get('correct_count', 0)}`",
        f"- Incorrect: `{global_summary.get('incorrect_count', 0)}`",
        f"- Uncertain: `{global_summary.get('uncertain_count', 0)}`",
        f"- False accepts: `{global_summary.get('false_accept_count', 0)}`",
        f"- False reviews: `{global_summary.get('false_review_count', 0)}`",
        f"- External API used: `{report.get('external_api_used', False)}`",
        f"- Public report contains operator notes: `{report.get('public_report_contains_operator_notes', False)}`",
        "",
        "## Class Summary",
        "",
        "| Class | Total | Accepted | Review | OCR Review | Empty | Errors | Correct | Incorrect | Uncertain | Not Reviewed | False Accept | False Review | OCR Issue |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for doc_class, stats in sorted((report.get("class_summary") or {}).items()):
        lines.append(
            f"| `{doc_class}` | {stats['total']} | {stats['accepted']} | {stats['review']} | "
            f"{stats['review_ocr_quality']} | {stats['empty']} | {stats['errors']} | "
            f"{stats['correct_count']} | {stats['incorrect_count']} | {stats['uncertain_count']} | "
            f"{stats['not_reviewed_count']} | {stats['false_accept_count']} | {stats['false_review_count']} | "
            f"{stats['ocr_quality_issue_count']} |"
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in report.get("recommendations", []))
    lines.extend(["", "## Safe Records", ""])
    lines.append("| Safe File ID | Filename Hash | Status | Verdict | Class | Reason |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in report.get("records", []):
        lines.append(
            f"| `{row['safe_file_id']}` | `{row['filename_hash']}` | `{row['status']}` | "
            f"`{row['operator_verdict']}` | `{row['operator_document_class']}` | `{row['operator_reason']}` |"
        )
    return "\n".join(lines) + "\n"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_feedback(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not payload:
        return []
    if isinstance(payload, list):
        return payload
    rows = payload.get("feedback", [])
    return rows if isinstance(rows, list) else []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    report = run_summary()
    print("MedAI Phase54 operator review feedback summary complete.")
    print(f"total_files: {report['global_summary']['total_files']}")
    print(f"reviewed_files: {report['global_summary']['reviewed_files']}")
    print(f"not_reviewed_files: {report['global_summary']['not_reviewed_files']}")
    print(f"false_accept_count: {report['global_summary']['false_accept_count']}")
    print(f"false_review_count: {report['global_summary']['false_review_count']}")
    print(f"external_api_used: {report['external_api_used']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"json_report: {REPORT_DIR / JSON_REPORT.name}")
    print(f"markdown_report: {REPORT_DIR / MD_REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
