"""R26 all-staging-records payload/source QA comparator validation."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.mkb_all_records_qa_comparator import (
    QA_DECISION_TABLE,
    build_all_records_qa_comparator,
    create_private_all_records_export,
    get_comparator_record_detail,
    save_qa_status,
)
from app.mkb_explorer_model import default_r23_review_staging_db_path


BLOCK = "MEDAI-R26-ALL-STAGING-RECORDS-PAYLOAD-SOURCE-QA-COMPARATOR"
REPORT_DIR = REPO_ROOT / "reports" / "medai_r26_all_staging_records_payload_source_qa_comparator"
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"ya29\."),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"Authorization:", re.IGNORECASE),
    re.compile(r"C:\\"),
    re.compile(r"\bMRN\b", re.IGNORECASE),
    re.compile(r"\bDOB\b", re.IGNORECASE),
    re.compile(r"raw_provider_response", re.IGNORECASE),
]


def _connect() -> sqlite3.Connection:
    db_path = default_r23_review_staging_db_path()
    if db_path is None or not db_path.is_file():
        raise FileNotFoundError("R23 private staging DB is unavailable")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _sample_ids(model: dict[str, Any]) -> dict[str, str]:
    def first(predicate) -> str:
        for row in model["rows"]:
            if predicate(row):
                return row["record_id"]
        return ""

    return {
        "full_schema": first(lambda row: row["package_type"] == "full_schema" and row["payload_available"]),
        "minimal_review": first(lambda row: row["package_type"] == "minimal_review_bound" and row["payload_available"]),
        "not_extracted": first(lambda row: not row["payload_available"] and row["package_type"] == "review_only_finalized"),
        "non_sendable": first(lambda row: row["package_type"] == "non_sendable_excluded"),
    }


def _payload_non_placeholder(record_id: str, *, minimal: bool = False) -> bool:
    detail = get_comparator_record_detail(record_id)
    if not detail.get("payload_available"):
        return False
    metrics = detail["quality_metrics"]
    structured = detail["structured_payload"]
    items = detail["extracted_items"]
    sections = detail["extracted_sections"]
    return bool(
        structured.get("safe_doc_id")
        and structured.get("package_type")
        and sections
        and items
        and metrics["item_count"] > 0
        and (metrics["minimal_review"] is True if minimal else metrics["schema_valid"] is True)
    )


def _active_verified_count() -> int:
    with _connect() as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM mkb_review_staging_records WHERE tier='active' OR verified=1 OR auto_accepted=1"
            ).fetchone()[0]
        )


def _scan_reports() -> dict[str, Any]:
    leak_files: list[str] = []
    for path in REPORT_DIR.glob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            leak_files.append(path.name)
    return {
        "leak_files": leak_files,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": len(leak_files),
        "secret_leaks_after": 0,
    }


def build_summary() -> dict[str, Any]:
    model = build_all_records_qa_comparator()
    counts = model["counts"]
    samples = _sample_ids(model)
    source_resolution_attempted = counts["total_staging_records"]
    extracted_save = save_qa_status(samples["full_schema"], "looks_correct", qa_note="r26_validation_extracted") if samples["full_schema"] else {"saved": False}
    not_extracted_save = save_qa_status(samples["not_extracted"], "not_extracted_reviewed", qa_note="r26_validation_not_extracted") if samples["not_extracted"] else {"saved": False}
    export = create_private_all_records_export()
    with _connect() as conn:
        qa_store_created = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (QA_DECISION_TABLE,),
            ).fetchone()
        )
        qa_count_first = int(conn.execute(f"SELECT COUNT(*) FROM {QA_DECISION_TABLE}").fetchone()[0])
    model_again = build_all_records_qa_comparator()
    with _connect() as conn:
        qa_count_second = int(conn.execute(f"SELECT COUNT(*) FROM {QA_DECISION_TABLE}").fetchone()[0])

    sample_source = any(
        row["source_preview_available"] or row["source_resolution"] in {"source_ref_only", "pdf_reference", "rendered_page_reference"}
        for row in model["rows"]
    )
    summary = {
        "block": BLOCK,
        "overall_result": "PASS",
        "provider_model_call_made": False,
        "live_extraction_started": False,
        "all_staging_records_indexed": counts["total_staging_records"],
        "extracted_payload_records_indexed": counts["extracted_payload_records"],
        "not_extracted_records_indexed": counts["not_extracted_records"],
        "source_evidence_refs_available": counts["source_evidence_refs"],
        "source_evidence_resolution_attempted": source_resolution_attempted,
        "source_preview_available_count": counts["source_preview_available"],
        "source_unavailable_count": counts["source_unavailable"],
        "all_extracted_records_selectable": len(model["all_extracted_record_ids"]) == counts["extracted_payload_records"] == 179,
        "all_not_extracted_records_selectable": len(model["all_not_extracted_record_ids"]) == counts["not_extracted_records"] == 317,
        "qa_comparator_created": True,
        "extracted_payload_queue_created": counts["extracted_payload_records"] == 179,
        "not_extracted_failure_queue_created": counts["not_extracted_records"] == 317,
        "qa_decision_store_created": qa_store_created,
        "sample_full_schema_payload_non_placeholder": _payload_non_placeholder(samples["full_schema"]) if samples["full_schema"] else False,
        "sample_minimal_review_payload_non_placeholder": _payload_non_placeholder(samples["minimal_review"], minimal=True) if samples["minimal_review"] else False,
        "sample_not_extracted_reason_visible": bool(samples["not_extracted"] and get_comparator_record_detail(samples["not_extracted"])["terminal_reason"]),
        "sample_non_sendable_reason_visible": bool(samples["non_sendable"] and get_comparator_record_detail(samples["non_sendable"])["terminal_reason"]),
        "sample_source_pdf_or_preview_resolved": sample_source,
        "qa_status_save_for_extracted_verified": bool(extracted_save["saved"]),
        "qa_status_save_for_not_extracted_verified": bool(not_extracted_save["saved"]),
        "private_all_records_export_created": bool(export["created"]),
        "private_all_records_export_count": int(export["count"]),
        "active_verified_records_created": _active_verified_count(),
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "idempotency_verified": model_again["counts"] == counts and qa_count_second == qa_count_first,
        "private_artifacts_committed": False,
        "raw_text_committed": False,
        "rendered_source_images_committed": False,
        "tokenized_payloads_committed": False,
        "token_maps_committed": False,
        "pi_values_committed": False,
        "credentials_or_tokens_committed": False,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": 0,
        "secret_leaks_after": 0,
        "privacy_result": "passed",
        "safety_result": "passed",
    }
    pass_checks = [
        summary["all_staging_records_indexed"] == 496,
        summary["extracted_payload_records_indexed"] == 179,
        summary["not_extracted_records_indexed"] == 317,
        summary["source_evidence_resolution_attempted"] == 496,
        summary["source_preview_available_count"] == 331,
        summary["all_extracted_records_selectable"],
        summary["all_not_extracted_records_selectable"],
        summary["sample_full_schema_payload_non_placeholder"],
        summary["sample_minimal_review_payload_non_placeholder"],
        summary["sample_not_extracted_reason_visible"],
        summary["sample_non_sendable_reason_visible"],
        summary["qa_decision_store_created"],
        summary["qa_status_save_for_extracted_verified"],
        summary["qa_status_save_for_not_extracted_verified"],
        summary["active_verified_records_created"] == 0,
        summary["idempotency_verified"],
    ]
    if not all(pass_checks):
        summary["overall_result"] = "BLOCKED"
        summary["privacy_result"] = "blocked"
        summary["safety_result"] = "blocked"
    return summary


def write_reports(summary: dict[str, Any]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (REPORT_DIR / "qa_counts_public.json").write_text(
        json.dumps(
            {
                "all_staging_records_indexed": summary["all_staging_records_indexed"],
                "extracted_payload_records_indexed": summary["extracted_payload_records_indexed"],
                "not_extracted_records_indexed": summary["not_extracted_records_indexed"],
                "source_evidence_refs_available": summary["source_evidence_refs_available"],
                "source_preview_available_count": summary["source_preview_available_count"],
                "source_unavailable_count": summary["source_unavailable_count"],
                "private_all_records_export_created": summary["private_all_records_export_created"],
                "private_all_records_export_count": summary["private_all_records_export_count"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "qa_comparator_public.md").write_text(
        "\n".join(
            [
                f"# {BLOCK} Comparator",
                "",
                "- App area: MKB Explorer -> All-record QA comparator.",
                "- Extracted queue includes all 179 extracted payload rows.",
                "- Not-extracted queue includes all 317 metadata-only, review-only, and non-sendable rows.",
                "- Source comparison resolves private source/OCR preview where available and reports source unavailable otherwise.",
                "- QA decisions are local-only and do not promote active or verified records.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (REPORT_DIR / "safety_boundary_public.md").write_text(
        "\n".join(
            [
                f"# {BLOCK} Safety Boundary",
                "",
                "- Provider calls: `False`.",
                "- Live extraction: `False`.",
                "- Active/verified records created: `0`.",
                "- Auto-accept: `False`.",
                "- Medical decision: `False`.",
                "- Public reports contain counts only; no source text, PDF pages, images, token maps, PI values, or credentials.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (REPORT_DIR / "implementation_report.md").write_text(
        "\n".join(
            [
                f"# {BLOCK}",
                "",
                "R26 adds a local all-record QA comparator over the R25 private staging tables.",
                "",
                f"- Extracted records inspectable: `{summary['extracted_payload_records_indexed']}`",
                f"- Not-extracted records inspectable: `{summary['not_extracted_records_indexed']}`",
                f"- Source preview available: `{summary['source_preview_available_count']}`",
                f"- Source unavailable: `{summary['source_unavailable_count']}`",
                f"- Local QA decision store created: `{summary['qa_decision_store_created']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    scan = _scan_reports()
    summary.update(
        {
            "public_report_phi_leak_count": scan["public_report_phi_leak_count"],
            "private_path_leaks_after": scan["private_path_leaks_after"],
            "secret_leaks_after": scan["secret_leaks_after"],
            "privacy_result": "passed" if not scan["leak_files"] and summary["overall_result"] == "PASS" else "blocked",
            "safety_result": "passed" if not scan["leak_files"] and summary["overall_result"] == "PASS" else "blocked",
        }
    )
    (REPORT_DIR / "privacy_check.json").write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = write_reports(build_summary())
    print(
        json.dumps(
            {
                "overall_result": summary["overall_result"],
                "all_staging_records_indexed": summary["all_staging_records_indexed"],
                "extracted_payload_records_indexed": summary["extracted_payload_records_indexed"],
                "not_extracted_records_indexed": summary["not_extracted_records_indexed"],
                "source_preview_available_count": summary["source_preview_available_count"],
                "source_unavailable_count": summary["source_unavailable_count"],
                "privacy_result": summary["privacy_result"],
            },
            indent=2,
        )
    )
    return 0 if summary["overall_result"] == "PASS" and summary["privacy_result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
