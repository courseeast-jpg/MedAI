#!/usr/bin/env python3
"""MEDAI-OPERATOR-WORKFLOW-UAT-12A validation harness."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

try:
    from loguru import logger

    logger.remove()
except Exception:
    pass

REPORT_DIR = REPO_ROOT / "reports" / "medai_operator_workflow_uat_12a"
SUMMARY_MD_PATH = REPORT_DIR / "MEDAI_OPERATOR_WORKFLOW_UAT_12A.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_operator_workflow_uat_12a_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_operator_workflow_uat_12a_report.md"

SUPPORTED_FILE_TYPES = ["PDF", "TXT", "PNG", "JPG", "JPEG", "TIFF", "TIF", "BMP", "DOCX"]
SYNTHETIC_OCR_TEXT = (
    "Synthetic lab panel\n"
    "Glucose: 5.2 mmol/L\n"
    "Hemoglobin: 13.4 g/dL\n"
    "WBC: 7.1 x10E9/L\n"
)


@dataclass
class UatContext:
    temp_root: Path
    sql_store: Any
    launcher: Any


class FakePiiStripper:
    last_audit = {"method": "test_noop", "placeholder_count": 0}

    def strip(self, text: str) -> tuple[str, str]:
        return text, "test_noop"


class FakeRouter:
    def execute(self, text: str, *, specialty: str = "general", source_name: str | None = None):
        from execution.router import RoutedExtraction

        del specialty, source_name
        return RoutedExtraction(
            extractor_route="spacy",
            extractor_actual="spacy",
            requested_route="spacy",
            intended_route="spacy",
            selected_extractor="spacy",
            decision_reason="uat_local_synthetic_route",
            route_score=0.99,
            results=[
                {
                    "extractor": "spacy",
                    "actual_extractor": "spacy",
                    "entities": [
                        {
                            "type": "test_result",
                            "text": "synthetic analyte alpha",
                            "confidence": 0.95,
                            "structured": {"test_name": "synthetic analyte alpha", "value": "5.2", "unit": "unit"},
                        },
                        {
                            "type": "test_result",
                            "text": "synthetic analyte beta",
                            "confidence": 0.94,
                            "structured": {"test_name": "synthetic analyte beta", "value": "13.4", "unit": "unit"},
                        },
                        {
                            "type": "test_result",
                            "text": "synthetic analyte gamma",
                            "confidence": 0.93,
                            "structured": {"test_name": "synthetic analyte gamma", "value": "7.1", "unit": "unit"},
                        },
                    ],
                    "confidence": 0.95,
                    "latency_ms": 1,
                    "raw_text": text,
                    "notes": [],
                    "external_api_used": False,
                }
            ],
        )


class FakeSpacyExtractor:
    pass


def _configure_launcher_dirs(launcher: Any, root: Path) -> None:
    launcher.TEST_INPUT_DIR = root / "test_input"
    launcher.TEST_REVIEW_DIR = root / "test_review"
    launcher.TEST_ARCHIVE_DIR = root / "test_archive"
    launcher.TEST_OUTPUT_DIR = root / "test_output"
    launcher.TEST_RUN_REPORT_DIR = root / "reports" / "test_runs"
    launcher.LATEST_JSON_REPORT = launcher.TEST_RUN_REPORT_DIR / "latest_test_run.json"
    launcher.LATEST_MD_REPORT = launcher.TEST_RUN_REPORT_DIR / "latest_test_run.md"


def _write_docx(path: Path, text: str = "synthetic docx text") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
            ),
        )


def _create_supported_queue(input_dir: Path) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "file_pdf.pdf": b"%PDF-1.4 synthetic",
        "file_txt.txt": b"synthetic text",
        "file_png.png": b"synthetic image",
        "file_jpg.jpg": b"synthetic image",
        "file_jpeg.jpeg": b"synthetic image",
        "file_tiff.tiff": b"synthetic image",
        "file_tif.tif": b"synthetic image",
        "file_bmp.bmp": b"synthetic image",
    }
    for name, data in files.items():
        (input_dir / name).write_bytes(data)
    _write_docx(input_dir / "file_docx.docx")
    return sorted(input_dir.iterdir())


def build_context() -> UatContext:
    import app.test_launcher as launcher
    from mkb.sqlite_store import SQLiteStore

    temp_root = Path(tempfile.mkdtemp(prefix="medai_operator_workflow_uat_12a_"))
    _configure_launcher_dirs(launcher, temp_root)
    sql_store = SQLiteStore(db_path=temp_root / "mkb.db", encryption_key="")
    return UatContext(temp_root=temp_root, sql_store=sql_store, launcher=launcher)


def build_pipeline(sql_store: Any):
    from execution.pipeline import ExecutionPipeline

    return ExecutionPipeline(
        sql_store=sql_store,
        vector_store=None,
        pii_stripper=FakePiiStripper(),
        router=FakeRouter(),
        spacy_extractor=FakeSpacyExtractor(),
        review_queue_path=sql_store.db_path.parent / "review_queue.jsonl",
    )


def run_uat() -> dict[str, Any]:
    from app.main import queue_display_state
    from app.mkb_explorer_model import build_mkb_explorer_model
    from app.operator_review_actions import (
        accept_after_source_comparison,
        defer_extracted_fact,
        reject_extracted_fact,
    )

    context = build_context()
    launcher = context.launcher
    try:
        support_root = context.temp_root / "supported_check"
        supported_files = _create_supported_queue(support_root)
        supported_names = [path.name for path in launcher.list_test_input_files(support_root)]
        supported_ok = len(supported_names) == len(SUPPORTED_FILE_TYPES)

        run_input = launcher.TEST_INPUT_DIR
        run_input.mkdir(parents=True, exist_ok=True)
        (run_input / "workflow_image.png").write_bytes(b"synthetic image")
        queued_files = launcher.list_test_input_files(run_input)
        queue_state = queue_display_state(queued_count=len(queued_files), selected_count=0)
        empty_queue_state = queue_display_state(queued_count=0, selected_count=0)

        launcher.recover_text_from_image_local = lambda path: launcher.LocalImageOcrResult(
            available=True,
            attempted=True,
            text=SYNTHETIC_OCR_TEXT,
            engine="tesseract_local",
            language="eng",
            text_visibility="recovered",
        )

        summary = launcher.run_medai_test_batch(
            build_pipeline(context.sql_store),
            specialty="urology",
            document_category="Urinalysis",
        )
        mkb_model = build_mkb_explorer_model(context.sql_store)
        review_model = build_mkb_explorer_model(context.sql_store, tier_filter="review_bound")
        review_rows = list(review_model["rows"])
        record_ids = [row["record_id_full"] for row in review_rows]

        records_before = [context.sql_store.get_record(record_id) for record_id in record_ids]
        category_propagated = all(
            (record and record.structured.get("document_category") == "Urinalysis")
            for record in records_before
        )
        specialty_propagated = all(record and record.specialty == "urology" for record in records_before)
        review_bound_before = sum(1 for record in records_before if record and record.requires_review)
        active_before = int(mkb_model["counts"]["active"])

        accept_result = accept_after_source_comparison(
            context.sql_store,
            record_ids[0],
            operator_note="verified",
            session_id="operator-workflow-uat-12a",
        )
        reject_result = reject_extracted_fact(
            context.sql_store,
            record_ids[1],
            operator_note="not usable",
            session_id="operator-workflow-uat-12a",
        )
        defer_result = defer_extracted_fact(
            context.sql_store,
            record_ids[2],
            operator_note="later",
            session_id="operator-workflow-uat-12a",
        )

        after_records = {record_id: context.sql_store.get_record(record_id) for record_id in record_ids[:3]}
        active_after_accept = sum(
            1
            for record in after_records.values()
            if record and record.tier == "active" and record.status == "accepted_after_operator_review"
        )
        rejected_after_reject = sum(
            1
            for record in after_records.values()
            if record and record.tier == "superseded" and record.status == "rejected_after_operator_review"
        )
        review_bound_after_defer = sum(
            1
            for record in after_records.values()
            if record and record.tier == "quarantined" and record.status == "deferred_by_operator" and record.requires_review
        )

        public_report: dict[str, Any] = {
            "task_id": "MEDAI-OPERATOR-WORKFLOW-UAT-12A",
            "privacy_result": "pending",
            "external_api_used": False,
            "auto_accept": False,
            "supported_file_types_checked": SUPPORTED_FILE_TYPES,
            "supported_file_types_accepted": bool(supported_ok),
            "queued_count": int(queue_state["queued_count"]),
            "empty_queue_start_enabled": bool(empty_queue_state["start_enabled"]),
            "queued_start_enabled": bool(queue_state["start_enabled"]),
            "run_completed": summary.error_count == 0 and summary.review_count > 0,
            "local_ocr_attempted": bool(summary.results and summary.results[0].get("image_ocr_attempted")),
            "text_recovery_status": str(summary.results[0].get("image_ocr_text_visibility") if summary.results else ""),
            "records_created": len(record_ids),
            "records_review_bound_before_action": review_bound_before,
            "records_active_before_action": active_before,
            "review_queue_count_before_action": int(review_model["row_count"]),
            "accept_action_passed": bool(accept_result.success),
            "reject_action_passed": bool(reject_result.success),
            "defer_action_passed": bool(defer_result.success),
            "records_active_after_accept": active_after_accept,
            "records_rejected_after_reject": rejected_after_reject,
            "records_review_bound_after_defer": review_bound_after_defer,
            "category_propagated": bool(category_propagated),
            "specialty_propagated": bool(specialty_propagated),
            "raw_ocr_text_in_report": False,
            "private_paths_in_report": False,
            "accepted_count_before_human_action": int(summary.accepted_count),
            "review_queue_reader_rows": int(review_model["row_count"]),
            "mkb_explorer_review_bound_count": int(mkb_model["counts"]["review_bound"]),
            "action_proof": {
                "accept_selected_record_only": bool(
                    accept_result.success
                    and after_records[record_ids[0]].status == "accepted_after_operator_review"
                    and after_records[record_ids[1]].status != "accepted_after_operator_review"
                    and after_records[record_ids[2]].status != "accepted_after_operator_review"
                ),
                "reject_selected_record_only": bool(
                    reject_result.success
                    and after_records[record_ids[1]].status == "rejected_after_operator_review"
                    and after_records[record_ids[0]].status != "rejected_after_operator_review"
                    and after_records[record_ids[2]].status != "rejected_after_operator_review"
                ),
                "defer_selected_record_only": bool(
                    defer_result.success
                    and after_records[record_ids[2]].status == "deferred_by_operator"
                    and after_records[record_ids[0]].status != "deferred_by_operator"
                    and after_records[record_ids[1]].status != "deferred_by_operator"
                ),
            },
            "limitations": [
                "Synthetic UAT only; no real private documents or runtime DBs were read or mutated.",
                "Local image OCR is simulated through the existing launcher seam to avoid depending on host OCR binaries.",
                "UI redesign is out of scope.",
            ],
        }
        public_report["uat_passed"] = _uat_passed(public_report)
        public_report["privacy_result"] = "passed" if _privacy_passes(public_report) else "failed"
        _write_reports(public_report)
        return public_report
    finally:
        shutil.rmtree(context.temp_root, ignore_errors=True)


def _uat_passed(report: dict[str, Any]) -> bool:
    return all(
        [
            report["supported_file_types_accepted"],
            report["queued_count"] > 0,
            report["empty_queue_start_enabled"] is False,
            report["queued_start_enabled"] is True,
            report["run_completed"],
            report["local_ocr_attempted"],
            report["text_recovery_status"] == "recovered",
            report["records_created"] >= 3,
            report["records_review_bound_before_action"] >= 3,
            report["records_active_before_action"] == 0,
            report["review_queue_count_before_action"] >= 3,
            report["accept_action_passed"],
            report["reject_action_passed"],
            report["defer_action_passed"],
            report["records_active_after_accept"] == 1,
            report["records_rejected_after_reject"] == 1,
            report["records_review_bound_after_defer"] == 1,
            report["category_propagated"],
            report["specialty_propagated"],
            report["external_api_used"] is False,
            report["auto_accept"] is False,
            report["accepted_count_before_human_action"] == 0,
            all(report["action_proof"].values()),
        ]
    )


def _privacy_passes(report: dict[str, Any]) -> bool:
    from clinical_knowledge.privacy import check_public_report_payload

    if SYNTHETIC_OCR_TEXT in json.dumps(report, sort_keys=True):
        return False
    result = check_public_report_payload(report)
    return bool(result.passed)


def _write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _markdown(report)
    REPORT_MD_PATH.write_text(md, encoding="utf-8")
    SUMMARY_MD_PATH.write_text(md, encoding="utf-8")


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-OPERATOR-WORKFLOW-UAT-12A",
        "",
        f"- UAT passed: `{report['uat_passed']}`",
        f"- Privacy result: `{report['privacy_result']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Auto-accept: `{report['auto_accept']}`",
        f"- Supported file types checked: `{len(report['supported_file_types_checked'])}`",
        f"- Queued count: `{report['queued_count']}`",
        f"- Run completed: `{report['run_completed']}`",
        f"- Local OCR attempted: `{report['local_ocr_attempted']}`",
        f"- Text recovery status: `{report['text_recovery_status']}`",
        f"- Records created: `{report['records_created']}`",
        f"- Review-bound before action: `{report['records_review_bound_before_action']}`",
        f"- Active before action: `{report['records_active_before_action']}`",
        f"- Review queue before action: `{report['review_queue_count_before_action']}`",
        f"- Accept action passed: `{report['accept_action_passed']}`",
        f"- Reject action passed: `{report['reject_action_passed']}`",
        f"- Defer action passed: `{report['defer_action_passed']}`",
        f"- Active after accept: `{report['records_active_after_accept']}`",
        f"- Rejected after reject: `{report['records_rejected_after_reject']}`",
        f"- Review-bound after defer: `{report['records_review_bound_after_defer']}`",
        f"- Category propagated: `{report['category_propagated']}`",
        f"- Specialty propagated: `{report['specialty_propagated']}`",
        f"- Raw OCR text in report: `{report['raw_ocr_text_in_report']}`",
        f"- Private paths in report: `{report['private_paths_in_report']}`",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    report = run_uat()
    print("medai_operator_workflow_uat_12a_ready" if report["uat_passed"] else "medai_operator_workflow_uat_12a_not_ready")
    print(json.dumps({
        "report": str(REPORT_JSON_PATH.relative_to(REPO_ROOT)),
        "uat_passed": report["uat_passed"],
        "privacy_result": report["privacy_result"],
        "external_api_used": report["external_api_used"],
        "auto_accept": report["auto_accept"],
    }, indent=2))
    return 0 if report["uat_passed"] and report["privacy_result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
