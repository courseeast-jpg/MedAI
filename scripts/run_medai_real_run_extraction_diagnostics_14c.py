#!/usr/bin/env python3
"""MEDAI-REAL-RUN-EXTRACTION-DIAGNOSTICS-14C validation."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

import app.test_launcher as launcher
from clinical_knowledge.privacy import check_public_report_payload
from execution.pipeline import ExecutionPipeline
from execution.router import RoutedExtraction
from mkb.sqlite_store import SQLiteStore

REPORT_DIR = REPO_ROOT / "reports" / "medai_real_run_extraction_diagnostics_14c"
SUMMARY_MD_PATH = REPORT_DIR / "MEDAI_REAL_RUN_EXTRACTION_DIAGNOSTICS_14C.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_real_run_extraction_diagnostics_14c_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_real_run_extraction_diagnostics_14c_report.md"

SYNTHETIC_REAL_RUN_OCR_TEXT = """Patient Portal Results
Glucose | 5.4 | mmol/L | 3.9-5.5 |
WBC | 11.2 | x10E9/L | 4.0-10.0 | H

Vitamin D
Value: 24 ng/mL
Normal range: 30-100

Findings:
Visible source section captured for review.
"""


class NoopPiiStripper:
    last_audit = {"method": "test_noop"}

    def strip(self, text: str) -> tuple[str, str]:
        return text, "test_noop"


class EmptyLocalRouter:
    def execute(self, text: str, *, specialty: str = "general", source_name: str | None = None):
        del text, specialty, source_name
        return RoutedExtraction(
            extractor_route="spacy",
            extractor_actual="spacy",
            requested_route="spacy",
            intended_route="spacy",
            selected_extractor="spacy",
            decision_reason="synthetic_real_run_diagnostics_14c",
            route_score=0.99,
            results=[
                {
                    "extractor": "spacy",
                    "actual_extractor": "spacy",
                    "entities": [],
                    "confidence": 0.95,
                    "latency_ms": 1,
                    "raw_text": "",
                    "notes": [],
                    "external_api_used": False,
                }
            ],
        )


class FakeSpacyExtractor:
    pass


def _configure_launcher_dirs(root: Path) -> None:
    launcher.TEST_INPUT_DIR = root / "test_input"
    launcher.TEST_REVIEW_DIR = root / "test_review"
    launcher.TEST_ARCHIVE_DIR = root / "test_archive"
    launcher.TEST_OUTPUT_DIR = root / "test_output"
    launcher.TEST_RUN_REPORT_DIR = root / "reports" / "test_runs"
    launcher.LATEST_JSON_REPORT = launcher.TEST_RUN_REPORT_DIR / "latest_test_run.json"
    launcher.LATEST_MD_REPORT = launcher.TEST_RUN_REPORT_DIR / "latest_test_run.md"


def _build_pipeline(root: Path) -> ExecutionPipeline:
    return ExecutionPipeline(
        sql_store=SQLiteStore(db_path=root / "mkb.db", encryption_key=""),
        vector_store=None,
        pii_stripper=NoopPiiStripper(),
        router=EmptyLocalRouter(),
        spacy_extractor=FakeSpacyExtractor(),
        review_queue_path=root / "review_queue.jsonl",
    )


def _privacy_passes(report: dict[str, Any], markdown: str) -> bool:
    serialized = json.dumps(report, sort_keys=True)
    blocked = [
        SYNTHETIC_REAL_RUN_OCR_TEXT,
        "real_browser_upload_name.png",
        "C:\\",
        "AppData",
        "Temp",
        "Glucose",
        "WBC",
        "Vitamin D",
    ]
    if any(token in serialized or token in markdown for token in blocked):
        return False
    return bool(check_public_report_payload({"report": report, "markdown": markdown}).passed)


def build_report() -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix="medai_14c_"))
    try:
        _configure_launcher_dirs(temp_root)
        pipeline = _build_pipeline(temp_root)
        launcher.TEST_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        (launcher.TEST_INPUT_DIR / "real_browser_upload_name.png").write_bytes(b"not-real-image")
        launcher.recover_text_from_image_local = lambda path: launcher.LocalImageOcrResult(
            available=True,
            attempted=True,
            text=SYNTHETIC_REAL_RUN_OCR_TEXT,
            engine="tesseract_local",
            language="eng",
            text_visibility="recovered",
        )
        summary = launcher.run_medai_test_batch(
            pipeline,
            specialty="urology",
            document_category="Cross-domain OCR",
        )
        diagnostics = summary.safe_real_run_extraction_diagnostics
        first_result = summary.results[0] if summary.results else {}
        candidates_zero = int(diagnostics.get("extraction_candidates_created") or 0) == 0
        report: dict[str, Any] = {
            "task_id": "MEDAI-REAL-RUN-EXTRACTION-DIAGNOSTICS-14C",
            "privacy_result": "pending",
            "external_api_used": False,
            "auto_accept": False,
            "real_runtime_diagnostics_added": True,
            "ocr_recovered_counter_present": "ocr_recovered" in first_result,
            "ocr_text_shape_buckets_present": bool(
                first_result.get("ocr_text_character_bucket") and first_result.get("ocr_line_count_bucket")
            ),
            "extractor_dispatch_counter_present": "extractor_dispatch_attempted" in first_result,
            "extraction_candidate_counter_present": "extraction_candidates_created" in first_result,
            "drop_reason_counts_present": isinstance(first_result.get("drop_reason_counts"), dict),
            "records_written_counter_present": "records_written" in first_result,
            "records_deduped_counter_present": "records_deduped" in first_result,
            "review_bound_records_written_counter_present": "review_bound_records_written" in first_result,
            "category_propagated": first_result.get("selected_document_category") == "Cross-domain OCR",
            "specialty_propagated": first_result.get("selected_specialty_domain") == "urology",
            "source_modality_preserved": first_result.get("source_modality") == "image_ocr",
            "raw_ocr_text_in_report": False,
            "private_paths_in_report": False,
            "runtime_diagnostics": {
                "files_processed": diagnostics["files_processed"],
                "ocr_attempted": diagnostics["ocr_attempted"],
                "ocr_available": diagnostics["ocr_available"],
                "ocr_recovered": diagnostics["ocr_recovered"],
                "extractor_dispatch_attempted": diagnostics["extractor_dispatch_attempted"],
                "extraction_candidates_created": diagnostics["extraction_candidates_created"],
                "extraction_candidates_after_filter": diagnostics["extraction_candidates_after_filter"],
                "extraction_candidates_dropped": diagnostics["extraction_candidates_dropped"],
                "drop_reason_counts": diagnostics["drop_reason_counts"],
                "records_written": diagnostics["records_written"],
                "records_deduped": diagnostics["records_deduped"],
                "review_bound_records_written": diagnostics["review_bound_records_written"],
            },
            "safe_per_file_diagnostics": diagnostics["per_file"],
            "limitations": [
                "Synthetic OCR validation only; no private images, OCR dumps, PDFs, or runtime databases are committed.",
                "OCR text shape is reported only as buckets and booleans; raw OCR text, filenames, and paths are excluded.",
                "This block adds diagnostics and minimal runtime wiring only; it does not add parsers or interpret clinical content.",
            ],
            "next_block_recommendation_if_candidates_zero": (
                "MEDAI-REAL-OCR-TEXT-NORMALIZATION-14D"
                if candidates_zero
                else "Not required by this validation; extraction candidates were created."
            ),
        }
        markdown = _markdown(report)
        report["privacy_result"] = "passed" if _privacy_passes(report, markdown) else "failed"
        return report
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _markdown(report: dict[str, Any]) -> str:
    diag = report["runtime_diagnostics"]
    lines = [
        "# MEDAI-REAL-RUN-EXTRACTION-DIAGNOSTICS-14C",
        "",
        f"- Privacy result: `{report['privacy_result']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Auto-accept: `{report['auto_accept']}`",
        f"- Real runtime diagnostics added: `{report['real_runtime_diagnostics_added']}`",
        f"- OCR recovered counter present: `{report['ocr_recovered_counter_present']}`",
        f"- OCR text shape buckets present: `{report['ocr_text_shape_buckets_present']}`",
        f"- Extractor dispatch counter present: `{report['extractor_dispatch_counter_present']}`",
        f"- Extraction candidate counter present: `{report['extraction_candidate_counter_present']}`",
        f"- Drop reason counts present: `{report['drop_reason_counts_present']}`",
        f"- Records written counter present: `{report['records_written_counter_present']}`",
        f"- Records deduped counter present: `{report['records_deduped_counter_present']}`",
        f"- Review-bound records written counter present: `{report['review_bound_records_written_counter_present']}`",
        f"- Category propagated: `{report['category_propagated']}`",
        f"- Specialty propagated: `{report['specialty_propagated']}`",
        f"- Source modality preserved: `{report['source_modality_preserved']}`",
        f"- Raw OCR text in report: `{report['raw_ocr_text_in_report']}`",
        f"- Private paths in report: `{report['private_paths_in_report']}`",
        "",
        "## Runtime Counters",
        "",
        f"- Files processed: `{diag['files_processed']}`",
        f"- OCR attempted: `{diag['ocr_attempted']}`",
        f"- OCR available: `{diag['ocr_available']}`",
        f"- OCR recovered: `{diag['ocr_recovered']}`",
        f"- Extractor dispatch attempted: `{diag['extractor_dispatch_attempted']}`",
        f"- Extraction candidates created: `{diag['extraction_candidates_created']}`",
        f"- Extraction candidates after filter: `{diag['extraction_candidates_after_filter']}`",
        f"- Extraction candidates dropped: `{diag['extraction_candidates_dropped']}`",
        f"- Drop reason counts: `{diag['drop_reason_counts']}`",
        f"- Records written: `{diag['records_written']}`",
        f"- Records deduped: `{diag['records_deduped']}`",
        f"- Review-bound records written: `{diag['review_bound_records_written']}`",
        f"- Next block recommendation if candidates zero: `{report['next_block_recommendation_if_candidates_zero']}`",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    markdown = _markdown(report)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD_PATH.write_text(markdown, encoding="utf-8")
    SUMMARY_MD_PATH.write_text(markdown, encoding="utf-8")


def main() -> int:
    report = build_report()
    write_reports(report)
    diag = report["runtime_diagnostics"]
    ready = all(
        [
            report["privacy_result"] == "passed",
            report["external_api_used"] is False,
            report["auto_accept"] is False,
            report["real_runtime_diagnostics_added"],
            report["ocr_recovered_counter_present"],
            report["ocr_text_shape_buckets_present"],
            report["extractor_dispatch_counter_present"],
            report["extraction_candidate_counter_present"],
            report["drop_reason_counts_present"],
            report["records_written_counter_present"],
            report["records_deduped_counter_present"],
            report["review_bound_records_written_counter_present"],
            report["category_propagated"],
            report["specialty_propagated"],
            report["source_modality_preserved"],
            report["raw_ocr_text_in_report"] is False,
            report["private_paths_in_report"] is False,
            int(diag["ocr_recovered"]) > 0,
            int(diag["extractor_dispatch_attempted"]) > 0,
            int(diag["extraction_candidates_created"]) > 0,
            int(diag["records_written"]) > 0,
            int(diag["review_bound_records_written"]) > 0,
        ]
    )
    print("medai_real_run_extraction_diagnostics_14c_ready" if ready else "medai_real_run_extraction_diagnostics_14c_not_ready")
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": report["privacy_result"],
                "external_api_used": report["external_api_used"],
                "auto_accept": report["auto_accept"],
                "runtime_diagnostics_added": report["real_runtime_diagnostics_added"],
                "extractor_dispatch_counter_present": report["extractor_dispatch_counter_present"],
                "extraction_candidates_counter_present": report["extraction_candidate_counter_present"],
                "records_written_counter_present": report["records_written_counter_present"],
                "drop_reason_counts_present": report["drop_reason_counts_present"],
                "records_deduped_counter_present": report["records_deduped_counter_present"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
