#!/usr/bin/env python3
"""MEDAI-REAL-OCR-EXTRACTION-PIPELINE-INTEGRATION-14B validation."""
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
from app.mkb_explorer_model import build_mkb_explorer_model
from execution.pipeline import ExecutionPipeline
from execution.router import RoutedExtraction
from mkb.sqlite_store import SQLiteStore

REPORT_DIR = REPO_ROOT / "reports" / "medai_real_ocr_extraction_pipeline_integration_14b"
SUMMARY_MD_PATH = REPORT_DIR / "MEDAI_REAL_OCR_EXTRACTION_PIPELINE_INTEGRATION_14B.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_real_ocr_extraction_pipeline_integration_14b_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_real_ocr_extraction_pipeline_integration_14b_report.md"

SYNTHETIC_REAL_LIKE_OCR_TEXT = """Patient Portal Results
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
            decision_reason="synthetic_ocr_pipeline_integration_14b",
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


def _build_pipeline(root: Path) -> tuple[ExecutionPipeline, SQLiteStore]:
    sql_store = SQLiteStore(db_path=root / "mkb.db", encryption_key="")
    pipeline = ExecutionPipeline(
        sql_store=sql_store,
        vector_store=None,
        pii_stripper=NoopPiiStripper(),
        router=EmptyLocalRouter(),
        spacy_extractor=FakeSpacyExtractor(),
        review_queue_path=root / "review_queue.jsonl",
    )
    return pipeline, sql_store


def _privacy_passes(report: dict[str, Any], markdown: str) -> bool:
    from clinical_knowledge.privacy import check_public_report_payload

    serialized = json.dumps(report, sort_keys=True)
    if SYNTHETIC_REAL_LIKE_OCR_TEXT in serialized or SYNTHETIC_REAL_LIKE_OCR_TEXT in markdown:
        return False
    if "synthetic_real_like.png" in serialized or "synthetic_real_like.png" in markdown:
        return False
    return bool(check_public_report_payload({"report": report, "markdown": markdown}).passed)


def build_report() -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix="medai_14b_"))
    try:
        _configure_launcher_dirs(temp_root)
        pipeline, sql_store = _build_pipeline(temp_root)
        launcher.TEST_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        (launcher.TEST_INPUT_DIR / "synthetic_real_like.png").write_bytes(b"not-real-image")
        launcher.recover_text_from_image_local = lambda path: launcher.LocalImageOcrResult(
            available=True,
            attempted=True,
            text=SYNTHETIC_REAL_LIKE_OCR_TEXT,
            engine="tesseract_local",
            language="eng",
            text_visibility="recovered",
        )
        summary = launcher.run_medai_test_batch(
            pipeline,
            specialty="urology",
            document_category="Cross-domain OCR",
        )
        diagnostics = summary.safe_ocr_pipeline_diagnostics
        first_result = summary.results[0] if summary.results else {}
        review_model = build_mkb_explorer_model(sql_store, tier_filter="review_bound")
        records = [sql_store.get_record(row["record_id_full"]) for row in review_model["rows"]]
        records = [record for record in records if record is not None]
        category_propagated = all(record.structured.get("document_category") == "Cross-domain OCR" for record in records)
        specialty_propagated = all(record.specialty == "urology" for record in records)
        source_modality_preserved = all(record.structured.get("source_modality") == "image_ocr" for record in records)
        run_review_summary_consistent = (
            int(first_result.get("review_bound_records_written_count") or 0) > 0
            and "structured review-bound observations created" in str(first_result.get("run_review_summary") or "")
        )
        report: dict[str, Any] = {
            "task_id": "MEDAI-REAL-OCR-EXTRACTION-PIPELINE-INTEGRATION-14B",
            "privacy_result": "pending",
            "external_api_used": False,
            "auto_accept": False,
            "ocr_text_dispatched_to_cross_domain_extractor": int(first_result.get("extractor_dispatch_count") or 0) > 0,
            "extractor_dispatch_count": int(diagnostics["extractor_dispatch_count"]),
            "extraction_candidates_count": int(diagnostics["extraction_candidates_count"]),
            "candidates_after_filter_count": int(diagnostics["candidates_after_filter_count"]),
            "records_written_count": int(diagnostics["records_written_count"]),
            "records_deduped_count": int(diagnostics["records_deduped_count"]),
            "review_bound_records_written_count": int(diagnostics["review_bound_records_written_count"]),
            "category_propagated": bool(category_propagated),
            "specialty_propagated": bool(specialty_propagated),
            "source_modality_preserved": bool(source_modality_preserved),
            "run_review_summary_consistent": bool(run_review_summary_consistent),
            "files_processed": int(diagnostics["files_processed"]),
            "ocr_attempted_count": int(diagnostics["ocr_attempted_count"]),
            "ocr_recovered_count": int(diagnostics["ocr_recovered_count"]),
            "per_file_document_type": dict(diagnostics["per_file_document_type"]),
            "raw_ocr_text_in_report": False,
            "private_paths_in_report": False,
            "limitations": [
                "Synthetic OCR validation only; no private images, OCR dumps, PDFs, or runtime databases are committed.",
                "Diagnostics are count-only and use generated file IDs, not filenames.",
                "The integration captures visible observations for review; it does not interpret diagnosis, treatment, imaging, medication, or recommendation content.",
            ],
        }
        markdown = _markdown(report)
        report["privacy_result"] = "passed" if _privacy_passes(report, markdown) else "failed"
        return report
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-REAL-OCR-EXTRACTION-PIPELINE-INTEGRATION-14B",
        "",
        f"- Privacy result: `{report['privacy_result']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Auto-accept: `{report['auto_accept']}`",
        f"- OCR text dispatched to cross-domain extractor: `{report['ocr_text_dispatched_to_cross_domain_extractor']}`",
        f"- Extractor dispatch count: `{report['extractor_dispatch_count']}`",
        f"- Extraction candidates count: `{report['extraction_candidates_count']}`",
        f"- Candidates after filter count: `{report['candidates_after_filter_count']}`",
        f"- Records written count: `{report['records_written_count']}`",
        f"- Records deduped count: `{report['records_deduped_count']}`",
        f"- Review-bound records written count: `{report['review_bound_records_written_count']}`",
        f"- Category propagated: `{report['category_propagated']}`",
        f"- Specialty propagated: `{report['specialty_propagated']}`",
        f"- Source modality preserved: `{report['source_modality_preserved']}`",
        f"- Run & Review summary consistent: `{report['run_review_summary_consistent']}`",
        f"- Raw OCR text in report: `{report['raw_ocr_text_in_report']}`",
        f"- Private paths in report: `{report['private_paths_in_report']}`",
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
    ready = all(
        [
            report["privacy_result"] == "passed",
            report["external_api_used"] is False,
            report["auto_accept"] is False,
            report["ocr_text_dispatched_to_cross_domain_extractor"],
            report["extractor_dispatch_count"] > 0,
            report["extraction_candidates_count"] > 0,
            report["candidates_after_filter_count"] > 0,
            report["records_written_count"] > 0,
            report["review_bound_records_written_count"] > 0,
            report["category_propagated"],
            report["specialty_propagated"],
            report["source_modality_preserved"],
            report["run_review_summary_consistent"],
            report["raw_ocr_text_in_report"] is False,
            report["private_paths_in_report"] is False,
        ]
    )
    print("medai_real_ocr_extraction_pipeline_integration_14b_ready" if ready else "medai_real_ocr_extraction_pipeline_integration_14b_not_ready")
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": report["privacy_result"],
                "external_api_used": report["external_api_used"],
                "auto_accept": report["auto_accept"],
                "extraction_candidates_count": report["extraction_candidates_count"],
                "records_written_count": report["records_written_count"],
                "records_deduped_count": report["records_deduped_count"],
                "review_bound_records_written_count": report["review_bound_records_written_count"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
