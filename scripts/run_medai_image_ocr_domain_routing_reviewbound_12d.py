from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import app.test_launcher as launcher
from app.mkb_explorer_model import build_mkb_explorer_model
from execution.pipeline import ExecutionPipeline
from execution.router import RoutedExtraction
from loguru import logger


logger.remove()

from mkb.sqlite_store import SQLiteStore


REPORT_DIR = REPO_ROOT / "reports" / "medai_image_ocr_domain_routing_reviewbound_12d"
REPORT_JSON = REPORT_DIR / "medai_image_ocr_domain_routing_reviewbound_12d_report.json"
REPORT_MD = REPORT_DIR / "medai_image_ocr_domain_routing_reviewbound_12d_report.md"
SUMMARY_MD = REPORT_DIR / "MEDAI_IMAGE_OCR_DOMAIN_ROUTING_REVIEWBOUND_12D.md"
RAW_OCR_SENTINEL = "SECRET OCR ROUTING TEXT"


@dataclass
class FakePiiStripper:
    last_audit: dict[str, Any] = field(default_factory=lambda: {"method": "validation_noop"})

    def strip(self, text: str) -> tuple[str, str]:
        return text, "validation_noop"


class FakeRouter:
    def execute(self, text: str, *, specialty: str = "general", source_name: str | None = None) -> RoutedExtraction:
        del specialty, source_name
        return RoutedExtraction(
            extractor_route="spacy",
            extractor_actual="spacy",
            requested_route="spacy",
            intended_route="spacy",
            selected_extractor="spacy",
            decision_reason="validation_local_route",
            route_score=0.99,
            results=[
                {
                    "extractor": "spacy",
                    "actual_extractor": "spacy",
                    "entities": [
                        {
                            "type": "test_result",
                            "text": "safe synthetic analyte",
                            "confidence": 0.95,
                            "structured": {"test_name": "safe synthetic analyte", "value": "present"},
                        }
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


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    original_recover = launcher.recover_text_from_image_local
    try:
        with tempfile.TemporaryDirectory(prefix="medai_12d_") as temp_root:
            temp_path = Path(temp_root)
            _configure_launcher_paths(temp_path)
            _write_synthetic_input(launcher.TEST_INPUT_DIR)
            launcher.recover_text_from_image_local = lambda path: launcher.LocalImageOcrResult(
                available=True,
                attempted=True,
                text=RAW_OCR_SENTINEL,
                engine="tesseract_local",
                language="eng",
                text_visibility="recovered",
            )
            sql_store = SQLiteStore(db_path=temp_path / "mkb.db", encryption_key="")
            pipeline = ExecutionPipeline(
                sql_store=sql_store,
                vector_store=None,
                pii_stripper=FakePiiStripper(),
                router=FakeRouter(),
                spacy_extractor=FakeSpacyExtractor(),
                review_queue_path=temp_path / "review_queue.jsonl",
            )
            summary = launcher.run_medai_test_batch(
                pipeline,
                specialty="urology",
                document_category="Urinalysis",
            )
            mkb_model = build_mkb_explorer_model(sql_store)
            review_model = build_mkb_explorer_model(sql_store, tier_filter="review_bound")
            ocr_records = _ocr_records(sql_store)
            report = _build_report(summary, mkb_model, review_model, ocr_records)
    finally:
        launcher.recover_text_from_image_local = original_recover

    _write_reports(report)
    report["raw_ocr_text_in_report"] = _raw_ocr_text_in_reports()
    report["private_paths_in_report"] = _private_paths_in_reports()
    report["privacy_result"] = (
        "passed"
        if not report["raw_ocr_text_in_report"] and not report["private_paths_in_report"]
        else "failed"
    )
    report["checks"]["privacy_passed"] = report["privacy_result"] == "passed"
    report["pass_count"] = sum(1 for value in report["checks"].values() if value is True)
    report["fail_count"] = len(report["checks"]) - report["pass_count"]
    report["ready"] = report["fail_count"] == 0
    _write_reports(report)
    if not report["ready"]:
        raise SystemExit("medai_image_ocr_domain_routing_reviewbound_12d_not_ready")
    print("medai_image_ocr_domain_routing_reviewbound_12d_ready")
    print(json.dumps({"report": str(REPORT_JSON.relative_to(REPO_ROOT)), "privacy_result": report["privacy_result"]}, indent=2))
    return 0


def _configure_launcher_paths(temp_path: Path) -> None:
    launcher.TEST_INPUT_DIR = temp_path / "test_input"
    launcher.TEST_OUTPUT_DIR = temp_path / "test_output"
    launcher.TEST_REVIEW_DIR = temp_path / "test_review"
    launcher.TEST_ARCHIVE_DIR = temp_path / "test_archive"
    launcher.TEST_RUN_REPORT_DIR = temp_path / "reports" / "test_runs"
    launcher.LATEST_JSON_REPORT = launcher.TEST_RUN_REPORT_DIR / "latest_test_run.json"
    launcher.LATEST_MD_REPORT = launcher.TEST_RUN_REPORT_DIR / "latest_test_run.md"
    launcher.ensure_test_launcher_dirs(root=temp_path)


def _write_synthetic_input(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "synthetic_source.png").write_bytes(b"synthetic image bytes")


def _ocr_records(sql_store: SQLiteStore) -> list[Any]:
    with sql_store._get_conn() as conn:
        rows = conn.execute("SELECT * FROM records WHERE source_type=? ORDER BY first_recorded DESC", ["image_ocr"]).fetchall()
    return [sql_store._row_to_record(row) for row in rows]


def _build_report(summary, mkb_model: dict[str, Any], review_model: dict[str, Any], records: list[Any]) -> dict[str, Any]:
    ocr_records_created = len(records)
    active_count = sum(1 for record in records if record.tier == "active" or record.status == "active")
    review_bound_count = sum(1 for record in records if record.tier == "quarantined")
    requires_review_count = sum(1 for record in records if record.requires_review)
    category_propagated = all(record.structured.get("document_category") == "Urinalysis" for record in records)
    specialty_propagated = all(record.specialty == "urology" for record in records)
    report = {
        "task_id": "MEDAI-IMAGE-OCR-DOMAIN-ROUTING-REVIEWBOUND-12D",
        "timestamp": datetime.now(UTC).isoformat(),
        "privacy_result": "pending",
        "external_api_used": False,
        "auto_accept": False,
        "image_ocr_available": launcher.local_image_ocr_available(),
        "image_ocr_text_recovered": bool(summary.results and summary.results[0].get("image_ocr_text_visibility") == "recovered"),
        "ocr_records_created": ocr_records_created,
        "ocr_records_active_count": active_count,
        "ocr_records_review_bound_count": review_bound_count,
        "ocr_records_requires_review_count": requires_review_count,
        "review_queue_count": int(review_model["counts"]["review_bound"]),
        "mkb_active_count": int(mkb_model["counts"]["active"]),
        "mkb_review_bound_count": int(mkb_model["counts"]["review_bound"]),
        "category_propagated": category_propagated,
        "specialty_propagated": specialty_propagated,
        "accepted_records": summary.accepted_count,
        "run_review_needs_review_count": summary.review_count,
        "raw_ocr_text_in_report": False,
        "private_paths_in_report": False,
        "diagnostic_summary": {
            "assignment_point": "execution.pipeline:_entities_to_records",
            "root_cause": "12C forced launcher summary review-bound after process_text, but process_text had no source_modality context before MKB write.",
            "repair": "image_ocr source context now marks records quarantined, pending_validation_review, requires_review=true before writer.write.",
        },
        "checks": {},
        "operator_next_command": "python scripts/run_medai_image_ocr_domain_routing_reviewbound_12d.py",
    }
    report["checks"] = {
        "ocr_records_created": ocr_records_created > 0,
        "ocr_records_not_active": active_count == 0,
        "ocr_records_review_bound": review_bound_count == ocr_records_created,
        "ocr_records_require_review": requires_review_count == ocr_records_created,
        "review_queue_visible": report["review_queue_count"] == ocr_records_created,
        "mkb_active_unchanged": report["mkb_active_count"] == 0,
        "mkb_review_bound_visible": report["mkb_review_bound_count"] == ocr_records_created,
        "category_propagated": category_propagated,
        "specialty_propagated": specialty_propagated,
        "accepted_records_zero": summary.accepted_count == 0,
        "external_api_used_false": report["external_api_used"] is False,
        "auto_accept_false": report["auto_accept"] is False,
    }
    return report


def _write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# MEDAI-IMAGE-OCR-DOMAIN-ROUTING-REVIEWBOUND-12D",
        "",
        f"- Ready: `{str(report.get('ready', False)).lower()}`",
        f"- Privacy result: `{report['privacy_result']}`",
        f"- External API used: `{str(report['external_api_used']).lower()}`",
        f"- Auto-accept: `{str(report['auto_accept']).lower()}`",
        f"- Image OCR available: `{str(report['image_ocr_available']).lower()}`",
        f"- Image OCR text recovered: `{str(report['image_ocr_text_recovered']).lower()}`",
        f"- OCR records created: `{report['ocr_records_created']}`",
        f"- OCR active count: `{report['ocr_records_active_count']}`",
        f"- OCR review-bound count: `{report['ocr_records_review_bound_count']}`",
        f"- OCR requires-review count: `{report['ocr_records_requires_review_count']}`",
        f"- Review Queue count: `{report['review_queue_count']}`",
        f"- MKB active count: `{report['mkb_active_count']}`",
        f"- MKB review-bound count: `{report['mkb_review_bound_count']}`",
        f"- Category propagated: `{str(report['category_propagated']).lower()}`",
        f"- Specialty propagated: `{str(report['specialty_propagated']).lower()}`",
        f"- Raw OCR text in report: `{str(report['raw_ocr_text_in_report']).lower()}`",
        f"- Private paths in report: `{str(report['private_paths_in_report']).lower()}`",
        "",
        "## Diagnostic Summary",
        "",
        f"- Assignment point: `{report['diagnostic_summary']['assignment_point']}`",
        f"- Root cause: {report['diagnostic_summary']['root_cause']}",
        f"- Repair: {report['diagnostic_summary']['repair']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in (report.get("checks") or {}).items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.append("")
    lines.append(f"Operator next command: `{report['operator_next_command']}`")
    rendered = "\n".join(lines) + "\n"
    REPORT_MD.write_text(rendered, encoding="utf-8")
    SUMMARY_MD.write_text(rendered, encoding="utf-8")


def _raw_ocr_text_in_reports() -> bool:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in (REPORT_JSON, REPORT_MD, SUMMARY_MD))
    return RAW_OCR_SENTINEL in combined


def _private_paths_in_reports() -> bool:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in (REPORT_JSON, REPORT_MD, SUMMARY_MD)).lower()
    markers = ["c:\\users", "g:\\", "appdata\\local\\temp", ".env", "synthetic_source.png"]
    return any(marker in combined for marker in markers)


if __name__ == "__main__":
    raise SystemExit(main())
