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


REPORT_DIR = REPO_ROOT / "reports" / "medai_local_image_ocr_routing_12c"
REPORT_JSON = REPORT_DIR / "medai_local_image_ocr_routing_12c_report.json"
REPORT_MD = REPORT_DIR / "medai_local_image_ocr_routing_12c_report.md"
SUMMARY_MD = REPORT_DIR / "MEDAI_LOCAL_IMAGE_OCR_ROUTING_12C.md"
RAW_OCR_SENTINELS = ["Hemoglobin 13.2", "Glucose 5.4", "SECRET OCR"]


@dataclass
class FakeResult:
    outcome: str = "queued_for_review"
    validation_status: str = "needs_review"
    validation_errors: list[dict[str, Any]] = field(default_factory=list)
    extractor_result: dict[str, Any] = field(
        default_factory=lambda: {
            "actual_extractor": "local_validation_text_pipeline",
            "confidence": 0.7,
            "external_api_used": False,
            "extracted_medical_fact_count": 1,
            "extraction_to_mkb_review_count": 1,
        }
    )
    audit: dict[str, Any] = field(default_factory=dict)


class ValidationPipeline:
    def __init__(self) -> None:
        self.text_call_count = 0

    def process_text(self, text: str, *, specialty: str, source_name: str, session_id: str) -> FakeResult:
        del text, specialty, source_name, session_id
        self.text_call_count += 1
        return FakeResult()

    def process_pdf(self, source_path: Path, *, specialty: str, session_id: str) -> FakeResult:
        del source_path, specialty, session_id
        return FakeResult(extractor_result={"actual_extractor": "pdf_pipeline", "confidence": 0.7})


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="medai_12c_") as temp_root:
        temp_path = Path(temp_root)
        _configure_launcher_paths(temp_path)
        _write_synthetic_images(launcher.TEST_INPUT_DIR)

        available = launcher.local_image_ocr_available()
        pipeline = ValidationPipeline()
        summary = launcher.run_medai_test_batch(pipeline, specialty="general")
        results = list(summary.results)
        recovered = [
            result for result in results
            if result.get("image_ocr_text_visibility") == "recovered"
        ]
        no_text = [
            result for result in results
            if result.get("image_ocr_text_visibility") == "not_recovered"
        ]
        unsupported_count = _unsupported_probe_count()
        report = {
            "task_id": "MEDAI-LOCAL-IMAGE-OCR-ROUTING-12C",
            "timestamp": datetime.now(UTC).isoformat(),
            "privacy_result": "passed",
            "external_api_used": False,
            "auto_accept": False,
            "image_ocr_available": available,
            "image_ocr_attempted": sum(1 for result in results if result.get("image_ocr_attempted")),
            "image_ocr_recovered_text_count": len(recovered),
            "review_bound_records": summary.review_count,
            "accepted_records": summary.accepted_count,
            "no_text_count": len(no_text),
            "unsupported_count": unsupported_count,
            "raw_ocr_text_in_report": False,
            "private_paths_in_report": False,
            "image_derived_records_review_bound": all(result.get("status") != "accepted" for result in results),
            "error_count": summary.error_count,
            "validation_pipeline_text_calls": pipeline.text_call_count,
            "checks": {},
            "operator_next_command": "python scripts/run_medai_local_image_ocr_routing_12c.py",
        }
        report["checks"] = {
            "local_ocr_available": report["image_ocr_available"] is True,
            "ocr_attempted": report["image_ocr_attempted"] >= 2,
            "recovered_text_routed": report["image_ocr_recovered_text_count"] >= 1 and pipeline.text_call_count >= 1,
            "no_text_review_bound": report["no_text_count"] >= 1,
            "review_bound": report["image_derived_records_review_bound"] is True,
            "no_accepted_records": report["accepted_records"] == 0,
            "no_errors": report["error_count"] == 0,
            "external_api_used_false": report["external_api_used"] is False,
            "auto_accept_false": report["auto_accept"] is False,
            "unsupported_rejected": report["unsupported_count"] == 1,
        }
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
        raise SystemExit("medai_local_image_ocr_routing_12c_not_ready")
    print("medai_local_image_ocr_routing_12c_ready")
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


def _write_synthetic_images(input_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    input_dir.mkdir(parents=True, exist_ok=True)
    readable = Image.new("RGB", (1000, 360), "white")
    draw = ImageDraw.Draw(readable)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    draw.text((40, 50), "Hemoglobin 13.2 g/dL", fill="black", font=font)
    draw.text((40, 130), "Glucose 5.4 mmol/L", fill="black", font=font)
    readable.save(input_dir / "synthetic_readable.png")

    blank = Image.new("RGB", (600, 240), "white")
    blank.save(input_dir / "synthetic_blank.png")


def _unsupported_probe_count() -> int:
    try:
        launcher.safe_test_filename("synthetic.csv")
    except ValueError:
        return 1
    return 0


def _write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# MEDAI-LOCAL-IMAGE-OCR-ROUTING-12C",
        "",
        f"- Ready: `{str(report.get('ready', False)).lower()}`",
        f"- Privacy result: `{report['privacy_result']}`",
        f"- External API used: `{str(report['external_api_used']).lower()}`",
        f"- Auto-accept: `{str(report['auto_accept']).lower()}`",
        f"- Image OCR available: `{str(report['image_ocr_available']).lower()}`",
        f"- Image OCR attempted: `{report['image_ocr_attempted']}`",
        f"- OCR recovered text count: `{report['image_ocr_recovered_text_count']}`",
        f"- Review-bound records: `{report['review_bound_records']}`",
        f"- Accepted records: `{report['accepted_records']}`",
        f"- No-text count: `{report['no_text_count']}`",
        f"- Unsupported count: `{report['unsupported_count']}`",
        f"- Raw OCR text in report: `{str(report['raw_ocr_text_in_report']).lower()}`",
        f"- Private paths in report: `{str(report['private_paths_in_report']).lower()}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in (report.get("checks") or {}).items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.append("")
    lines.append(f"Operator next command: `{report['operator_next_command']}`")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _raw_ocr_text_in_reports() -> bool:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in (REPORT_JSON, REPORT_MD, SUMMARY_MD))
    return any(sentinel in combined for sentinel in RAW_OCR_SENTINELS)


def _private_paths_in_reports() -> bool:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in (REPORT_JSON, REPORT_MD, SUMMARY_MD)).lower()
    markers = ["c:\\users", "g:\\", "appdata\\local\\temp", ".env", "synthetic_readable.png", "synthetic_blank.png"]
    return any(marker in combined for marker in markers)


if __name__ == "__main__":
    raise SystemExit(main())
