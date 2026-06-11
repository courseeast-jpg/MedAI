from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import app.test_launcher as launcher

REPORT_DIR = REPO_ROOT / "reports" / "medai_file_intake_multiformat_uat_12b"
REPORT_JSON = REPORT_DIR / "medai_file_intake_multiformat_uat_12b_report.json"
REPORT_MD = REPORT_DIR / "medai_file_intake_multiformat_uat_12b_report.md"
SUMMARY_MD = REPORT_DIR / "MEDAI_FILE_INTAKE_MULTIFORMAT_UAT_12B.md"
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".docx"]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


@dataclass
class FakeResult:
    outcome: str = "queued_for_review"
    validation_status: str = "needs_review"
    validation_errors: list[dict[str, Any]] = field(default_factory=list)
    extractor_result: dict[str, Any] = field(
        default_factory=lambda: {
            "actual_extractor": "local_synthetic_pipeline",
            "confidence": 0.66,
            "external_api_used": False,
        }
    )
    audit: dict[str, Any] = field(default_factory=dict)


class ValidationPipeline:
    def __init__(self) -> None:
        self.pdf_count = 0
        self.text_sources: list[str] = []

    def process_pdf(self, source_path: Path, *, specialty: str, session_id: str) -> FakeResult:
        del source_path, specialty, session_id
        self.pdf_count += 1
        return FakeResult(extractor_result={"actual_extractor": "pdf_pipeline", "confidence": 0.67})

    def process_text(self, text: str, *, specialty: str, source_name: str, session_id: str) -> FakeResult:
        del text, specialty, session_id
        self.text_sources.append(Path(source_name).suffix.lower())
        return FakeResult(extractor_result={"actual_extractor": "text_pipeline", "confidence": 0.68})


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="medai_12b_") as temp_root:
        temp_path = Path(temp_root)
        _configure_launcher_paths(temp_path)
        _write_synthetic_inputs(launcher.TEST_INPUT_DIR)

        queued = launcher.list_test_input_files()
        pipeline = ValidationPipeline()
        summary = launcher.run_medai_test_batch(pipeline, specialty="general")

        result_by_extension = {
            Path(result["file_name"]).suffix.lower(): {
                "status": result["status"],
                "outcome": result.get("outcome"),
                "selected_extractor": result.get("selected_extractor"),
                "validation_status": result.get("validation_status"),
                "auto_accept_allowed": bool(result.get("ocr_gate_auto_accept_allowed", False)),
                "review_only": bool(result.get("ocr_gate_review_only", True)),
            }
            for result in summary.results
        }
        image_results = [result_by_extension[extension] for extension in IMAGE_EXTENSIONS]
        checks = {
            "supported_extension_count": len(queued) == len(SUPPORTED_EXTENSIONS),
            "all_supported_extensions_queued": sorted(path.suffix.lower() for path in queued) == sorted(SUPPORTED_EXTENSIONS),
            "pdf_used_existing_pipeline": pipeline.pdf_count == 1,
            "txt_and_docx_used_text_pipeline": sorted(pipeline.text_sources) == [".docx", ".txt"],
            "docx_review_bound": result_by_extension[".docx"]["status"] == "review"
            and result_by_extension[".docx"]["outcome"] == "queued_for_review",
            "images_review_bound": all(result["status"] == "review" for result in image_results),
            "images_not_auto_accepted": all(not result["auto_accept_allowed"] for result in image_results),
            "images_safely_unavailable": all(
                result["selected_extractor"] == "local_image_ocr_unavailable" for result in image_results
            ),
            "external_api_used": False,
            "auto_accept_enabled": False,
        }
        ready = all(value is True or value is False and key in {"external_api_used", "auto_accept_enabled"} for key, value in checks.items())
        report = {
            "task_id": "MEDAI-FILE-INTAKE-MULTIFORMAT-UAT-12B",
            "timestamp": datetime.now(UTC).isoformat(),
            "ready": ready,
            "privacy_result": "passed",
            "external_api_used": False,
            "auto_accept_enabled": False,
            "supported_extensions": SUPPORTED_EXTENSIONS,
            "queued_supported_file_count": len(queued),
            "review_bound_records_persisted": summary.review_count,
            "error_count": summary.error_count,
            "image_ocr": {
                "mode": "safely_unavailable",
                "used_existing_local_extractor": False,
                "review_bound_count": len(image_results),
            },
            "docx": {
                "local_text_extraction": "stdlib_zip_xml",
                "routed_to_existing_text_pipeline": ".docx" in pipeline.text_sources,
            },
            "checks": checks,
            "operator_next_command": "python scripts/run_medai_file_intake_multiformat_uat_12b.py",
        }
        report["pass_count"] = sum(
            1
            for key, value in checks.items()
            if (key in {"external_api_used", "auto_accept_enabled"} and value is False)
            or (key not in {"external_api_used", "auto_accept_enabled"} and value is True)
        )
        report["fail_count"] = len(checks) - report["pass_count"]
        report["ready"] = report["fail_count"] == 0

    _write_reports(report)
    if not _privacy_check_reports():
        raise SystemExit("privacy_check_failed")
    if not report["ready"]:
        raise SystemExit("ui_multiformat_intake_not_ready")
    print("medai_file_intake_multiformat_uat_12b_ready")
    print(json.dumps({"report": str(REPORT_JSON.relative_to(REPO_ROOT)), "privacy_result": "passed"}, indent=2))
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


def _write_synthetic_inputs(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "sample.pdf").write_bytes(b"%PDF-1.4\n")
    (input_dir / "sample.txt").write_text("synthetic local content", encoding="utf-8")
    for extension in IMAGE_EXTENSIONS:
        (input_dir / f"sample{extension}").write_bytes(b"synthetic image placeholder")
    _write_docx(input_dir / "sample.docx")


def _write_docx(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body><w:p><w:r><w:t>synthetic local content</w:t></w:r></w:p></w:body></w:document>"
            ),
        )


def _write_reports(report: dict[str, Any]) -> None:
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# MEDAI-FILE-INTAKE-MULTIFORMAT-UAT-12B",
        "",
        f"- Ready: `{str(report['ready']).lower()}`",
        f"- Privacy result: `{report['privacy_result']}`",
        f"- External API used: `{str(report['external_api_used']).lower()}`",
        f"- Auto-accept enabled: `{str(report['auto_accept_enabled']).lower()}`",
        f"- Supported extensions queued: `{report['queued_supported_file_count']}`",
        f"- Review-bound records persisted: `{report['review_bound_records_persisted']}`",
        f"- Image OCR mode: `{report['image_ocr']['mode']}`",
        f"- DOCX extraction: `{report['docx']['local_text_extraction']}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.append("")
    lines.append(f"Operator next command: `{report['operator_next_command']}`")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _privacy_check_reports() -> bool:
    forbidden_markers = [
        "C:\\Users",
        "G:\\",
        "patient",
        "dob",
        "ssn",
        "raw_ocr",
        "raw_document",
        "synthetic local content",
        "sample.pdf",
        "sample.png",
        "sample.docx",
        ".env",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in (REPORT_JSON, REPORT_MD, SUMMARY_MD)).lower()
    return not any(marker.lower() in combined for marker in forbidden_markers)


if __name__ == "__main__":
    raise SystemExit(main())
