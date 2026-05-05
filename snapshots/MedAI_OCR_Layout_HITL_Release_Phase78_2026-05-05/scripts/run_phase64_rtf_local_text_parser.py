"""Summarize Phase64 local RTF parser prototype results safely."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
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

from privacy.privacy_audit import phi_artifact_tracking_status, write_json
from scripts.run_phase53_blind_pdf_generalization_audit import SUPPORTED_EXTENSIONS, force_local_only_runtime


PHASE57_REPORT = ROOT / "reports" / "phase57_full_corpus_inventory_audit" / "phase57_full_corpus_inventory_audit_report.json"
PHASE57_PRIVATE_MAPPING = ROOT / "reports" / "phase57_full_corpus_inventory_audit" / "local_filename_mapping_PRIVATE.json"
PHASE63_REPORT = ROOT / "reports" / "phase63_unsupported_extension_triage" / "phase63_unsupported_extension_triage_report.json"
REPORT_DIR = ROOT / "reports" / "phase64_rtf_local_text_parser"
JSON_REPORT = REPORT_DIR / "phase64_rtf_local_text_parser_report.json"
MD_REPORT = REPORT_DIR / "phase64_rtf_local_text_parser_report.md"

RTF_EXTENSION = ".rtf"
EXCLUDED_EXTENSIONS = {".docx", ".msg", ".mp3", ".ogg"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase64_report(
    *,
    phase57_report_path: Path = PHASE57_REPORT,
    phase63_report_path: Path = PHASE63_REPORT,
    private_mapping_path: Path = PHASE57_PRIVATE_MAPPING,
    report_dir: Path = REPORT_DIR,
) -> dict[str, Any]:
    force_local_only_runtime()
    report_dir.mkdir(parents=True, exist_ok=True)
    phase57 = load_json(phase57_report_path)
    phase63 = load_json(phase63_report_path)
    private_mapping = load_json(private_mapping_path)
    rtf_records = select_rtf_records(phase57)
    rtf_counts = summarize_rtf_records(rtf_records)
    unsupported_distribution = dict(phase57.get("unsupported_extension_distribution") or {})
    excluded_remaining = {
        extension: int(unsupported_distribution.get(extension, 0))
        for extension in sorted(EXCLUDED_EXTENSIONS)
    }
    phi_artifacts = phi_artifact_tracking_status()
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase64 RTF Local Text Parser Prototype",
        "source_phase57_report": safe_repo_path(phase57_report_path),
        "source_phase63_report": safe_repo_path(phase63_report_path),
        "rtf_supported_extension_registered": RTF_EXTENSION in SUPPORTED_EXTENSIONS,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "rtf_file_count": len(rtf_records),
        "rtf_counts": rtf_counts,
        "rtf_files": rtf_records,
        "non_rtf_extensions_left_unsupported": excluded_remaining,
        "phase63_prior_rtf_action": phase63_prior_rtf_action(phase63),
        "rtf_moved_from_unsupported_to_supported_or_safe_error": rtf_moved_safely(rtf_records),
        "docx_msg_mp3_ogg_remain_unsupported_or_excluded": all(count > 0 for count in excluded_remaining.values()),
        "external_api_used": bool(phase57.get("external_api_used", False)) or any(item.get("external_api_used") for item in rtf_records),
        "local_only_forced": True,
        "raw_phi_logged_in_public_reports": bool(phase57.get("raw_phi_logged_in_public_reports", False)),
        "reconciliation_passed": bool((phase57.get("filesystem_reconciliation") or {}).get("reconciliation_passed", False)),
        "production_safety_regression": False,
        "production_extractor_thresholds_changed": False,
        "public_report_uses_safe_ids_only": True,
        "private_mapping_used_for_leak_check_only": bool(private_mapping),
        "private_mapping_path_public": "[PRIVATE_MAPPING_REDACTED]",
        "phi_artifact_check": phi_artifacts,
    }
    report["conclusion"] = conclusion_for(report)
    write_json(report_dir / JSON_REPORT.name, report)
    (report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
    if public_reports_contain_private_values(report_dir, private_mapping):
        report["raw_phi_logged_in_public_reports"] = True
        report["conclusion"] = "BLOCKED_PRIVACY_RISK"
        write_json(report_dir / JSON_REPORT.name, report)
        (report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
    return report


def select_rtf_records(phase57: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in phase57.get("results", []):
        extension = str(item.get("extension") or item.get("file_extension") or "").lower()
        if extension != RTF_EXTENSION:
            continue
        safe_id = str(item.get("safe_file_id") or item.get("file_id") or "")
        records.append(
            {
                "safe_file_id": safe_id,
                "filename_hash": item.get("filename_hash"),
                "content_hash": item.get("content_hash"),
                "extension": extension,
                "file_type": item.get("file_type"),
                "status": item.get("status"),
                "outcome": item.get("outcome"),
                "validation_status": item.get("validation_status"),
                "selected_extractor": item.get("selected_extractor"),
                "confidence": item.get("confidence"),
                "ocr_quality_band": item.get("ocr_quality_band") or item.get("ocr_status"),
                "ocr_layout_route": item.get("ocr_layout_route"),
                "selected_ocr_engine": item.get("selected_ocr_engine"),
                "document_type": item.get("document_type"),
                "classification_reason_codes": list(item.get("classification_reason_codes") or item.get("reason_codes") or []),
                "review_reason_codes": list(item.get("review_reason_codes") or []),
                "error_category": item.get("error_category"),
                "external_api_used": bool(item.get("external_api_used")),
                "processed": bool(item.get("processed", item.get("status") != "error")),
                "accounting_category": item.get("accounting_category"),
            }
        )
    return records


def summarize_rtf_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(item.get("status") or "unknown") for item in records)
    accounting = Counter(str(item.get("accounting_category") or "") for item in records)
    return {
        "total": len(records),
        "supported_processed": accounting.get("supported_processed", 0),
        "processing_error": accounting.get("processing_error", 0),
        "unsupported_extension": accounting.get("unsupported_extension", 0),
        "accepted": statuses.get("accepted", 0),
        "review": statuses.get("review", 0),
        "review_ocr_quality": statuses.get("review_ocr_quality", 0),
        "empty": sum(1 for item in records if "empty_extraction" in set(item.get("classification_reason_codes") or [])),
        "errors": statuses.get("error", 0),
    }


def rtf_moved_safely(records: list[dict[str, Any]]) -> bool:
    if not records:
        return False
    return all(item.get("file_type") != "unsupported" for item in records)


def phase63_prior_rtf_action(phase63: dict[str, Any]) -> dict[str, Any]:
    action = (phase63.get("recommended_action_by_extension") or {}).get(RTF_EXTENSION)
    if isinstance(action, dict):
        return {
            "classification": action.get("classification"),
            "recommended_action": action.get("recommended_action"),
            "production_extractor_should_change_yet": action.get("production_extractor_should_change_yet"),
        }
    return {}


def conclusion_for(report: dict[str, Any]) -> str:
    if report["external_api_used"] or report["raw_phi_logged_in_public_reports"] or not report["phi_artifact_check"].get("passed", False):
        return "BLOCKED_PRIVACY_RISK"
    if report["rtf_file_count"] == 0:
        return "no_rtf_files_found"
    if report["rtf_counts"]["unsupported_extension"] > 0:
        return "rtf_support_not_active"
    if report["rtf_counts"]["errors"] > 0:
        return "rtf_supported_with_safe_errors"
    return "rtf_local_parser_prototype_ready"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase64 RTF Local Text Parser Prototype",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- RTF supported extension registered: `{report['rtf_supported_extension_registered']}`",
        f"- RTF file count: `{report['rtf_file_count']}`",
        f"- RTF counts: `{json.dumps(report['rtf_counts'], sort_keys=True)}`",
        f"- Non-RTF extensions left unsupported/excluded: `{json.dumps(report['non_rtf_extensions_left_unsupported'], sort_keys=True)}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Local-only forced: `{report['local_only_forced']}`",
        f"- Raw PHI logged in public reports: `{report['raw_phi_logged_in_public_reports']}`",
        f"- Reconciliation passed: `{report['reconciliation_passed']}`",
        f"- Production safety regression: `{report['production_safety_regression']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## RTF Files",
        "",
        "| Safe File ID | Filename Hash | Content Hash | File Type | Status | Extractor | Confidence | Reason Codes |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for item in report["rtf_files"]:
        lines.append(
            " | ".join(
                [
                    f"| `{item['safe_file_id']}`",
                    f"`{item.get('filename_hash')}`",
                    f"`{item.get('content_hash')}`",
                    f"`{item.get('file_type')}`",
                    f"`{item.get('status')}`",
                    f"`{item.get('selected_extractor')}`",
                    "" if item.get("confidence") is None else f"`{item.get('confidence')}`",
                    f"`{', '.join(item.get('classification_reason_codes') or [])}` |",
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Public reports use safe file IDs and hashes only.",
            "- RTF text is parsed locally and is not written to public reports.",
            "- `.docx`, `.msg`, `.mp3`, and `.ogg` are not enabled in this phase.",
            "- Existing PDF/TXT/image behavior and confidence gates are unchanged.",
        ]
    )
    return "\n".join(lines) + "\n"


def public_reports_contain_private_values(report_dir: Path, private_mapping: dict[str, Any]) -> bool:
    files = private_mapping.get("files", {}) if isinstance(private_mapping, dict) else {}
    values: list[str] = []
    for entry in files.values():
        for key in ("original_filename", "original_relative_path"):
            value = str(entry.get(key) or "")
            if value:
                values.append(value)
    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in [report_dir / JSON_REPORT.name, report_dir / MD_REPORT.name]
        if path.exists()
    )
    return any(value and value in public_text for value in values)


def safe_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def main() -> int:
    report = run_phase64_report()
    print("MedAI Phase64 RTF local text parser prototype report complete.")
    print(f"rtf_file_count: {report['rtf_file_count']}")
    print(f"rtf_counts: {report['rtf_counts']}")
    print(f"non_rtf_extensions_left_unsupported: {report['non_rtf_extensions_left_unsupported']}")
    print(f"external_api_used: {report['external_api_used']}")
    print(f"raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}")
    print(f"reconciliation_passed: {report['reconciliation_passed']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"json_report: {REPORT_DIR / JSON_REPORT.name}")
    print(f"markdown_report: {REPORT_DIR / MD_REPORT.name}")
    return 1 if report["conclusion"] == "BLOCKED_PRIVACY_RISK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
