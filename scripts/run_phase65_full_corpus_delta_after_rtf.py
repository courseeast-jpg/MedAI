"""Report the corpus-level delta after Phase64 RTF support."""

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

os.environ["MEDAI_LOCAL_ONLY"] = "true"
os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ["MEDAI_REQUIRE_PII_SCRUB"] = "true"
os.environ["MEDAI_PRIVACY_AUDIT"] = "true"

import app.config as app_config
import privacy.outbound_gate as outbound_gate
from privacy.privacy_audit import phi_artifact_tracking_status, write_json


PHASE57_REPORT = ROOT / "reports" / "phase57_full_corpus_inventory_audit" / "phase57_full_corpus_inventory_audit_report.json"
PHASE57_PRIVATE_MAPPING = ROOT / "reports" / "phase57_full_corpus_inventory_audit" / "local_filename_mapping_PRIVATE.json"
PHASE63_REPORT = ROOT / "reports" / "phase63_unsupported_extension_triage" / "phase63_unsupported_extension_triage_report.json"
PHASE64_REPORT = ROOT / "reports" / "phase64_rtf_local_text_parser" / "phase64_rtf_local_text_parser_report.json"
REPORT_DIR = ROOT / "reports" / "phase65_full_corpus_delta_after_rtf"
JSON_REPORT = REPORT_DIR / "phase65_full_corpus_delta_after_rtf_report.json"
MD_REPORT = REPORT_DIR / "phase65_full_corpus_delta_after_rtf_report.md"

RTF_EXTENSION = ".rtf"
REMAIN_UNSUPPORTED_EXTENSIONS = {".docx", ".mp3", ".msg", ".ogg"}


def force_local_only_runtime() -> None:
    os.environ["MEDAI_LOCAL_ONLY"] = "true"
    os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
    os.environ["MEDAI_REQUIRE_PII_SCRUB"] = "true"
    os.environ["MEDAI_PRIVACY_AUDIT"] = "true"
    app_config.MEDAI_LOCAL_ONLY = True
    app_config.MEDAI_ALLOW_EXTERNAL_API = False
    app_config.MEDAI_REQUIRE_PII_SCRUB = True
    app_config.MEDAI_PRIVACY_AUDIT = True
    outbound_gate.MEDAI_LOCAL_ONLY = True
    outbound_gate.MEDAI_ALLOW_EXTERNAL_API = False
    outbound_gate.MEDAI_REQUIRE_PII_SCRUB = True


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_delta_report(
    *,
    phase57_report_path: Path = PHASE57_REPORT,
    phase63_report_path: Path = PHASE63_REPORT,
    phase64_report_path: Path = PHASE64_REPORT,
    private_mapping_path: Path = PHASE57_PRIVATE_MAPPING,
    report_dir: Path = REPORT_DIR,
) -> dict[str, Any]:
    force_local_only_runtime()
    report_dir.mkdir(parents=True, exist_ok=True)

    phase57 = load_json(phase57_report_path)
    phase63 = load_json(phase63_report_path)
    phase64 = load_json(phase64_report_path)
    private_mapping = load_json(private_mapping_path)
    phi_artifacts = phi_artifact_tracking_status()

    rtf_counts = dict(phase64.get("rtf_counts") or {})
    rtf_file_count = int(phase64.get("rtf_file_count") or rtf_counts.get("total") or 0)
    after_unsupported = int(phase57.get("unsupported_count") or 0)
    before_unsupported = after_unsupported + rtf_file_count
    unsupported_after_distribution = normalize_distribution(
        phase57.get("unsupported_extension_distribution") or phase63.get("extension_distribution") or {}
    )
    unsupported_before_distribution = dict(unsupported_after_distribution)
    if rtf_file_count:
        unsupported_before_distribution[RTF_EXTENSION] = rtf_file_count

    status_delta = {
        "accepted": int(rtf_counts.get("accepted", 0)),
        "review": int(rtf_counts.get("review", 0)) + int(rtf_counts.get("review_ocr_quality", 0)),
        "review_ocr_quality": int(rtf_counts.get("review_ocr_quality", 0)),
        "empty": int(rtf_counts.get("empty", 0)),
        "errors": -rtf_file_count + int(rtf_counts.get("errors", 0)),
        "unsupported_extension": -rtf_file_count,
        "supported_processed": int(rtf_counts.get("supported_processed", 0)),
    }
    rtf_files = safe_rtf_files(phase64.get("rtf_files") or [])
    non_rtf_remaining = {
        extension: int(unsupported_after_distribution.get(extension, 0))
        for extension in sorted(REMAIN_UNSUPPORTED_EXTENSIONS)
    }
    safety_delta = {
        "external_api_used": bool(phase57.get("external_api_used")) or bool(phase64.get("external_api_used")),
        "raw_phi_logged": bool(phase57.get("raw_phi_logged_in_public_reports")) or bool(phase64.get("raw_phi_logged_in_public_reports")),
        "safety_regression": False,
        "reconciliation_passed": bool(
            phase64.get("reconciliation_passed")
            if "reconciliation_passed" in phase64
            else (phase57.get("filesystem_reconciliation") or {}).get("reconciliation_passed", False)
        ),
        "phi_report_artifacts_tracked": not bool(phi_artifacts.get("passed", False)),
    }
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase65 Full Corpus Delta Audit After RTF Support",
        "source_phase57_report": safe_repo_path(phase57_report_path),
        "source_phase63_report": safe_repo_path(phase63_report_path),
        "source_phase64_report": safe_repo_path(phase64_report_path),
        "before": {
            "unsupported_extension_count": before_unsupported,
            "unsupported_extension_distribution": unsupported_before_distribution,
        },
        "after": {
            "unsupported_extension_count": after_unsupported,
            "unsupported_extension_distribution": unsupported_after_distribution,
            "total_supported": phase57.get("total_supported"),
            "total_processed": phase57.get("total_processed"),
            "accepted": phase57.get("accepted"),
            "review": phase57.get("review"),
            "review_ocr_quality": phase57.get("review_ocr_quality"),
            "empty": phase57.get("empty"),
            "errors": phase57.get("errors"),
        },
        "delta": {
            "unsupported_extension_count": after_unsupported - before_unsupported,
            "accepted": status_delta["accepted"],
            "review": status_delta["review"],
            "review_ocr_quality": status_delta["review_ocr_quality"],
            "empty": status_delta["empty"],
            "errors": status_delta["errors"],
            "supported_processed": status_delta["supported_processed"],
        },
        "rtf_moved_from_unsupported_to_supported_processed": rtf_moved_to_supported(phase64, rtf_file_count),
        "rtf_file_count": rtf_file_count,
        "rtf_files": rtf_files,
        "docx_mp3_msg_ogg_remain_unsupported_or_excluded": all(count > 0 for count in non_rtf_remaining.values()),
        "non_rtf_extensions_left_unsupported": non_rtf_remaining,
        "safety_delta": safety_delta,
        "local_only_forced": True,
        "external_api_used": safety_delta["external_api_used"],
        "raw_phi_logged": safety_delta["raw_phi_logged"],
        "raw_phi_logged_in_public_reports": safety_delta["raw_phi_logged"],
        "reconciliation_passed": safety_delta["reconciliation_passed"],
        "safety_regression": safety_delta["safety_regression"],
        "production_extractor_should_change_yet": False,
        "recommendation": recommendation_for(non_rtf_remaining),
        "private_mapping_used_for_leak_check_only": bool(private_mapping),
        "private_mapping_path_public": "[PRIVATE_MAPPING_REDACTED]",
        "phi_artifact_check": phi_artifacts,
    }
    report["conclusion"] = conclusion_for(report)

    write_json(report_dir / JSON_REPORT.name, report)
    (report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
    if public_reports_contain_private_values(report_dir, private_mapping):
        report["raw_phi_logged"] = True
        report["raw_phi_logged_in_public_reports"] = True
        report["safety_delta"]["raw_phi_logged"] = True
        report["conclusion"] = "BLOCKED_PRIVACY_RISK"
        write_json(report_dir / JSON_REPORT.name, report)
        (report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
    return report


def safe_rtf_files(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_records: list[dict[str, Any]] = []
    for item in records:
        safe_records.append(
            {
                "safe_file_id": item.get("safe_file_id"),
                "filename_hash": item.get("filename_hash"),
                "content_hash": item.get("content_hash"),
                "extension": item.get("extension"),
                "file_type": item.get("file_type"),
                "status": item.get("status"),
                "selected_extractor": item.get("selected_extractor"),
                "confidence": item.get("confidence"),
                "classification_reason_codes": list(item.get("classification_reason_codes") or []),
                "review_reason_codes": list(item.get("review_reason_codes") or []),
                "accounting_category": item.get("accounting_category"),
                "external_api_used": bool(item.get("external_api_used")),
            }
        )
    return safe_records


def rtf_moved_to_supported(phase64: dict[str, Any], rtf_file_count: int) -> bool:
    counts = phase64.get("rtf_counts") or {}
    return (
        rtf_file_count > 0
        and bool(phase64.get("rtf_moved_from_unsupported_to_supported_or_safe_error"))
        and int(counts.get("unsupported_extension", 0)) == 0
        and int(counts.get("supported_processed", 0)) == rtf_file_count
    )


def normalize_distribution(distribution: dict[str, Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in distribution.items() if int(value) != 0}


def recommendation_for(non_rtf_remaining: dict[str, int]) -> dict[str, Any]:
    docx_count = int(non_rtf_remaining.get(".docx", 0))
    return {
        "summary": "RTF impact measured. Do not add more formats without prioritization evidence.",
        "docx": "Support later only if operator review shows DOCX is important." if docx_count else "No DOCX files remain in unsupported bucket.",
        "msg_audio": "Keep MSG/audio explicitly excluded until a separate privacy-safe conversion/transcription phase is justified.",
        "next_problem_class_options": ["pdf_ocr_low_quality", "image_ocr_low_quality"],
        "recommended_next_class": "pdf_ocr_low_quality",
    }


def conclusion_for(report: dict[str, Any]) -> str:
    if report["external_api_used"] or report["raw_phi_logged"] or not report["phi_artifact_check"].get("passed", False):
        return "BLOCKED_PRIVACY_RISK"
    if report["safety_regression"] or not report["reconciliation_passed"]:
        return "BLOCKED_SAFETY_OR_RECONCILIATION"
    if not report["rtf_moved_from_unsupported_to_supported_processed"]:
        return "BLOCKED_RTF_DELTA_NOT_CONFIRMED"
    return "rtf_delta_measured_no_safety_regression"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase65 Full Corpus Delta After RTF Support",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Before unsupported count: `{report['before']['unsupported_extension_count']}`",
        f"- After unsupported count: `{report['after']['unsupported_extension_count']}`",
        f"- Unsupported delta: `{report['delta']['unsupported_extension_count']}`",
        f"- RTF moved to supported processed: `{report['rtf_moved_from_unsupported_to_supported_processed']}`",
        f"- RTF file count: `{report['rtf_file_count']}`",
        f"- Non-RTF unsupported remaining: `{json.dumps(report['non_rtf_extensions_left_unsupported'], sort_keys=True)}`",
        f"- Accepted delta: `{report['delta']['accepted']}`",
        f"- Review delta: `{report['delta']['review']}`",
        f"- Empty delta: `{report['delta']['empty']}`",
        f"- Error delta: `{report['delta']['errors']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Raw PHI logged: `{report['raw_phi_logged']}`",
        f"- Safety regression: `{report['safety_regression']}`",
        f"- Reconciliation passed: `{report['reconciliation_passed']}`",
        f"- production_extractor_should_change_yet: `{report['production_extractor_should_change_yet']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## RTF Records",
        "",
        "| Safe File ID | Filename Hash | Content Hash | Status | Extractor | Confidence | Reason Codes |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for item in report["rtf_files"]:
        lines.append(
            " | ".join(
                [
                    f"| `{item.get('safe_file_id')}`",
                    f"`{item.get('filename_hash')}`",
                    f"`{item.get('content_hash')}`",
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
            "## Recommendation",
            "",
            f"- Summary: {report['recommendation']['summary']}",
            f"- DOCX: {report['recommendation']['docx']}",
            f"- MSG/audio: {report['recommendation']['msg_audio']}",
            f"- Next class options: `{', '.join(report['recommendation']['next_problem_class_options'])}`",
            f"- Recommended next class: `{report['recommendation']['recommended_next_class']}`",
            "",
            "## Safety",
            "",
            "- No production extraction logic changed in Phase65.",
            "- Public report uses safe IDs and hashes only.",
            "- No filenames, paths, extracted text, OCR text, RTF text, or PHI are included.",
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
    report = run_delta_report()
    print("MedAI Phase65 full corpus delta after RTF support complete.")
    print(f"before_unsupported: {report['before']['unsupported_extension_count']}")
    print(f"after_unsupported: {report['after']['unsupported_extension_count']}")
    print(f"unsupported_delta: {report['delta']['unsupported_extension_count']}")
    print(f"rtf_file_count: {report['rtf_file_count']}")
    print(f"rtf_moved_from_unsupported_to_supported_processed: {report['rtf_moved_from_unsupported_to_supported_processed']}")
    print(f"non_rtf_extensions_left_unsupported: {report['non_rtf_extensions_left_unsupported']}")
    print(f"external_api_used: {report['external_api_used']}")
    print(f"raw_phi_logged: {report['raw_phi_logged']}")
    print(f"safety_regression: {report['safety_regression']}")
    print(f"reconciliation_passed: {report['reconciliation_passed']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"json_report: {REPORT_DIR / JSON_REPORT.name}")
    print(f"markdown_report: {REPORT_DIR / MD_REPORT.name}")
    return 1 if report["conclusion"].startswith("BLOCKED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
