"""Phase63 unsupported extension triage without PHI in public reports."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
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
from scripts.run_phase53_blind_pdf_generalization_audit import SUPPORTED_EXTENSIONS


PHASE57_REPORT = ROOT / "reports" / "phase57_full_corpus_inventory_audit" / "phase57_full_corpus_inventory_audit_report.json"
PHASE57_PRIVATE_MAPPING = (
    ROOT / "reports" / "phase57_full_corpus_inventory_audit" / "local_filename_mapping_PRIVATE.json"
)
REPORT_DIR = ROOT / "reports" / "phase63_unsupported_extension_triage"
JSON_REPORT = REPORT_DIR / "phase63_unsupported_extension_triage_report.json"
MD_REPORT = REPORT_DIR / "phase63_unsupported_extension_triage_report.md"

TEXT_LIKE_EXTENSIONS = {".rtf", ".md", ".csv", ".tsv", ".log"}
OFFICE_DOCUMENT_EXTENSIONS = {".docx", ".doc", ".odt", ".pages"}
ARCHIVE_OR_CONTAINER_EXTENSIONS = {".zip", ".7z", ".rar", ".tar", ".gz", ".msg", ".eml"}
AUDIO_VIDEO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".m4a", ".mp4", ".mov"}


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


def run_triage(
    *,
    phase57_report_path: Path = PHASE57_REPORT,
    private_mapping_path: Path = PHASE57_PRIVATE_MAPPING,
    report_dir: Path = REPORT_DIR,
) -> dict[str, Any]:
    force_local_only_runtime()
    report_dir.mkdir(parents=True, exist_ok=True)
    phase57 = load_json(phase57_report_path)
    private_mapping = load_json(private_mapping_path)
    unsupported = select_unsupported_records(phase57, private_mapping)
    grouped = group_by_extension(unsupported)
    action_by_extension = {
        extension: classify_extension(extension, records)
        for extension, records in sorted(grouped.items(), key=lambda item: item[0])
    }
    phi_artifacts = phi_artifact_tracking_status()
    public_report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase63 Unsupported Extension Triage and Narrow Format Support Decision",
        "source_phase57_report": safe_repo_path(phase57_report_path),
        "unsupported_count": len(unsupported),
        "extension_distribution": dict(Counter(item["extension"] for item in unsupported)),
        "recommended_action_by_extension": action_by_extension,
        "unsupported_files": unsupported,
        "phase64_recommendation": phase64_recommendation(action_by_extension),
        "production_extractor_should_change_yet": False,
        "local_only_forced": True,
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "private_mapping_used_locally": bool(private_mapping),
        "private_mapping_path_public": "[PRIVATE_MAPPING_REDACTED]",
        "supported_formats_unchanged": sorted(SUPPORTED_EXTENSIONS),
        "phi_artifact_check": phi_artifacts,
        "conclusion": conclusion_for(action_by_extension, len(unsupported), phi_artifacts),
    }
    write_json(report_dir / JSON_REPORT.name, public_report)
    (report_dir / MD_REPORT.name).write_text(render_markdown(public_report), encoding="utf-8")
    if public_reports_contain_private_values(report_dir, private_mapping):
        public_report["raw_phi_logged_in_public_reports"] = True
        public_report["conclusion"] = "BLOCKED_PRIVACY_RISK"
        write_json(report_dir / JSON_REPORT.name, public_report)
        (report_dir / MD_REPORT.name).write_text(render_markdown(public_report), encoding="utf-8")
    return public_report


def select_unsupported_records(phase57: dict[str, Any], private_mapping: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = private_mapping.get("files", {}) if isinstance(private_mapping, dict) else {}
    unsupported: list[dict[str, Any]] = []
    for item in phase57.get("results", []):
        safe_id = str(item.get("safe_file_id") or item.get("file_id") or "")
        private_entry = mapping.get(safe_id, {})
        accounting_category = str(private_entry.get("accounting_category") or item.get("accounting_category") or "")
        reason_codes = {str(code) for code in item.get("reason_codes") or item.get("classification_reason_codes") or []}
        is_unsupported = (
            accounting_category == "unsupported_extension"
            or item.get("file_type") == "unsupported"
            or item.get("error_category") == "unsupported_format"
            or "unsupported_format" in reason_codes
        )
        if not is_unsupported:
            continue
        extension = str(item.get("extension") or item.get("file_extension") or "").lower()
        if not extension and private_entry.get("original_filename"):
            extension = Path(str(private_entry["original_filename"])).suffix.lower()
        unsupported.append(
            {
                "safe_file_id": safe_id,
                "filename_hash": item.get("filename_hash") or private_entry.get("filename_hash"),
                "content_hash": item.get("content_hash") or private_entry.get("content_hash"),
                "extension": extension or "unknown",
                "file_size_bytes": item.get("file_size_bytes") or private_entry.get("file_size_bytes"),
                "accounting_category": "unsupported_extension",
            }
        )
    return sorted(unsupported, key=lambda row: row["safe_file_id"])


def group_by_extension(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["extension"]).lower()].append(record)
    return dict(grouped)


def classify_extension(extension: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    extension = extension.lower()
    safe_ids = [record["safe_file_id"] for record in records]
    base: dict[str, Any] = {
        "extension": extension,
        "file_count": len(records),
        "safe_file_ids": safe_ids,
        "production_extractor_should_change_yet": False,
    }
    if extension in TEXT_LIKE_EXTENSIONS:
        return {
            **base,
            "classification": "safe_to_support_now",
            "recommended_action": "support_prototype",
            "operator_guidance": "RTF/text-like format can be evaluated in a narrow local-only prototype. Keep output review-gated until validated.",
            "rationale": "Text-like formats can be parsed locally without cloud calls or OCR routing changes.",
        }
    if extension in OFFICE_DOCUMENT_EXTENSIONS:
        return {
            **base,
            "classification": "support_later",
            "recommended_action": "future_privacy_safe_document_conversion_phase",
            "operator_guidance": "Convert manually to PDF/TXT for now, or queue a future local-only office-document conversion phase.",
            "rationale": "Office documents are containers and may contain metadata/embedded media. Add support only in a separate privacy-reviewed phase.",
        }
    if extension in ARCHIVE_OR_CONTAINER_EXTENSIONS:
        return {
            **base,
            "classification": "explicit_exclusion",
            "recommended_action": "exclude_container_format_from_medai_ingest",
            "operator_guidance": "Do not process this container directly. Export the relevant medical document to PDF/TXT/image first.",
            "rationale": "Container/email/archive formats can hide filenames, attachments, metadata, and nested PHI.",
        }
    if extension in AUDIO_VIDEO_EXTENSIONS:
        return {
            **base,
            "classification": "explicit_exclusion",
            "recommended_action": "exclude_audio_video_from_medai_document_ingest",
            "operator_guidance": "Audio/video transcription is out of scope for the current document OCR/Layout release.",
            "rationale": "Audio/video requires a separate local transcription pipeline and different privacy controls.",
        }
    return {
        **base,
        "classification": "manual_review_only",
        "recommended_action": "manual_review_unknown_binary_format",
        "operator_guidance": "Manually inspect and convert to a supported format before MedAI processing.",
        "rationale": "Unknown or binary formats should fail closed until a narrow local-only support plan exists.",
    }


def phase64_recommendation(action_by_extension: dict[str, dict[str, Any]]) -> dict[str, Any]:
    safe_now = [
        extension
        for extension, action in sorted(action_by_extension.items())
        if action.get("classification") == "safe_to_support_now"
    ]
    if safe_now:
        return {
            "recommended_next_phase": "Phase64 Narrow RTF/Text-Like Local Parser Prototype",
            "target_extensions": safe_now,
            "production_extractor_should_change_yet": False,
            "reason": "Only text-like unsupported extensions should be considered next, and only as a review-gated local prototype.",
        }
    return {
        "recommended_next_phase": "No immediate format-support implementation",
        "target_extensions": [],
        "production_extractor_should_change_yet": False,
        "reason": "No unsupported extension is simple enough for safe immediate support.",
    }


def conclusion_for(action_by_extension: dict[str, dict[str, Any]], unsupported_count: int, phi_artifacts: dict[str, Any]) -> str:
    if not phi_artifacts.get("passed", False):
        return "BLOCKED_PRIVACY_RISK"
    if unsupported_count == 0:
        return "no_unsupported_extensions"
    if any(action.get("classification") == "safe_to_support_now" for action in action_by_extension.values()):
        return "triage_complete_support_prototype_recommended"
    return "triage_complete_manual_or_later_only"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase63 Unsupported Extension Triage",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Unsupported count: `{report['unsupported_count']}`",
        f"- Local-only forced: `{report['local_only_forced']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Raw PHI logged in public reports: `{report['raw_phi_logged_in_public_reports']}`",
        f"- Production extractor should change yet: `{report['production_extractor_should_change_yet']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## Extension Distribution",
        "",
    ]
    for extension, count in sorted(report["extension_distribution"].items()):
        lines.append(f"- `{extension}`: `{count}`")
    lines.extend(["", "## Recommended Action By Extension", ""])
    for extension, action in sorted(report["recommended_action_by_extension"].items()):
        lines.extend(
            [
                f"### `{extension}`",
                f"- File count: `{action['file_count']}`",
                f"- Classification: `{action['classification']}`",
                f"- Recommended action: `{action['recommended_action']}`",
                f"- Safe file IDs: `{', '.join(action['safe_file_ids'])}`",
                f"- Guidance: {action['operator_guidance']}",
                f"- Rationale: {action['rationale']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Phase64 Recommendation",
            "",
            f"- Next phase: `{report['phase64_recommendation']['recommended_next_phase']}`",
            f"- Target extensions: `{', '.join(report['phase64_recommendation']['target_extensions']) or 'none'}`",
            f"- Production extractor should change yet: `{report['phase64_recommendation']['production_extractor_should_change_yet']}`",
            f"- Reason: {report['phase64_recommendation']['reason']}",
            "",
            "## Safe Unsupported Files",
            "",
            "| Safe File ID | Filename Hash | Content Hash | Extension | Accounting Category |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in report["unsupported_files"]:
        lines.append(
            f"| `{item['safe_file_id']}` | `{item['filename_hash']}` | `{item['content_hash']}` | `{item['extension']}` | `{item['accounting_category']}` |"
        )
    return "\n".join(lines) + "\n"


def public_reports_contain_private_values(report_dir: Path, private_mapping: dict[str, Any]) -> bool:
    if not private_mapping:
        return False
    values: list[str] = []
    for entry in private_mapping.get("files", {}).values():
        for key in ("original_filename", "original_relative_path"):
            value = str(entry.get(key) or "")
            if value and value != "[OUTSIDE_INPUT_ROOT]":
                values.append(value)
    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in [report_dir / JSON_REPORT.name, report_dir / MD_REPORT.name]
        if path.exists()
    )
    return any(value in public_text for value in values)


def safe_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return "[REDACTED_PATH]"


def main() -> int:
    report = run_triage()
    print("MedAI Phase63 unsupported extension triage complete.")
    print(f"unsupported_count: {report['unsupported_count']}")
    print(f"extension_distribution: {report['extension_distribution']}")
    print(f"production_extractor_should_change_yet: {report['production_extractor_should_change_yet']}")
    print(f"external_api_used: {report['external_api_used']}")
    print(f"raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"json_report: {JSON_REPORT}")
    print(f"markdown_report: {MD_REPORT}")
    return 1 if report["conclusion"] == "BLOCKED_PRIVACY_RISK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
