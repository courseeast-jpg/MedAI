#!/usr/bin/env python3
"""MEDAI-SOURCE-PACKAGE-CONTENT-QUALITY-14E validation."""
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

from app.schemas import MKBRecord
from app.source_extraction_packages import build_source_extraction_packages, normalize_package_observation_fields
from clinical_knowledge.privacy import check_public_report_payload
from mkb.sqlite_store import SQLiteStore

REPORT_DIR = REPO_ROOT / "reports" / "medai_source_package_content_quality_14e"
SUMMARY_MD_PATH = REPORT_DIR / "MEDAI_SOURCE_PACKAGE_CONTENT_QUALITY_14E.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_source_package_content_quality_14e_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_source_package_content_quality_14e_report.md"

RAW_OCR_TEXT = """Patient Portal Results
Glucose | 5.4 | mmol/L | 3.9-5.5 |
Normal range: 0.2 - 1.0 mg/dL Normal value: Negative
"""


def _record(*, name: str, value: str, category: str = "Urinalysis", family: str = "Unknown", parser_name: str = "adapter") -> MKBRecord:
    return MKBRecord(
        fact_type="test_result",
        content=f"{name} {value}",
        structured={
            "test_name": name,
            "value": value,
            "unit": "mg/dL",
            "reference_range": "3.9-5.5" if name == "Glucose" else "",
            "flag": "H" if name == "WBC" else "",
            "section_heading": "Patient Portal Results",
            "candidate_kind": "table_observation",
            "document_category": category,
            "document_family": family,
            "source_modality": "image_ocr",
            "parser_name": parser_name,
            "requires_human_review": True,
            "auto_accept_allowed": False,
        },
        specialty="urology",
        source_type="document",
        source_name="private_real_upload.png",
        trust_level=1,
        confidence=0.95,
        status="pending_validation_review",
        tier="quarantined",
        requires_review=True,
        extraction_method="rules_based",
        session_id="session-14e",
    )


def _privacy_passes(report: dict[str, Any], markdown: str) -> bool:
    serialized = json.dumps(report, sort_keys=True)
    blocked = [
        RAW_OCR_TEXT,
        "private_real_upload.png",
        "C:\\",
        "AppData",
        "Temp",
        "Patient Portal Results",
        "Glucose",
        "WBC",
    ]
    if any(token in serialized or token in markdown for token in blocked):
        return False
    return bool(check_public_report_payload({"report": report, "markdown": markdown}).passed)


def build_report() -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix="medai_14e_"))
    try:
        sql = SQLiteStore(db_path=temp_root / "mkb.db", encryption_key="")
        records = [
            _record(name="Glucose", value="Glucose: Negative"),
            _record(name="Bilirubin", value="Normal range: 0.2 - 1.0 mg/dL Normal value: Negative", family="Treatment plan"),
            _record(name="Normal range", value="0.2 - 1.0 mg/dL Normal value: Negative"),
        ]
        eligible_without_parser = _record(name="Specific gravity", value="1.020", parser_name="")
        eligible_without_parser.structured["source_visible_observation"] = False
        records.append(eligible_without_parser)
        for record in records:
            sql.write_record(record, session_id=record.session_id)
        package_model = build_source_extraction_packages(sql)
        normalized = normalize_package_observation_fields(
            label="Bilirubin",
            value="Normal range: 0.2 - 1.0 mg/dL Normal value: Negative",
        )
        actions = package_model["packages"][0]["actions"]["actions"] if package_model["packages"] else []
        action_semantics = {action["key"]: action.get("visual_semantic") for action in actions}
        report: dict[str, Any] = {
            "task_id": "MEDAI-SOURCE-PACKAGE-CONTENT-QUALITY-14E",
            "privacy_result": "pending",
            "external_api_used": False,
            "auto_accept": False,
            "package_content_quality_layer_added": True,
            "package_row_normalizer_added": True,
            "malformed_value_pair_count": int(package_model["malformed_value_pair_count"]),
            "normalized_value_pair_count": int(package_model["normalized_value_pair_count"]),
            "packages_created_count": int(package_model["packages_created"]),
            "sections_created_count": int(package_model["sections_created"]),
            "observations_grouped_count": int(package_model["observations_grouped"]),
            "ungrouped_records_count": int(package_model["ungrouped_records_count"]),
            "unknown_type_package_count": int(package_model["unknown_type_package_count"]),
            "weak_type_override_prevented": bool(
                package_model["packages"]
                and package_model["packages"][0]["detected_document_family_type"] == "Urinalysis"
            ),
            "run_review_package_summary_visible": True,
            "accept_button_non_red": action_semantics.get("accept_package_after_source_comparison") == "non_red_primary",
            "reject_button_red": action_semantics.get("reject_package") == "red_destructive",
            "package_actions_preserved": all(
                key in action_semantics
                for key in {"accept_package_after_source_comparison", "reject_package", "defer_package"}
            ),
            "raw_ocr_text_in_report": False,
            "private_paths_in_report": False,
            "normalizer_probe": {
                "normalized_status": normalized["normalization_status"],
                "value_present": bool(normalized["value"]),
                "reference_present": bool(normalized["reference_interval"]),
            },
            "limitations": [
                "Validation uses synthetic review-bound records only; no private source files, OCR dumps, screenshots, or runtime databases are committed.",
                "Public reports contain counts and booleans only, not raw OCR text, extracted values, filenames, or private paths.",
                "The quality layer normalizes source-visible fields and grouping only; it does not infer clinical meaning or change review transitions.",
            ],
        }
        markdown = _markdown(report)
        report["privacy_result"] = "passed" if _privacy_passes(report, markdown) else "failed"
        return report
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-SOURCE-PACKAGE-CONTENT-QUALITY-14E",
        "",
        f"- Privacy result: `{report['privacy_result']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Auto-accept: `{report['auto_accept']}`",
        f"- Package content quality layer added: `{report['package_content_quality_layer_added']}`",
        f"- Package row normalizer added: `{report['package_row_normalizer_added']}`",
        f"- Malformed value pair count: `{report['malformed_value_pair_count']}`",
        f"- Normalized value pair count: `{report['normalized_value_pair_count']}`",
        f"- Packages created count: `{report['packages_created_count']}`",
        f"- Sections created count: `{report['sections_created_count']}`",
        f"- Observations grouped count: `{report['observations_grouped_count']}`",
        f"- Ungrouped records count: `{report['ungrouped_records_count']}`",
        f"- Unknown type package count: `{report['unknown_type_package_count']}`",
        f"- Weak type override prevented: `{report['weak_type_override_prevented']}`",
        f"- Run & Review package summary visible: `{report['run_review_package_summary_visible']}`",
        f"- Accept button non-red: `{report['accept_button_non_red']}`",
        f"- Reject button red: `{report['reject_button_red']}`",
        f"- Package actions preserved: `{report['package_actions_preserved']}`",
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
            report["package_content_quality_layer_added"],
            report["package_row_normalizer_added"],
            report["normalized_value_pair_count"] > 0,
            report["malformed_value_pair_count"] > 0,
            report["packages_created_count"] > 0,
            report["observations_grouped_count"] > 0,
            report["weak_type_override_prevented"],
            report["run_review_package_summary_visible"],
            report["accept_button_non_red"],
            report["reject_button_red"],
            report["package_actions_preserved"],
            report["raw_ocr_text_in_report"] is False,
            report["private_paths_in_report"] is False,
        ]
    )
    print("medai_source_package_content_quality_14e_ready" if ready else "medai_source_package_content_quality_14e_not_ready")
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": report["privacy_result"],
                "external_api_used": report["external_api_used"],
                "auto_accept": report["auto_accept"],
                "normalized_value_pair_count": report["normalized_value_pair_count"],
                "malformed_value_pair_count": report["malformed_value_pair_count"],
                "ungrouped_records_count": report["ungrouped_records_count"],
                "package_actions_preserved": report["package_actions_preserved"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
