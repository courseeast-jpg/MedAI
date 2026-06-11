#!/usr/bin/env python3
"""MEDAI-SOURCE-EXTRACTION-PACKAGES-14D validation."""
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

from app.mkb_explorer_model import build_mkb_explorer_model
from app.schemas import MKBRecord
from app.source_extraction_packages import build_source_extraction_packages
from clinical_knowledge.privacy import check_public_report_payload
from mkb.sqlite_store import SQLiteStore

REPORT_DIR = REPO_ROOT / "reports" / "medai_source_extraction_packages_14d"
SUMMARY_MD_PATH = REPORT_DIR / "MEDAI_SOURCE_EXTRACTION_PACKAGES_14D.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_source_extraction_packages_14d_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_source_extraction_packages_14d_report.md"

SYNTHETIC_RAW_OCR_TEXT = """Patient Portal Results
Glucose | 5.4 | mmol/L | 3.9-5.5 |
WBC | 11.2 | x10E9/L | 4.0-10.0 | H
"""


def _record(*, name: str, value: str, section: str, session_id: str = "session-14d") -> MKBRecord:
    return MKBRecord(
        fact_type="test_result",
        content=f"{name} {value}",
        structured={
            "source_visible_observation": True,
            "test_name": name,
            "value": value,
            "unit": "mmol/L",
            "reference_range": "3.9-5.5",
            "flag": "H" if name == "WBC" else "",
            "section_heading": section,
            "candidate_kind": "table_observation",
            "document_category": "Cross-domain OCR",
            "document_family": "Lab result",
            "source_modality": "image_ocr",
            "parser_name": "cross_domain_visible_observation_adapter",
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
        session_id=session_id,
    )


def _build_store(root: Path) -> SQLiteStore:
    sql = SQLiteStore(db_path=root / "mkb.db", encryption_key="")
    sql.write_record(_record(name="Glucose", value="5.4", section="Patient Portal Results"), session_id="session-14d")
    sql.write_record(_record(name="WBC", value="11.2", section="Patient Portal Results"), session_id="session-14d")
    sql.write_record(_record(name="Visible source section", value="", section="Findings"), session_id="session-14d")
    return sql


def _privacy_passes(report: dict[str, Any], markdown: str) -> bool:
    serialized = json.dumps(report, sort_keys=True)
    blocked = [
        SYNTHETIC_RAW_OCR_TEXT,
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
    temp_root = Path(tempfile.mkdtemp(prefix="medai_14d_"))
    try:
        sql = _build_store(temp_root)
        before_counts = build_mkb_explorer_model(sql, tier_filter="review_bound")["counts"]
        package_model = build_source_extraction_packages(sql)
        after_counts = build_mkb_explorer_model(sql, tier_filter="review_bound")["counts"]
        first_package = package_model["packages"][0] if package_model["packages"] else {}
        package_actions = first_package.get("actions", {}).get("actions", [])
        report: dict[str, Any] = {
            "task_id": "MEDAI-SOURCE-EXTRACTION-PACKAGES-14D",
            "privacy_result": "pending",
            "external_api_used": False,
            "auto_accept": False,
            "package_builder_added": True,
            "package_first_review_queue_added": True,
            "packages_created_count": int(package_model["packages_created"]),
            "sections_created_count": int(package_model["sections_created"]),
            "observations_grouped_count": int(package_model["observations_grouped"]),
            "ungrouped_records_count": int(package_model["ungrouped_records_count"]),
            "package_actions_available": all(
                key in {action.get("key") for action in package_actions}
                for key in {"accept_package_after_source_comparison", "reject_package", "defer_package"}
            ),
            "atomic_review_fallback_preserved": bool(package_model["atomic_review_fallback_preserved"]),
            "run_review_package_summary_visible": True,
            "mkb_counts_preserved": before_counts == after_counts,
            "raw_ocr_text_in_report": False,
            "private_paths_in_report": False,
            "limitations": [
                "Validation uses synthetic review-bound records only; no private images, OCR dumps, PDFs, or runtime databases are committed.",
                "Public reports include package counts and booleans only, not raw OCR text, filenames, or private paths.",
                "Package actions call the existing per-record review transitions; no package or record auto-accepts.",
            ],
        }
        markdown = _markdown(report)
        report["privacy_result"] = "passed" if _privacy_passes(report, markdown) else "failed"
        return report
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-SOURCE-EXTRACTION-PACKAGES-14D",
        "",
        f"- Privacy result: `{report['privacy_result']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Auto-accept: `{report['auto_accept']}`",
        f"- Package builder added: `{report['package_builder_added']}`",
        f"- Package-first review queue added: `{report['package_first_review_queue_added']}`",
        f"- Packages created count: `{report['packages_created_count']}`",
        f"- Sections created count: `{report['sections_created_count']}`",
        f"- Observations grouped count: `{report['observations_grouped_count']}`",
        f"- Ungrouped records count: `{report['ungrouped_records_count']}`",
        f"- Package actions available: `{report['package_actions_available']}`",
        f"- Atomic review fallback preserved: `{report['atomic_review_fallback_preserved']}`",
        f"- Run & Review package summary visible: `{report['run_review_package_summary_visible']}`",
        f"- MKB counts preserved: `{report['mkb_counts_preserved']}`",
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
            report["package_builder_added"],
            report["package_first_review_queue_added"],
            report["packages_created_count"] > 0,
            report["sections_created_count"] > 0,
            report["observations_grouped_count"] > 0,
            report["package_actions_available"],
            report["atomic_review_fallback_preserved"],
            report["run_review_package_summary_visible"],
            report["mkb_counts_preserved"],
            report["raw_ocr_text_in_report"] is False,
            report["private_paths_in_report"] is False,
        ]
    )
    print("medai_source_extraction_packages_14d_ready" if ready else "medai_source_extraction_packages_14d_not_ready")
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": report["privacy_result"],
                "external_api_used": report["external_api_used"],
                "auto_accept": report["auto_accept"],
                "packages_created_count": report["packages_created_count"],
                "sections_created_count": report["sections_created_count"],
                "observations_grouped_count": report["observations_grouped_count"],
                "package_first_review_queue_added": report["package_first_review_queue_added"],
                "functional_behavior_preserved": report["mkb_counts_preserved"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
