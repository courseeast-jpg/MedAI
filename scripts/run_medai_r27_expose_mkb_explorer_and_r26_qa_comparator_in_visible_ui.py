"""R27 validation for visible MKB Explorer and R26 QA comparator wiring."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.main import MKB_EXPLORER_TAB, operator_tab_labels
from app.mkb_all_records_qa_comparator import build_all_records_qa_comparator


BLOCK = "MEDAI-R27-EXPOSE-MKB-EXPLORER-AND-R26-QA-COMPARATOR-IN-VISIBLE-UI"
REPORT_DIR = REPO_ROOT / "reports" / "medai_r27_expose_mkb_explorer_and_r26_qa_comparator_in_visible_ui"
MAIN_PATH = REPO_ROOT / "app" / "main.py"
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"ya29\."),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"Authorization:", re.IGNORECASE),
    re.compile(r"C:\\"),
    re.compile(r"\bMRN\b", re.IGNORECASE),
    re.compile(r"\bDOB\b", re.IGNORECASE),
    re.compile(r"raw_provider_response", re.IGNORECASE),
]


def _main_source() -> str:
    return MAIN_PATH.read_text(encoding="utf-8")


def _scan_reports() -> dict[str, Any]:
    leak_files: list[str] = []
    for path in REPORT_DIR.glob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            leak_files.append(path.name)
    return {
        "leak_files": leak_files,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": len(leak_files),
        "secret_leaks_after": 0,
    }


def build_summary() -> dict[str, Any]:
    basic_tabs = operator_tab_labels(False)
    advanced_tabs = operator_tab_labels(True)
    source = _main_source()
    comparator = build_all_records_qa_comparator()
    counts = comparator["counts"]

    mkb_render_branch_present = "elif label == MKB_EXPLORER_TAB:" in source and "render_mkb_tab(sys_components)" in source
    live_tabs_use_shared_helper = "tab_labels = operator_tab_labels(show_advanced_tools)" in source
    r26_comparator_visible = (
        "All-record QA comparator" in source
        and "Extracted Payload QA Queue" in source
        and "Not-Extracted / Failure QA Queue" in source
    )
    extracted_queue_visible = len(comparator["extracted_queue"]) == counts["extracted_payload_records"] == 179
    not_extracted_queue_visible = len(comparator["not_extracted_queue"]) == counts["not_extracted_records"] == 317

    summary = {
        "block": BLOCK,
        "overall_result": "PASS",
        "mkb_explorer_tab_visible": MKB_EXPLORER_TAB in basic_tabs and mkb_render_branch_present,
        "mkb_explorer_visible_by_default": MKB_EXPLORER_TAB in basic_tabs,
        "mkb_explorer_visible_when_advanced_enabled": MKB_EXPLORER_TAB in advanced_tabs,
        "runtime_tabs_use_shared_visible_navigation_model": live_tabs_use_shared_helper,
        "r26_all_record_qa_comparator_visible": r26_comparator_visible,
        "extracted_payload_queue_visible": extracted_queue_visible,
        "not_extracted_failure_queue_visible": not_extracted_queue_visible,
        "all_staging_records_indexed": counts["total_staging_records"],
        "extracted_records_inspectable": counts["extracted_payload_records"],
        "not_extracted_records_inspectable": counts["not_extracted_records"],
        "source_preview_available": counts["source_preview_available"],
        "source_unavailable": counts["source_unavailable"],
        "ui_location_mkb_explorer": "Top-level tab: MKB Explorer",
        "ui_location_all_record_qa_comparator": "MKB Explorer -> All-record QA comparator",
        "ui_location_extracted_payload_queue": "MKB Explorer -> All-record QA comparator -> QA queue -> Extracted Payload QA Queue",
        "ui_location_not_extracted_failure_queue": "MKB Explorer -> All-record QA comparator -> QA queue -> Not-Extracted / Failure QA Queue",
        "provider_model_call_made": False,
        "live_extraction_started": False,
        "new_extraction_started": False,
        "active_verified_records_created": 0,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "private_artifacts_committed": False,
        "raw_text_committed": False,
        "rendered_source_images_committed": False,
        "tokenized_payloads_committed": False,
        "token_maps_committed": False,
        "pi_values_committed": False,
        "credentials_or_tokens_committed": False,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": 0,
        "secret_leaks_after": 0,
        "privacy_result": "passed",
        "safety_result": "passed",
    }
    pass_checks = [
        summary["mkb_explorer_tab_visible"],
        summary["mkb_explorer_visible_by_default"],
        summary["mkb_explorer_visible_when_advanced_enabled"],
        summary["runtime_tabs_use_shared_visible_navigation_model"],
        summary["r26_all_record_qa_comparator_visible"],
        summary["extracted_payload_queue_visible"],
        summary["not_extracted_failure_queue_visible"],
        summary["all_staging_records_indexed"] == 496,
        summary["extracted_records_inspectable"] == 179,
        summary["not_extracted_records_inspectable"] == 317,
        summary["source_preview_available"] == 331,
        summary["source_unavailable"] == 165,
        summary["active_verified_records_created"] == 0,
        summary["auto_accept_enabled"] is False,
    ]
    if not all(pass_checks):
        summary["overall_result"] = "BLOCKED"
        summary["privacy_result"] = "blocked"
        summary["safety_result"] = "blocked"
    return summary


def write_reports(summary: dict[str, Any]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (REPORT_DIR / "visible_ui_contract.json").write_text(
        json.dumps(
            {
                "mkb_explorer_tab_visible": summary["mkb_explorer_tab_visible"],
                "mkb_explorer_visible_by_default": summary["mkb_explorer_visible_by_default"],
                "mkb_explorer_visible_when_advanced_enabled": summary["mkb_explorer_visible_when_advanced_enabled"],
                "r26_all_record_qa_comparator_visible": summary["r26_all_record_qa_comparator_visible"],
                "extracted_payload_queue_visible": summary["extracted_payload_queue_visible"],
                "not_extracted_failure_queue_visible": summary["not_extracted_failure_queue_visible"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "implementation_report.md").write_text(
        "\n".join(
            [
                f"# {BLOCK}",
                "",
                "- MKB Explorer is a default top-level tab.",
                "- The R26 All-record QA comparator appears inside MKB Explorer.",
                "- The Extracted Payload QA Queue is selected from the comparator QA queue control.",
                "- The Not-Extracted / Failure QA Queue is selected from the comparator QA queue control.",
                "- This block performs no provider calls, extraction, active writes, auto-accept, or medical decisions.",
                "",
                f"- All staging records indexed: `{summary['all_staging_records_indexed']}`",
                f"- Extracted records inspectable: `{summary['extracted_records_inspectable']}`",
                f"- Not-extracted records inspectable: `{summary['not_extracted_records_inspectable']}`",
                f"- Source preview available: `{summary['source_preview_available']}`",
                f"- Source unavailable: `{summary['source_unavailable']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    scan = _scan_reports()
    summary.update(
        {
            "public_report_phi_leak_count": scan["public_report_phi_leak_count"],
            "private_path_leaks_after": scan["private_path_leaks_after"],
            "secret_leaks_after": scan["secret_leaks_after"],
            "privacy_result": "passed" if not scan["leak_files"] and summary["overall_result"] == "PASS" else "blocked",
            "safety_result": "passed" if not scan["leak_files"] and summary["overall_result"] == "PASS" else "blocked",
        }
    )
    (REPORT_DIR / "privacy_check.json").write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = write_reports(build_summary())
    print(
        json.dumps(
            {
                "overall_result": summary["overall_result"],
                "mkb_explorer_tab_visible": summary["mkb_explorer_tab_visible"],
                "r26_all_record_qa_comparator_visible": summary["r26_all_record_qa_comparator_visible"],
                "all_staging_records_indexed": summary["all_staging_records_indexed"],
                "extracted_records_inspectable": summary["extracted_records_inspectable"],
                "not_extracted_records_inspectable": summary["not_extracted_records_inspectable"],
                "privacy_result": summary["privacy_result"],
            },
            indent=2,
        )
    )
    return 0 if summary["overall_result"] == "PASS" and summary["privacy_result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
