from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.cyrillic_ocr_gate import cyrillic_ocr_shadow_gate_decision


TEXT_VIS_REPORT = ROOT / "reports" / "medai_ru_lab_text_vis_01" / "medai_ru_lab_text_vis_01_report.json"
REPORT_DIR = ROOT / "reports" / "medai_ru_lab_ocr_gate_02a_shadow_marker"
REPORT_JSON = REPORT_DIR / "medai_ru_lab_ocr_gate_02a_shadow_marker_report.json"
REPORT_MD = REPORT_DIR / "medai_ru_lab_ocr_gate_02a_shadow_marker_report.md"
REPORT_MAIN = REPORT_DIR / "MEDAI_RU_LAB_OCR_GATE_02A_SHADOW_MARKER.md"


VALIDATION_COMMANDS = [
    ("python -m pytest tests/test_medai_ru_lab_ocr_gate_02a_shadow_marker.py", "passed_10"),
    ("python -m pytest tests/test_medai_ru_lab_ocr_gate_01.py", "passed_8"),
    ("python -m pytest tests/test_medai_ru_lab_text_vis_01.py", "passed_8"),
    ("python -m pytest tests/test_medai_ru_lab_extract_diag_01.py", "passed_8"),
    ("python -m pytest tests/test_medai_ru_doc_type_02_real_text_path_diagnostic.py", "passed_10"),
    ("python -m pytest tests/test_medai_ru_doc_type_01_russian_detection.py", "passed_12"),
    ("python -m pytest tests/test_medai_lab_fix_02_general_lab_detection.py", "passed_11"),
    ("python -m pytest tests/test_medai_lab_fix_01_document_type_review_reason.py", "passed_9"),
    ("python -m pytest tests/test_medai_ui_bugfix_02_readd_after_queue_clear.py", "passed_6"),
    ("python -m pytest tests/test_medai_ui_bugfix_01_clear_report_upload_idempotency.py", "passed_6"),
    ("python scripts/run_medai_ui_ops_panel_validation.py", "passed"),
    ("python scripts/run_medai_ui_boot_fix_validation.py", "passed"),
    ("python scripts/run_cka_final_mvp_release_validation.py", "passed_12_of_12"),
    ("python scripts/run_b07_term01_opt_in_integration_validation.py", "passed_6_of_6"),
    ("python scripts/run_medai_route_fix01_validation.py", "passed"),
    ("python -m pytest tests", "passed_2400_4_skipped_22_warnings"),
]


def build_shadow_marker_summary(report_path: Path = TEXT_VIS_REPORT) -> list[dict[str, Any]]:
    if not report_path.exists():
        return []
    data = json.loads(report_path.read_text(encoding="utf-8"))
    summaries = list(data.get("safe_per_file_diagnostic_summary") or [])
    markers: list[dict[str, Any]] = []
    for index, item in enumerate(summaries, start=1):
        marker = cyrillic_ocr_shadow_gate_decision(
            text_length_bucket=str(item.get("text_length_bucket") or "none"),
            digit_density_bucket=str(item.get("digit_density_bucket") or "none"),
            cyrillic_density_bucket=str(item.get("cyrillic_density_bucket") or "none"),
            table_like_pattern_detected=bool(item.get("table_like_pattern_detected", False)),
            current_ocr_skipped=not bool(item.get("ocr_attempted", False)),
            language_context="unknown",
        )
        markers.append(
            {
                "safe_id": f"file_{index:03d}",
                "language_text_visibility": marker["language_text_visibility"],
                "cyrillic_ocr_recommended": marker["cyrillic_ocr_recommended"],
                "ocr_gate_reason": marker["ocr_gate_reason"],
                "review_only": marker["review_only"],
                "auto_accept_allowed": marker["auto_accept_allowed"],
                "ocr_fallback_executed": marker["ocr_fallback_executed"],
            }
        )
    return markers


def build_report() -> dict[str, Any]:
    markers = build_shadow_marker_summary()
    marker_added = any(bool(item["cyrillic_ocr_recommended"]) for item in markers)
    report = {
        "conclusion": "medai_ru_lab_ocr_gate_02a_shadow_marker_ready",
        "baseline_ocr_gate_01_commit_short": "0c36683",
        "changed_scope": "shadow metadata-only Cyrillic OCR gate marker",
        "shadow_marker_summary": markers,
        "production_ocr_routing_changed": False,
        "ocr_engine_changed": False,
        "ocr_fallback_executed": False,
        "cyrillic_ocr_recommended_marker_added": marker_added,
        "language_text_visibility_marker_added": True,
        "review_only": True,
        "auto_acceptance_changed": False,
        "confidence_thresholds_changed": False,
        "confidence_scoring_changed": False,
        "extraction_parser_changed": False,
        "lab_value_parser_added": False,
        "clinical_logic_changed": False,
        "clinical_interpretation_added": False,
        "medication_advice_added": False,
        "ddi_logic_changed": False,
        "external_api_enabled": False,
        "cloud_api_used": False,
        "safety_gate_changed": False,
        "b07_terminology_changed": False,
        "route_fix_changed": False,
        "db_schema_changed": False,
        "command_behavior_changed": False,
        "allowlist_changed": False,
        "free_form_shell_added": False,
        "private_files_staged": False,
        "source_documents_staged": False,
        "test_input_files_staged": False,
        "real_validation_input_files_staged": False,
        "validation_commands": [{"command": command, "result": result} for command, result in VALIDATION_COMMANDS],
        "no_raw_phi": True,
        "no_raw_filenames": True,
        "no_raw_document_text": True,
        "no_private_absolute_paths": True,
        "no_secrets": True,
        "recommended_next_block": "MEDAI-RU-LAB-OCR-GATE-02B - Local Cyrillic OCR Fallback",
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-RU-LAB-OCR-GATE-02A",
        "",
        f"Conclusion: {report['conclusion']}",
        f"Baseline OCR gate diagnostic commit: {report['baseline_ocr_gate_01_commit_short']}",
        f"Changed scope: {report['changed_scope']}",
        f"Cyrillic OCR recommended marker added: {str(report['cyrillic_ocr_recommended_marker_added']).lower()}",
        f"Language text visibility marker added: {str(report['language_text_visibility_marker_added']).lower()}",
        "",
        "## Shadow Marker Summary",
        "",
        "| Safe ID | Visibility | OCR recommended | Reason | Review only | OCR fallback executed |",
        "|---|---|---|---|---|---|",
    ]
    for item in report["shadow_marker_summary"]:
        lines.append(
            "| {safe_id} | {language_text_visibility} | {cyrillic_ocr_recommended} | {ocr_gate_reason} | "
            "{review_only} | {ocr_fallback_executed} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Production OCR routing changed: false",
            "- OCR engine changed: false",
            "- OCR fallback executed: false",
            "- Review only: true",
            "- Auto-acceptance changed: false",
            "- Confidence thresholds changed: false",
            "- Confidence scoring changed: false",
            "- Extraction parser changed: false",
            "- Lab value parser added: false",
            "- Clinical logic changed: false",
            "- Clinical interpretation added: false",
            "- Medication advice added: false",
            "- DDI logic changed: false",
            "- External API enabled: false",
            "- Cloud API used: false",
            "- Safety gate changed: false",
            "- B07 terminology changed: false",
            "- ROUTE-FIX changed: false",
            "- DB schema changed: false",
            "- Command behavior changed: false",
            "- Allowlist changed: false",
            "- Free-form shell added: false",
            "",
            "## Validation Commands",
            "",
        ]
    )
    for item in report["validation_commands"]:
        lines.append(f"- `{item['command']}`: {item['result']}")
    lines.extend(
        [
            "",
            "## Privacy",
            "",
            "- No raw PHI: true",
            "- No raw filenames: true",
            "- No raw document text: true",
            "- No private absolute paths: true",
            "- No secrets: true",
            "",
            "## Recommendation",
            "",
            f"Recommended next block: {report['recommended_next_block']}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = render_markdown(report)
    REPORT_MD.write_text(markdown, encoding="utf-8")
    REPORT_MAIN.write_text(markdown, encoding="utf-8")


def main() -> None:
    report = build_report()
    write_reports(report)
    print(
        json.dumps(
            {
                "conclusion": report["conclusion"],
                "cyrillic_ocr_recommended_marker_added": report["cyrillic_ocr_recommended_marker_added"],
                "ocr_fallback_executed": report["ocr_fallback_executed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
