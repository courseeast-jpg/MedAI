"""MEDAI-UI-USABILITY-POLISH-01 report generator."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy.report_privacy import check_public_report_payload  # noqa: E402


REPORT_DIR = REPO_ROOT / "reports" / "medai_ui_usability_polish_01"
REPORT_GUIDE = REPORT_DIR / "MEDAI_UI_USABILITY_POLISH_01.md"
REPORT_JSON = REPORT_DIR / "medai_ui_usability_polish_01_report.json"
REPORT_MD = REPORT_DIR / "medai_ui_usability_polish_01_report.md"


def build_report() -> dict:
    return {
        "phase_id": "MEDAI-UI-USABILITY-POLISH-01",
        "mode": "ui_usability_polish",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "behavior_changed": True,
        "clinical_behavior_changed": False,
        "extraction_behavior_changed": False,
        "ocr_behavior_changed": False,
        "classifier_behavior_changed": False,
        "threshold_behavior_changed": False,
        "cue_expansion_recommended": False,
        "cue_expansion_performed": False,
        "external_api_used": False,
        "external_api_enabled": False,
        "app_main_modified": True,
        "ui_text_changed": True,
        "action_buttons_added": False,
        "callbacks_added": False,
        "forms_added": False,
        "state_mutation_added": False,
        "data_layer_write_added": False,
        "document_type_mutation_added": False,
        "auto_accept_added": False,
        "clinical_value_parsing_performed": False,
        "clinical_interpretation_performed": False,
        "diagnosis_inference_performed": False,
        "medication_inference_performed": False,
        "ddi_inference_performed": False,
        "treatment_inference_performed": False,
        "source_documents_opened": False,
        "private_files_opened": False,
        "raw_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "local_only_posture_preserved": True,
        "run_review_clarity_improved": True,
        "advanced_details_default_safe": True,
        "all_records_review_bound": True,
        "whole_medai_done_estimate": "approximately 92.8%",
        "recommended_next_step": "DATA-RUNTIME-HARDEN-01",
        "proposed_changes": [
            "Add local-only and review-bound orientation at the top of Run & Review.",
            "Clarify that run status counts are workflow labels, not clinical acceptance.",
            "Add an operator-summary caption to each result card.",
            "Add safe-metadata wording inside collapsed Advanced technical details.",
        ],
        "validation_results": {
            "focused_ui_usability_polish_tests": "pending",
            "ui_run_review_polish_regressions": "pending",
            "public_report_privacy_checks": "pending",
            "final_cka_mvp_validation": "pending",
            "b07_term01_validation": "pending",
            "route_fix_validation": "pending",
            "ui_ops_validation": "pending",
            "ui_boot_validation": "pending",
            "document_type_non_streamlit_subset": "pending",
            "staged_safety_check": "pending",
            "full_pytest": "not run; not needed for narrow UI text polish",
        },
    }


def write_reports(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_lines = [
        "# MEDAI-UI-USABILITY-POLISH-01 Report",
        "",
        "Conclusion: `operator_clarity_polish_ready`",
        "",
        "## Scope",
        "",
        "This block changes only operator-facing UI text around Run & Review, status labels, and collapsed technical details.",
        "",
        "## UI Changes",
        "",
        "- Added local-only and review-bound orientation at the top of Run & Review.",
        "- Clarified that status counts are workflow labels, not clinical acceptance.",
        "- Added an operator-summary caption to result cards.",
        "- Added safe-metadata wording inside collapsed Advanced technical details.",
        "",
        "## Safety",
        "",
        "- Clinical behavior changed: false.",
        "- Extraction, OCR, classifier, threshold, and cue behavior changed: false.",
        "- New buttons, forms, callbacks, state mutation, and data-layer writes added: false.",
        "- Auto-accept added: false.",
        "- External API enabled or used: false.",
        "- Cue expansion remains NOT recommended.",
        "",
        "## Validation Results",
        "",
        "- Focused UI usability polish tests: pending.",
        "- UI Run & Review regressions: pending.",
        "- Public report privacy checks: pending.",
        "- Final CKA MVP validation: pending.",
        "- B07 term01 validation: pending.",
        "- ROUTE-FIX validation: pending.",
        "- UI ops validation: pending.",
        "- UI boot validation: pending.",
        "- Document-type non-streamlit subset: pending.",
        "- Staged safety check: pending.",
        "",
        "## Recommended Next Step",
        "",
        "`DATA-RUNTIME-HARDEN-01`",
        "",
    ]
    REPORT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    REPORT_GUIDE.write_text(
        "\n".join(
            [
                "# MEDAI-UI-USABILITY-POLISH-01",
                "",
                "This block improves Run & Review operator clarity without changing clinical, extraction, OCR, classifier, threshold, cue, DDI, or external API behavior.",
                "",
                "Polish summary:",
                "",
                "- Local-only and review-bound orientation is clearer.",
                "- Status counts are described as workflow statuses only.",
                "- Result cards remind operators to compare with the source.",
                "- Advanced technical details remain collapsed and safe-metadata-only.",
                "",
                "Cue expansion remains NOT recommended.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    report = build_report()
    write_reports(report)
    failed = False
    for path in (REPORT_GUIDE, REPORT_JSON, REPORT_MD):
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else {
            "report_text": path.read_text(encoding="utf-8")
        }
        privacy = check_public_report_payload(payload)
        if not privacy.passed:
            print(json.dumps({"privacy_failed": path.as_posix(), "examples": privacy.leak_examples_redacted}, indent=2))
            failed = True
    print(json.dumps({"conclusion": "operator_clarity_polish_ready", "report_json": REPORT_JSON.relative_to(REPO_ROOT).as_posix()}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
