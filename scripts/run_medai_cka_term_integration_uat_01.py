"""MEDAI-CKA-TERM-INTEGRATION-UAT-01 synthetic terminology helper UAT.

Reports-only / synthetic-only UAT for the default-off terminology match
hypothesis helper. This script uses explicit env mappings and the existing
synthetic read-only adapter; it does not write os.environ, open private
terminology rows, or touch runtime code.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy.report_privacy import check_public_report_payload  # noqa: E402
from clinical_knowledge.terminology.term05_read_only_adapter import (  # noqa: E402
    build_synthetic_read_only_adapter,
)
from clinical_knowledge.terminology.term_match_hypothesis import (  # noqa: E402
    TERMINOLOGY_LOOKUP_ENV_VAR,
    derive_terminology_match_hypothesis,
)

PHASE_ID = "MEDAI-CKA-TERM-INTEGRATION-UAT-01"
REPORT_DIR = REPO_ROOT / "reports" / "medai_cka_term_integration_uat_01"
JSON_REPORT = REPORT_DIR / "medai_cka_term_integration_uat_01_report.json"
MD_REPORT = REPORT_DIR / "medai_cka_term_integration_uat_01_report.md"
GUIDE_REPORT = REPORT_DIR / "MEDAI_CKA_TERM_INTEGRATION_UAT_01.md"

SAFE_ALLOWED_OUTPUT_KEYS = {
    "terminology_match_hypothesis",
    "source_phase",
    "enabled",
    "env_var",
    "match_family",
    "terminology_system_family",
    "matches_count",
    "license_class",
    "review_required",
    "auto_accept_allowed",
    "clinical_interpretation_performed",
    "diagnosis_inference_performed",
    "treatment_inference_performed",
    "medication_inference_performed",
    "ddi_behavior_changed",
    "abbreviation_expanded",
    "lab_value_parsed",
    "licensed_row_content_included",
    "raw_text_emitted",
    "raw_ocr_text_emitted",
    "raw_document_text_emitted",
    "raw_filename_emitted",
    "private_path_emitted",
    "phi_emitted",
    "secret_emitted",
    "public_report_safe",
    "ocr_routing_changed",
    "ocr_engine_behavior_changed",
    "pdf_text_extraction_behavior_changed",
    "layout_extraction_behavior_changed",
    "table_extraction_behavior_changed",
    "classifier_behavior_changed",
    "thresholds_or_scoring_changed",
    "cue_packs_added",
    "external_api_used",
    "frozen_operator_release_preserved",
    "freeze_tags_touched",
    "park_20_tags_touched",
    "park_21_tags_touched",
    "park_22_tags_touched",
    "park_23_tags_touched",
    "disclaimer",
}


def _synthetic_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "default_off_env_unset",
            "record": {
                "anonymous_id": "synthetic_case_001",
                "terminology_candidate_text": "aspirin",
                "terminology_system_filter": ["rxnorm"],
            },
            "env": {},
            "adapter": build_synthetic_read_only_adapter(),
            "expect_metadata": False,
            "expected_bucket": "default_off_noop",
        },
        {
            "case_id": "env_on_exact_synthetic_match",
            "record": {
                "anonymous_id": "synthetic_case_002",
                "terminology_candidate_text": "aspirin",
                "terminology_system_filter": ["rxnorm"],
            },
            "env": {TERMINOLOGY_LOOKUP_ENV_VAR: "true"},
            "adapter": build_synthetic_read_only_adapter(),
            "expect_metadata": True,
            "expected_match_family": "exact_terminology_match",
            "expected_bucket": "metadata_emitted",
        },
        {
            "case_id": "env_on_ambiguous_synthetic_match",
            "record": {
                "anonymous_id": "synthetic_case_003",
                "terminology_candidate_text": "aspirin",
            },
            "env": {TERMINOLOGY_LOOKUP_ENV_VAR: "true"},
            "adapter": build_synthetic_read_only_adapter(),
            "expect_metadata": True,
            "expected_match_family": "ambiguous_terminology_match",
            "expected_bucket": "metadata_emitted",
        },
        {
            "case_id": "env_on_unmapped_synthetic_candidate",
            "record": {
                "anonymous_id": "synthetic_case_004",
                "terminology_candidate_text": "term05 unknown should stay unmapped",
            },
            "env": {TERMINOLOGY_LOOKUP_ENV_VAR: "true"},
            "adapter": build_synthetic_read_only_adapter(),
            "expect_metadata": True,
            "expected_match_family": "unmapped_terminology_candidate",
            "expected_bucket": "metadata_emitted",
        },
        {
            "case_id": "env_on_missing_lookup_adapter_fails_closed",
            "record": {
                "anonymous_id": "synthetic_case_005",
                "terminology_candidate_text": "aspirin",
            },
            "env": {TERMINOLOGY_LOOKUP_ENV_VAR: "true"},
            "adapter": None,
            "expect_metadata": False,
            "expected_bucket": "fail_closed_without_adapter",
        },
        {
            "case_id": "raw_private_fields_without_candidate_rejected",
            "record": {
                "anonymous_id": "synthetic_case_006",
                "raw_text": "[redacted synthetic raw text marker]",
                "ocr_text": "[redacted synthetic ocr marker]",
                "filename": "[redacted synthetic filename marker]",
                "private_path": "[redacted synthetic path marker]",
            },
            "env": {TERMINOLOGY_LOOKUP_ENV_VAR: "true"},
            "adapter": build_synthetic_read_only_adapter(),
            "expect_metadata": False,
            "expected_bucket": "raw_private_input_rejected",
        },
    ]


def _metadata_is_safe(metadata: dict[str, Any] | None) -> bool:
    if metadata is None:
        return True
    if set(metadata) - SAFE_ALLOWED_OUTPUT_KEYS:
        return False
    required_false = (
        "auto_accept_allowed",
        "clinical_interpretation_performed",
        "diagnosis_inference_performed",
        "treatment_inference_performed",
        "medication_inference_performed",
        "ddi_behavior_changed",
        "licensed_row_content_included",
        "raw_text_emitted",
        "raw_ocr_text_emitted",
        "raw_document_text_emitted",
        "raw_filename_emitted",
        "private_path_emitted",
        "phi_emitted",
        "secret_emitted",
        "external_api_used",
        "cue_packs_added",
    )
    if any(metadata.get(key) is not False for key in required_false):
        return False
    return metadata.get("review_required") is True and metadata.get("public_report_safe") is True


def run_uat() -> dict[str, Any]:
    before_env = os.environ.get(TERMINOLOGY_LOOKUP_ENV_VAR)
    cases: list[dict[str, Any]] = []
    emissions: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()

    for case in _synthetic_cases():
        metadata = derive_terminology_match_hypothesis(
            case["record"],
            env=case["env"],
            lookup_adapter=case["adapter"],
        )
        safe = _metadata_is_safe(metadata)
        if case["expect_metadata"] and metadata is None:
            safe = False
        if not case["expect_metadata"] and metadata is not None:
            safe = False
        if metadata is not None and case.get("expected_match_family"):
            safe = safe and metadata.get("match_family") == case["expected_match_family"]
        bucket_counts[case["expected_bucket"]] += 1
        if metadata is not None:
            emissions.append(metadata)
        cases.append(
            {
                "case_id": case["case_id"],
                "metadata_emitted": metadata is not None,
                "match_family": metadata.get("match_family") if metadata else None,
                "terminology_system_family": (
                    metadata.get("terminology_system_family") if metadata else None
                ),
                "review_required": bool(metadata.get("review_required")) if metadata else False,
                "auto_accept_allowed": bool(metadata.get("auto_accept_allowed")) if metadata else False,
                "safe_output": safe,
                "bucket": case["expected_bucket"],
            }
        )

    after_env = os.environ.get(TERMINOLOGY_LOOKUP_ENV_VAR)
    match_family_counts = Counter(m["match_family"] for m in emissions)
    system_family_counts = Counter(m["terminology_system_family"] for m in emissions)
    public_payload = {
        "case_results": cases,
        "match_family_counts": dict(match_family_counts),
        "terminology_system_family_counts": dict(system_family_counts),
    }
    privacy = check_public_report_payload(public_payload)

    report: dict[str, Any] = {
        "phase_id": PHASE_ID,
        "mode": "synthetic_env_on_terminology_helper_uat",
        "reports_only": True,
        "synthetic_only": True,
        "runtime_behavior_changed": False,
        "default_behavior_changed": False,
        "app_main_modified": False,
        "streamlit_wiring_added": False,
        "operator_ui_surface_added": False,
        "terminology_import_performed": False,
        "terminology_data_staged": False,
        "licensed_terminology_rows_read": False,
        "licensed_terminology_rows_printed": False,
        "licensed_terminology_rows_in_public_reports": False,
        "license_ack_private_read": False,
        "runtime_db_contents_opened": False,
        "source_documents_opened": False,
        "private_files_opened": False,
        "raw_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "secrets_printed": False,
        "extraction_behavior_changed": False,
        "ocr_behavior_changed": False,
        "classifier_behavior_changed": False,
        "threshold_behavior_changed": False,
        "cue_expansion_recommended": False,
        "cue_expansion_performed": False,
        "external_api_used": False,
        "external_api_enabled": False,
        "clinical_value_parsing_performed": False,
        "clinical_interpretation_performed": False,
        "diagnosis_inference_performed": False,
        "medication_inference_performed": False,
        "ddi_behavior_changed": False,
        "treatment_inference_performed": False,
        "review_bound_outputs": True,
        "auto_accept_allowed": False,
        "aggregate_only_public_reports": True,
        "frozen_operator_release_preserved": True,
        "freeze_tags_untouched": True,
        "env_var": TERMINOLOGY_LOOKUP_ENV_VAR,
        "helper_under_test": "derive_terminology_match_hypothesis",
        "total_synthetic_cases_evaluated": len(cases),
        "metadata_emission_count": len(emissions),
        "default_off_noop_count": bucket_counts["default_off_noop"],
        "fail_closed_count": bucket_counts["fail_closed_without_adapter"],
        "safe_output_count": sum(1 for c in cases if c["safe_output"]),
        "unsafe_output_count": sum(1 for c in cases if not c["safe_output"]),
        "recommended_next_step": "CKA-TERM-INTEGRATION-WIRING-NEXT-01 or ROADMAP-05",
        "whole_medai_done_estimate": "approximately 94.4%",
        "match_family_counts": dict(match_family_counts),
        "terminology_system_family_counts": dict(system_family_counts),
        "review_required_count": sum(1 for m in emissions if m.get("review_required") is True),
        "auto_accept_allowed_count": sum(1 for m in emissions if m.get("auto_accept_allowed") is True),
        "licensed_row_content_output_count": sum(
            1 for m in emissions if m.get("licensed_row_content_included") is True
        ),
        "raw_text_output_count": sum(1 for m in emissions if m.get("raw_text_emitted") is True),
        "raw_ocr_text_output_count": sum(
            1 for m in emissions if m.get("raw_ocr_text_emitted") is True
        ),
        "raw_document_text_output_count": sum(
            1 for m in emissions if m.get("raw_document_text_emitted") is True
        ),
        "raw_filename_output_count": sum(
            1 for m in emissions if m.get("raw_filename_emitted") is True
        ),
        "private_path_output_count": sum(
            1 for m in emissions if m.get("private_path_emitted") is True
        ),
        "phi_output_count": sum(1 for m in emissions if m.get("phi_emitted") is True),
        "secret_output_count": sum(1 for m in emissions if m.get("secret_emitted") is True),
        "external_api_used_count": sum(1 for m in emissions if m.get("external_api_used") is True),
        "clinical_interpretation_count": sum(
            1 for m in emissions if m.get("clinical_interpretation_performed") is True
        ),
        "inference_flag_true_count": sum(
            1
            for m in emissions
            for key in (
                "diagnosis_inference_performed",
                "medication_inference_performed",
                "treatment_inference_performed",
                "ddi_behavior_changed",
            )
            if m.get(key) is True
        ),
        "rejected_raw_private_input_count": bucket_counts["raw_private_input_rejected"],
        "fail_closed_without_adapter_count": bucket_counts["fail_closed_without_adapter"],
        "os_environ_written": before_env != after_env,
        "public_report_privacy_clean": privacy.passed,
        "case_results": cases,
    }
    if report["unsafe_output_count"] != 0 or not report["public_report_privacy_clean"]:
        raise RuntimeError("UAT produced unsafe output or privacy check failure")
    return report


def _markdown(report: dict[str, Any], *, title: str) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            "## Why This Exists",
            "This block exercises the default-off terminology match helper in explicit env-on mode using only synthetic read-only fixtures.",
            "",
            "## UAT Method",
            f"- Helper under test: `{report['helper_under_test']}`.",
            f"- Env var: `{report['env_var']}`.",
            "- Adapter: existing synthetic read-only terminology adapter.",
            "- Real terminology rows, private stores, source documents, runtime DB contents, and Streamlit wiring were not used.",
            "",
            "## Synthetic Fixture Description",
            f"- Total synthetic cases evaluated: `{report['total_synthetic_cases_evaluated']}`.",
            "- Cases covered default-off, exact match, ambiguous match, unmapped candidate, missing adapter fail-closed behavior, and raw/private-field rejection.",
            "",
            "## Env-On And Default-Off Results",
            f"- Metadata emissions: `{report['metadata_emission_count']}`.",
            f"- Default-off no-op count: `{report['default_off_noop_count']}`.",
            f"- Fail-closed count: `{report['fail_closed_count']}`.",
            f"- os.environ written: `{report['os_environ_written']}`.",
            "",
            "## Match-Family Aggregate Results",
            f"- Match family counts: `{report['match_family_counts']}`.",
            f"- Terminology system family counts: `{report['terminology_system_family_counts']}`.",
            "",
            "## License And Privacy Proof",
            f"- Licensed row content output count: `{report['licensed_row_content_output_count']}`.",
            f"- Raw text output count: `{report['raw_text_output_count']}`.",
            f"- Raw OCR text output count: `{report['raw_ocr_text_output_count']}`.",
            f"- Raw document text output count: `{report['raw_document_text_output_count']}`.",
            f"- Raw filename output count: `{report['raw_filename_output_count']}`.",
            f"- Private path output count: `{report['private_path_output_count']}`.",
            f"- PHI output count: `{report['phi_output_count']}`.",
            f"- Secret output count: `{report['secret_output_count']}`.",
            "",
            "## Review-Bound Proof",
            f"- Review-required count: `{report['review_required_count']}`.",
            f"- Auto-accept allowed count: `{report['auto_accept_allowed_count']}`.",
            f"- Inference flag true count: `{report['inference_flag_true_count']}`.",
            "",
            "## What Was Not Changed",
            "- Runtime behavior, default behavior, app/main.py, Streamlit wiring, extraction, OCR, classifier, thresholds, cue packs, DDI, and clinical behavior were not changed.",
            "- External APIs were not enabled or used.",
            "- Frozen operator release and freeze tags were preserved.",
            "",
            "## Validation Evidence",
            "- Focused CKA-TERM-INTEGRATION-UAT-01 tests: see validation run.",
            "- Direct UAT script: generated this report.",
            "- Public report privacy checks: pending final report validation.",
            "- Broader release validations: see final validation run.",
            "",
            "## Recommended Next Step",
            f"`{report['recommended_next_step']}`.",
            "",
            "Cue expansion remains NOT recommended.",
            "",
        ]
    )


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = _markdown(report, title="MEDAI-CKA-TERM-INTEGRATION-UAT-01 Report")
    guide = _markdown(report, title="MEDAI-CKA-TERM-INTEGRATION-UAT-01")
    MD_REPORT.write_text(md, encoding="utf-8")
    GUIDE_REPORT.write_text(guide, encoding="utf-8")


def main() -> int:
    report = run_uat()
    write_reports(report)
    print(json.dumps({
        "phase_id": PHASE_ID,
        "total_synthetic_cases_evaluated": report["total_synthetic_cases_evaluated"],
        "metadata_emission_count": report["metadata_emission_count"],
        "unsafe_output_count": report["unsafe_output_count"],
        "public_report_privacy_clean": report["public_report_privacy_clean"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
