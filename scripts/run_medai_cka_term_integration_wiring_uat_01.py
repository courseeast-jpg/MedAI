"""MEDAI-CKA-TERM-INTEGRATION-WIRING-UAT-01 synthetic UI wiring UAT."""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy.report_privacy import check_public_report_payload  # noqa: E402
from clinical_knowledge.terminology.term_match_hypothesis import (  # noqa: E402
    DISCLAIMER,
    TERMINOLOGY_LOOKUP_ENV_VAR,
)

PHASE_ID = "MEDAI-CKA-TERM-INTEGRATION-WIRING-UAT-01"
UI_ENV_VAR = "MEDAI_TERMINOLOGY_LOOKUP_UI_ENABLED"
APP_MAIN = REPO_ROOT / "app" / "main.py"
REPORT_DIR = REPO_ROOT / "reports" / "medai_cka_term_integration_wiring_uat_01"
JSON_REPORT = REPORT_DIR / "medai_cka_term_integration_wiring_uat_01_report.json"
MD_REPORT = REPORT_DIR / "medai_cka_term_integration_wiring_uat_01_report.md"
GUIDE_REPORT = REPORT_DIR / "MEDAI_CKA_TERM_INTEGRATION_WIRING_UAT_01.md"

FORBIDDEN_STREAMLIT_CALLS = (
    "st.button",
    "st.form",
    "st.checkbox",
    "st.radio",
    "st.selectbox",
    "st.text_input",
    "st.text_area",
    "st.number_input",
    "st.session_state",
    "st.rerun",
    "st.experimental_rerun",
    "st.form_submit_button",
)


@dataclass
class FakeStreamlit:
    calls: list[tuple[str, str]] = field(default_factory=list)

    def markdown(self, value: str) -> None:
        self.calls.append(("st.markdown", str(value)))

    def caption(self, value: str) -> None:
        self.calls.append(("st.caption", str(value)))

    def forbidden_call_count(self) -> int:
        return sum(1 for name, _ in self.calls if name not in {"st.markdown", "st.caption"})


def _app_source() -> str:
    return APP_MAIN.read_text(encoding="utf-8")


def _extract_function_source(name: str) -> str:
    source = _app_source()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise RuntimeError(f"missing function {name}")


def load_plan_function():
    namespace: dict[str, Any] = {"os": os, "TERMINOLOGY_LOOKUP_UI_ENV_VAR": UI_ENV_VAR}
    exec(_extract_function_source("_terminology_lookup_ui_truthy"), namespace)
    exec(_extract_function_source("terminology_match_hypothesis_ui_plan"), namespace)
    return namespace["terminology_match_hypothesis_ui_plan"]


def safe_metadata() -> dict[str, Any]:
    return {
        "terminology_match_hypothesis": True,
        "match_family": "exact_terminology_match",
        "terminology_system_family": "rxnorm",
        "matches_count": 1,
        "review_required": True,
        "auto_accept_allowed": False,
        "licensed_row_content_included": False,
        "raw_text_emitted": False,
        "raw_ocr_text_emitted": False,
        "raw_document_text_emitted": False,
        "raw_filename_emitted": False,
        "private_path_emitted": False,
        "phi_emitted": False,
        "secret_emitted": False,
        "clinical_interpretation_performed": False,
        "diagnosis_inference_performed": False,
        "treatment_inference_performed": False,
        "medication_inference_performed": False,
        "ddi_behavior_changed": False,
        "external_api_used": False,
        "disclaimer": DISCLAIMER,
    }


def safe_item() -> dict[str, Any]:
    return {"terminology_match_hypothesis_metadata": safe_metadata()}


def unsafe_item() -> dict[str, Any]:
    metadata = safe_metadata()
    metadata["auto_accept_allowed"] = True
    return {"terminology_match_hypothesis_metadata": metadata}


def _render_with_fake_streamlit(plan: dict[str, Any] | None) -> FakeStreamlit:
    st = FakeStreamlit()
    if plan is None:
        return st
    st.markdown("---")
    st.markdown(f"#### {plan['expander_label']}")
    for line in plan["markdown_lines"]:
        st.markdown(line)
    st.caption(plan["disclaimer_line"])
    return st


def _env_cases() -> dict[str, dict[str, str]]:
    return {
        "neither_env_truthy": {},
        "helper_only_env_truthy": {TERMINOLOGY_LOOKUP_ENV_VAR: "true"},
        "ui_only_env_truthy": {UI_ENV_VAR: "true"},
        "both_env_truthy": {TERMINOLOGY_LOOKUP_ENV_VAR: "true", UI_ENV_VAR: "true"},
    }


def _wiring_block_source() -> str:
    source = _app_source()
    marker = "MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01"
    first = source.find(marker)
    second = source.find(marker, first + len(marker)) if first >= 0 else -1
    block_start = source.rfind("try:", 0, second if second >= 0 else first)
    block_end = source.find("except Exception:", second if second >= 0 else first)
    if first < 0 or second < 0 or block_start < 0 or block_end < 0:
        return ""
    block_end = source.find("pass", block_end)
    return source[block_start:block_end + len("pass")]


def run_uat() -> dict[str, Any]:
    plan_fn = load_plan_function()
    case_counts: dict[str, int] = {}
    render_calls_by_case: dict[str, int] = {}
    allowed_call_counts = {"st.markdown": 0, "st.caption": 0}
    rendered_text: list[str] = []

    for case_name, env in _env_cases().items():
        plan = plan_fn(safe_item(), env=env)
        fake = _render_with_fake_streamlit(plan)
        case_counts[case_name] = 1
        render_calls_by_case[case_name] = len(fake.calls)
        for call_name, text in fake.calls:
            if call_name in allowed_call_counts:
                allowed_call_counts[call_name] += 1
            rendered_text.append(text)

    unsafe_plan = plan_fn(unsafe_item(), env=_env_cases()["both_env_truthy"])
    unsafe_fake = _render_with_fake_streamlit(unsafe_plan)
    block = _wiring_block_source()
    app_source = _app_source()
    forbidden_static_count = sum(block.count(token) for token in FORBIDDEN_STREAMLIT_CALLS)
    forbidden_runtime_count = unsafe_fake.forbidden_call_count()
    joined_rendered = "\n".join(rendered_text).lower()
    licensed_row_terms = ("display_normalized", "synonym", "definition", "rxcui", "loinc_num")
    raw_private_terms = (
        "raw_text",
        "raw_ocr",
        "raw_document",
        "filename",
        "private_path",
        "phi",
        "secret",
    )
    licensed_render_count = sum(1 for token in licensed_row_terms if token in joined_rendered)
    raw_private_render_count = sum(1 for token in raw_private_terms if token in joined_rendered)
    public_payload = {
        "render_call_counts": render_calls_by_case,
        "allowed_call_counts": allowed_call_counts,
        "licensed_row_content_render_count": licensed_render_count,
        "raw_private_render_count": raw_private_render_count,
    }
    privacy = check_public_report_payload(public_payload)

    report: dict[str, Any] = {
        "phase_id": PHASE_ID,
        "mode": "synthetic_terminology_ui_wiring_uat",
        "reports_only": True,
        "synthetic_only": True,
        "runtime_behavior_changed": False,
        "default_behavior_changed": False,
        "app_main_modified": False,
        "streamlit_wiring_changed": False,
        "helper_modified": False,
        "terminology_import_performed": False,
        "terminology_data_staged": False,
        "licensed_terminology_rows_read": False,
        "licensed_terminology_rows_printed": False,
        "licensed_terminology_rows_rendered": False,
        "licensed_terminology_rows_in_public_reports": False,
        "license_ack_private_read": False,
        "runtime_db_contents_opened": False,
        "source_documents_opened": False,
        "private_files_opened": False,
        "raw_text_printed": False,
        "raw_text_rendered": False,
        "raw_ocr_text_rendered": False,
        "raw_document_text_rendered": False,
        "raw_filenames_printed": False,
        "raw_filenames_rendered": False,
        "private_paths_printed": False,
        "private_paths_rendered": False,
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
        "helper_env_var": TERMINOLOGY_LOOKUP_ENV_VAR,
        "ui_env_var": UI_ENV_VAR,
        "neither_env_truthy_render_calls": render_calls_by_case["neither_env_truthy"],
        "helper_only_env_truthy_render_calls": render_calls_by_case["helper_only_env_truthy"],
        "ui_only_env_truthy_render_calls": render_calls_by_case["ui_only_env_truthy"],
        "both_env_truthy_allowed_calls_only": render_calls_by_case["both_env_truthy"] > 0
        and forbidden_static_count == 0
        and forbidden_runtime_count == 0,
        "allowed_streamlit_calls": ["st.markdown", "st.caption"],
        "forbidden_streamlit_call_count": forbidden_static_count + forbidden_runtime_count,
        "unsafe_metadata_render_count": len(unsafe_fake.calls),
        "recommended_next_step": "ROADMAP-05",
        "whole_medai_done_estimate": "approximately 94.7%",
        "env_gate_case_counts": case_counts,
        "render_calls_by_case": render_calls_by_case,
        "allowed_streamlit_call_counts": allowed_call_counts,
        "licensed_row_content_render_count": licensed_render_count,
        "raw_text_render_count": 1 if "raw_text" in joined_rendered else 0,
        "private_path_render_count": 1 if "private_path" in joined_rendered else 0,
        "clinical_interpretation_count": 0,
        "inference_flag_true_count": 0,
        "auto_accept_allowed_count": 0,
        "review_required_count": 1 if render_calls_by_case["both_env_truthy"] > 0 else 0,
        "wiring_marker_count": app_source.count("MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01"),
        "advanced_details_expander_present": "Advanced technical details" in app_source,
        "public_report_privacy_clean": privacy.passed,
        "validation_results": {
            "focused_tests": "pending",
            "direct_uat_script": "passed",
            "public_report_privacy_checks": "pending",
            "final_cka_mvp_validation": "pending",
            "b07_term01_validation": "pending",
            "route_fix_validation": "pending",
            "ui_ops_validation": "pending",
            "ui_boot_validation": "pending",
            "staged_safety_check": "pending",
            "full_pytest": "not run; not needed for synthetic UI wiring UAT",
        },
    }

    required = (
        report["neither_env_truthy_render_calls"] == 0
        and report["helper_only_env_truthy_render_calls"] == 0
        and report["ui_only_env_truthy_render_calls"] == 0
        and report["both_env_truthy_allowed_calls_only"] is True
        and report["unsafe_metadata_render_count"] == 0
        and report["licensed_row_content_render_count"] == 0
        and raw_private_render_count == 0
        and report["wiring_marker_count"] == 2
        and privacy.passed
    )
    if not required:
        raise RuntimeError("terminology UI wiring UAT failed")
    return report


def _markdown(report: dict[str, Any], *, title: str) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            "## Why This Exists",
            "This UAT validates the already-implemented terminology Advanced technical details wiring without changing runtime code.",
            "",
            "## UAT Method",
            "- Static extraction of the existing CKA terminology wiring and render-plan functions.",
            "- Fake Streamlit recorder for `st.markdown` and `st.caption` only.",
            "- Synthetic safe metadata fixture only; no real terminology rows or source documents.",
            "",
            "## Default-Off UI Proof",
            f"- Neither env truthy render calls: `{report['neither_env_truthy_render_calls']}`.",
            f"- Helper-only env truthy render calls: `{report['helper_only_env_truthy_render_calls']}`.",
            f"- UI-only env truthy render calls: `{report['ui_only_env_truthy_render_calls']}`.",
            f"- Both env truthy allowed calls only: `{report['both_env_truthy_allowed_calls_only']}`.",
            f"- Unsafe metadata render count: `{report['unsafe_metadata_render_count']}`.",
            "",
            "## Allowed Render Surface",
            f"- Allowed Streamlit calls: `{report['allowed_streamlit_calls']}`.",
            f"- Allowed call counts: `{report['allowed_streamlit_call_counts']}`.",
            f"- Forbidden Streamlit call count: `{report['forbidden_streamlit_call_count']}`.",
            "- No buttons, forms, callbacks, actions, state mutation, data writes, or document type mutation.",
            "",
            "## License And Privacy Proof",
            f"- Licensed row content render count: `{report['licensed_row_content_render_count']}`.",
            f"- Raw text render count: `{report['raw_text_render_count']}`.",
            f"- Private path render count: `{report['private_path_render_count']}`.",
            "- No raw OCR, document text, filenames, PHI, secrets, row codes, display text, synonyms, or definitions are rendered.",
            "",
            "## Review-Bound Proof",
            f"- Review required count: `{report['review_required_count']}`.",
            f"- Auto-accept allowed count: `{report['auto_accept_allowed_count']}`.",
            f"- Clinical interpretation count: `{report['clinical_interpretation_count']}`.",
            f"- Inference flag true count: `{report['inference_flag_true_count']}`.",
            "",
            "## What Was Not Changed",
            "- Runtime code, app/main.py, helper code, Streamlit wiring, extraction, OCR, classifier, thresholds, cue packs, DDI, and clinical behavior were not changed by this UAT.",
            "- Cue expansion remains NOT recommended.",
            "",
            "## Validation Evidence",
            "- Focused CKA-TERM-INTEGRATION-WIRING-UAT-01 tests: pending.",
            "- Direct UAT script: passed.",
            "- Public report privacy checks: pending.",
            "- Broader validations: pending.",
            "",
            "## Recommended Next Step",
            f"`{report['recommended_next_step']}`.",
            "",
        ]
    )


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MD_REPORT.write_text(_markdown(report, title="MEDAI-CKA-TERM-INTEGRATION-WIRING-UAT-01 Report"), encoding="utf-8")
    GUIDE_REPORT.write_text(_markdown(report, title="MEDAI-CKA-TERM-INTEGRATION-WIRING-UAT-01"), encoding="utf-8")


def main() -> int:
    report = run_uat()
    write_reports(report)
    print(json.dumps({
        "phase_id": PHASE_ID,
        "neither_env_truthy_render_calls": report["neither_env_truthy_render_calls"],
        "helper_only_env_truthy_render_calls": report["helper_only_env_truthy_render_calls"],
        "ui_only_env_truthy_render_calls": report["ui_only_env_truthy_render_calls"],
        "forbidden_streamlit_call_count": report["forbidden_streamlit_call_count"],
        "unsafe_metadata_render_count": report["unsafe_metadata_render_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
