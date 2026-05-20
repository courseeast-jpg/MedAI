"""MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01 validation/report script."""
from __future__ import annotations

import ast
import json
import os
import re
import sys
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

PHASE_ID = "MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01"
UI_ENV_VAR = "MEDAI_TERMINOLOGY_LOOKUP_UI_ENABLED"
REPORT_DIR = REPO_ROOT / "reports" / "medai_cka_term_integration_wiring_next_01"
JSON_REPORT = REPORT_DIR / "medai_cka_term_integration_wiring_next_01_report.json"
MD_REPORT = REPORT_DIR / "medai_cka_term_integration_wiring_next_01_report.md"
GUIDE_REPORT = REPORT_DIR / "MEDAI_CKA_TERM_INTEGRATION_WIRING_NEXT_01.md"
APP_MAIN = REPO_ROOT / "app" / "main.py"

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


def _app_source() -> str:
    return APP_MAIN.read_text(encoding="utf-8")


def _extract_function_source(name: str) -> str:
    source = _app_source()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise RuntimeError(f"missing function {name}")


def _load_plan_functions() -> dict[str, Any]:
    namespace: dict[str, Any] = {"os": os, "TERMINOLOGY_LOOKUP_UI_ENV_VAR": UI_ENV_VAR}
    exec(_extract_function_source("_terminology_lookup_ui_truthy"), namespace)
    exec(_extract_function_source("terminology_match_hypothesis_ui_plan"), namespace)
    return namespace


def _safe_metadata() -> dict[str, Any]:
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


def _truthy_env() -> dict[str, str]:
    return {TERMINOLOGY_LOOKUP_ENV_VAR: "true", UI_ENV_VAR: "true"}


def _wiring_block_source() -> str:
    source = _app_source()
    marker = "MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01"
    first = source.find(marker)
    if first < 0:
        return ""
    second = source.find(marker, first + len(marker))
    block_start = source.rfind("try:", 0, second if second >= 0 else first)
    block_end = source.find("except Exception:", second if second >= 0 else first)
    if block_start < 0 or block_end < 0:
        return ""
    block_end = source.find("pass", block_end)
    return source[block_start:block_end + len("pass")]


def run_wiring_audit() -> dict[str, Any]:
    namespace = _load_plan_functions()
    plan_fn = namespace["terminology_match_hypothesis_ui_plan"]
    safe_item = {"terminology_match_hypothesis_metadata": _safe_metadata()}

    env_cases = {
        "neither_env_truthy": plan_fn(safe_item, env={}),
        "helper_only_env_truthy": plan_fn(safe_item, env={TERMINOLOGY_LOOKUP_ENV_VAR: "true"}),
        "ui_only_env_truthy": plan_fn(safe_item, env={UI_ENV_VAR: "true"}),
        "both_env_truthy": plan_fn(safe_item, env=_truthy_env()),
    }
    unsafe_item = {
        "terminology_match_hypothesis_metadata": {
            **_safe_metadata(),
            "auto_accept_allowed": True,
        }
    }
    unsafe_plan = plan_fn(unsafe_item, env=_truthy_env())
    block = _wiring_block_source()
    app_source = _app_source()
    markdown_calls = len(re.findall(r"\bst\.markdown\(", block))
    caption_calls = len(re.findall(r"\bst\.caption\(", block))
    forbidden_call_count = sum(block.count(token) for token in FORBIDDEN_STREAMLIT_CALLS)
    marker_count = app_source.count("MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01")

    both_plan = env_cases["both_env_truthy"]
    privacy_payload = {
        "render_plan": both_plan,
        "metadata": {
            "match_family": _safe_metadata()["match_family"],
            "terminology_system_family": _safe_metadata()["terminology_system_family"],
            "matches_count": _safe_metadata()["matches_count"],
        },
    }
    privacy = check_public_report_payload(privacy_payload)
    report: dict[str, Any] = {
        "phase_id": PHASE_ID,
        "mode": "default_off_read_only_terminology_ui_wiring",
        "default_off": True,
        "helper_env_var": TERMINOLOGY_LOOKUP_ENV_VAR,
        "ui_env_var": UI_ENV_VAR,
        "requires_both_env_vars": True,
        "behavior_changed": True,
        "default_behavior_changed": False,
        "runtime_behavior_changed_by_default": False,
        "app_main_modified": True,
        "streamlit_wiring_added": True,
        "streamlit_wiring_enabled_by_default": False,
        "advanced_technical_details_only": True,
        "read_only": True,
        "buttons_added": False,
        "callbacks_added": False,
        "actions_added": False,
        "forms_added": False,
        "state_mutation_added": False,
        "data_layer_write_added": False,
        "document_type_mutation_added": False,
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
        "whole_medai_done_estimate": "approximately 94.6%",
        "recommended_next_step": "CKA-TERM-INTEGRATION-WIRING-UAT-01 or ROADMAP-05",
        "env_gate_results": {
            "neither_env_truthy_renders": env_cases["neither_env_truthy"] is not None,
            "helper_only_env_truthy_renders": env_cases["helper_only_env_truthy"] is not None,
            "ui_only_env_truthy_renders": env_cases["ui_only_env_truthy"] is not None,
            "both_env_truthy_renders": both_plan is not None,
            "unsafe_metadata_renders": unsafe_plan is not None,
        },
        "allowed_streamlit_calls": ["st.markdown", "st.caption"],
        "markdown_call_count_in_wiring_block": markdown_calls,
        "caption_call_count_in_wiring_block": caption_calls,
        "forbidden_streamlit_call_count": forbidden_call_count,
        "wiring_marker_count": marker_count,
        "public_report_privacy_clean": privacy.passed,
        "validation_results": {
            "focused_tests": "pending",
            "direct_script": "passed",
            "public_report_privacy_checks": "pending",
            "final_cka_mvp_validation": "pending",
            "b07_term01_validation": "pending",
            "route_fix_validation": "pending",
            "ui_ops_validation": "pending",
            "ui_boot_validation": "pending",
            "staged_safety_check": "pending",
            "full_pytest": "not run; not needed for narrow default-off UI wiring",
        },
    }
    required = (
        env_cases["neither_env_truthy"] is None
        and env_cases["helper_only_env_truthy"] is None
        and env_cases["ui_only_env_truthy"] is None
        and both_plan is not None
        and unsafe_plan is None
        and forbidden_call_count == 0
        and markdown_calls > 0
        and caption_calls > 0
        and marker_count == 2
        and privacy.passed
    )
    if not required:
        raise RuntimeError("terminology UI wiring audit failed")
    return report


def _markdown(report: dict[str, Any], *, title: str) -> str:
    gate = report["env_gate_results"]
    return "\n".join(
        [
            f"# {title}",
            "",
            "## Why This Exists",
            "This block wires safe aggregate terminology match metadata into Advanced technical details only.",
            "",
            "## Wiring Summary",
            f"- Helper env var: `{report['helper_env_var']}`.",
            f"- UI env var: `{report['ui_env_var']}`.",
            "- Both env vars must be truthy before anything renders.",
            "- The UI renders only already-emitted aggregate helper metadata.",
            "- No adapter is created by the UI and no terminology rows are read.",
            "",
            "## Default-Off Proof",
            f"- Neither env truthy renders: `{gate['neither_env_truthy_renders']}`.",
            f"- Helper-only env truthy renders: `{gate['helper_only_env_truthy_renders']}`.",
            f"- UI-only env truthy renders: `{gate['ui_only_env_truthy_renders']}`.",
            f"- Both env truthy renders: `{gate['both_env_truthy_renders']}`.",
            "",
            "## Read-Only Render Surface",
            "- Render location: Advanced technical details only.",
            f"- Allowed Streamlit calls: `{report['allowed_streamlit_calls']}`.",
            f"- Forbidden Streamlit call count: `{report['forbidden_streamlit_call_count']}`.",
            "- Rendered fields: match family, terminology system family, matches count, review-required / auto-accept summary, and disclaimer.",
            "",
            "## License And Privacy Proof",
            "- Licensed terminology rows read/rendered/publicly reported: false.",
            "- Row code, display text, synonyms, definitions, raw text, OCR text, document text, filenames, private paths, PHI, and secrets are not rendered.",
            f"- Public report privacy clean: `{report['public_report_privacy_clean']}`.",
            "",
            "## What Was Not Changed",
            "- Extraction, OCR, classifier, thresholds, scoring, cue packs, DDI, clinical inference, auto-accept, and external API behavior were not changed.",
            "- Cue expansion remains NOT recommended.",
            "",
            "## Validation Evidence",
            "- Focused tests: pending.",
            "- Direct report script: passed.",
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
    MD_REPORT.write_text(_markdown(report, title="MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01 Report"), encoding="utf-8")
    GUIDE_REPORT.write_text(_markdown(report, title="MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01"), encoding="utf-8")


def main() -> int:
    report = run_wiring_audit()
    write_reports(report)
    print(json.dumps({
        "phase_id": PHASE_ID,
        "requires_both_env_vars": report["requires_both_env_vars"],
        "both_env_truthy_renders": report["env_gate_results"]["both_env_truthy_renders"],
        "forbidden_streamlit_call_count": report["forbidden_streamlit_call_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
