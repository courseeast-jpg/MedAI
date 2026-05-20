"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-21."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_21_streamlit_fixture_audit import (
    ALLOWED_STREAMLIT_CALLS,
    APP_MAIN,
    DIAG_19_MARKER,
    PHASE_ID,
    build_report,
    env_gate_fixture_audit,
    extract_diag_19_block,
    simulate_diag_19_streamlit_calls,
    static_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports/medai_doc_type_unknown_diag_21_streamlit_fixture_audit"
RUNTIME_FILES = [
    "app/main.py",
    "clinical_knowledge/document_type/pdf_text_layout_quality_impl.py",
    "clinical_knowledge/document_type/pdf_text_layout_quality_ui.py",
]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT).decode().strip()


def test_diag_21_modifies_no_runtime_files():
    changed = set(_git("diff", "--name-only", "HEAD").splitlines())
    staged = set(_git("diff", "--cached", "--name-only").splitlines())
    touched = changed | staged
    assert not (touched & set(RUNTIME_FILES))


def test_diag_19_block_exists_exactly_once():
    body = APP_MAIN.read_text(encoding="utf-8")
    assert body.count(DIAG_19_MARKER) == 1
    assert extract_diag_19_block(body)


def test_diag_19_block_remains_inside_advanced_technical_details():
    assert static_audit()["diag_19_inside_advanced_technical_details"] is True


def test_both_env_vars_are_required():
    audit = env_gate_fixture_audit()
    assert audit["neither_env_truthy_streamlit_calls"] == 0
    assert audit["metadata_only_env_truthy_streamlit_calls"] == 0
    assert audit["ui_only_env_truthy_streamlit_calls"] == 0
    assert audit["both_env_truthy_streamlit_calls"] > 0


def test_neither_env_truthy_has_zero_streamlit_calls():
    assert simulate_diag_19_streamlit_calls({})["call_count"] == 0


def test_metadata_only_env_truthy_has_zero_streamlit_calls():
    env = {"MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED": "1"}
    assert simulate_diag_19_streamlit_calls(env)["call_count"] == 0


def test_ui_only_env_truthy_has_zero_streamlit_calls():
    env = {"MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED": "1"}
    assert simulate_diag_19_streamlit_calls(env)["call_count"] == 0


def test_both_env_truthy_uses_only_markdown_and_caption():
    env = {
        "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED": "1",
        "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED": "1",
    }
    result = simulate_diag_19_streamlit_calls(env)
    assert result["call_count"] > 0
    assert result["allowed_calls_only"] is True
    assert set(result["call_names"]).issubset(set(ALLOWED_STREAMLIT_CALLS))


def test_no_forbidden_streamlit_calls_appear():
    audit = static_audit()
    assert audit["forbidden_streamlit_calls_static"] == []
    assert audit["allowed_streamlit_calls_only_static"] is True


def test_no_button_callback_action_form_state_or_mutation_surface():
    audit = static_audit()
    assert audit["no_button_callback_action_form_state_mutation_tokens"] is True
    report = build_report()
    for key in [
        "buttons_added",
        "callbacks_added",
        "actions_added",
        "forms_added",
        "state_mutation_added",
        "data_layer_write_added",
        "document_type_mutation_added",
    ]:
        assert report[key] is False, key


def test_no_raw_text_filenames_or_private_paths_rendered():
    report = build_report()
    assert report["raw_text_rendered"] is False
    assert report["raw_filenames_rendered"] is False
    assert report["private_paths_rendered"] is False
    assert report["fixture_audit"]["raw_or_private_token_rendered"] is False


def test_no_auto_accept_or_clinical_inference_or_abbreviation_expansion():
    report = build_report()
    assert report["accepted_count"] == 0
    assert report["auto_accept_allowed_count"] == 0
    assert report["clinical_interpretation_performed"] is False
    assert report["clinical_value_parsing_performed"] is False
    assert report["diagnosis_inference_performed"] is False
    assert report["medication_inference_performed"] is False
    assert report["ddi_inference_performed"] is False
    assert report["treatment_inference_performed"] is False
    assert report["abbreviation_expansion_performed"] is False


def test_parking_tags_not_touched_and_diag_20_untagged():
    report = build_report()
    assert report["park_20_tags_touched"] is False
    assert report["park_21_tags_touched"] is False
    assert report["park_22_tags_touched"] is False
    assert report["park_23_tags_touched"] is False
    assert report["park_24_tags_touched"] is False
    assert report["diag_20_commit_tagged"] is False
    assert report["tag_verification"]["diag_20_tags"] == []


def test_cue_expansion_remains_not_recommended():
    report = build_report()
    assert report["cue_expansion_recommended"] is False
    assert report["cue_expansion_performed"] is False


def test_build_report_required_fields():
    report = build_report()
    assert report["phase_id"] == PHASE_ID
    assert report["mode"] == "streamlit_fixture_audit"
    assert report["reports_only"] is True
    assert report["default_off"] is True
    assert report["app_main_modified"] is False
    assert report["metadata_env_var"] == "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED"
    assert report["ui_env_var"] == "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED"
    assert report["requires_both_env_vars"] is True
    assert report["external_api_used"] is False
    assert report["external_api_used_count"] == 0
    assert report["all_records_review_bound"] is True


def test_report_files_exist_after_script_run():
    for name in [
        "MEDAI_DOC_TYPE_UNKNOWN_DIAG_21_STREAMLIT_FIXTURE_AUDIT.md",
        "medai_doc_type_unknown_diag_21_streamlit_fixture_audit_report.json",
        "medai_doc_type_unknown_diag_21_streamlit_fixture_audit_report.md",
    ]:
        assert (REPORT_DIR / name).exists()


def test_report_json_privacy_clean():
    path = REPORT_DIR / "medai_doc_type_unknown_diag_21_streamlit_fixture_audit_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = check_public_report_payload(payload)
    assert result.passed, result.leak_examples_redacted


def test_report_markdown_privacy_clean():
    for name in [
        "MEDAI_DOC_TYPE_UNKNOWN_DIAG_21_STREAMLIT_FIXTURE_AUDIT.md",
        "medai_doc_type_unknown_diag_21_streamlit_fixture_audit_report.md",
    ]:
        text = (REPORT_DIR / name).read_text(encoding="utf-8")
        result = check_public_report_payload({"public_report_text": text})
        assert result.passed, result.leak_examples_redacted
