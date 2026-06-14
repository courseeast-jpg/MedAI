"""Tests for the 17C-R2-R7 missing-schema-fields triage. Inspect artifacts + helpers;
never call a provider model."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.strict_json import missing_required_keys
import scripts.run_medai_ai_first_corpus_17c_r2_missing_schema_fields_triage_local_only_17c_r2_r7 as mod
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_missing_schema_fields_triage_local_only_17c_r2_r7"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_MISSING_SCHEMA_FIELDS_TRIAGE_LOCAL_ONLY_17C_R2_R7"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "missing_schema_fields_public.json",
                  "local_replay_validation_public.json", "prompt_contract_hardening_public.md",
                  "live_resume_entry_gate.md", "safety_boundary_public.md")
ALLOWED_CLASSES = {
    "top_level_required_fields_missing", "nested_required_fields_missing", "empty_required_arrays",
    "wrong_type_for_required_field", "null_where_object_required", "schema_version_or_contract_mismatch",
    "provider_omitted_empty_sections", "unknown_missing_expected_schema_fields",
}


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_required_docs_exist():
    for d in (
        "MEDAI_17C_R2_MISSING_SCHEMA_FIELDS_TRIAGE_LOCAL_ONLY_17C_R2_R7.md",
        "MEDAI_17C_R2_REQUIRED_SCHEMA_FIELDS_POLICY_R7.md",
        "MEDAI_17C_R2_LIVE_RESUME_ENTRY_GATE_AFTER_R7.md",
    ):
        assert (DOC_DIR / d).exists(), d


def test_no_provider_model_call_no_live_no_mkb():
    s = _summary()
    assert s["provider_model_call_made"] is False
    assert s["vertex_model_call_made"] is False
    assert s["gemini_call_made"] is False
    assert s["live_gate_set"] is False
    assert s["live_extraction_started"] is False
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    src = inspect.getsource(mod)
    assert "_default_http_post" not in src and "generate_content" not in src


def test_prior_live_counts_preserved():
    s = _summary()
    assert s["prior_request_count_loaded"] == 478
    assert s["prior_request_count_sent"] == 2
    assert s["prior_request_count_succeeded"] == 1
    assert s["prior_request_count_failed"] == 1
    assert s["prior_failure_stage"] == "schema"
    assert s["prior_failure_category"] == "missing_expected_schema_fields"


def test_failure_class_allowed():
    assert _summary()["schema_failure_class"] in ALLOWED_CLASSES


def test_missing_field_reporting_shape():
    s = _summary()
    assert isinstance(s["missing_required_field_names"], list)
    assert isinstance(s["missing_required_field_schema_paths"], list)
    assert s["missing_required_field_count"] == len(s["missing_required_field_names"])
    # When evidence exists, names should be reported; when absent, an empty list is honest.
    if s["failed_response_body_available_in_context"] and s["missing_required_field_count"] > 0:
        assert all(p.startswith("$.") for p in s["missing_required_field_schema_paths"])


def test_schema_not_weakened_no_inference():
    s = _summary()
    assert s["schema_weakened"] is False
    assert s["required_fields_made_optional"] is False
    assert s["clinical_values_inferred"] is False
    assert s["missing_required_values_synthesized"] is False


def test_required_fields_precheck_works():
    req = live.EXPECTED_TOP_LEVEL_FIELDS
    assert missing_required_keys({}, req) == list(req)
    full = {k: [] for k in req}
    assert missing_required_keys(full, req) == []
    assert missing_required_keys({"extracted_labs": []}, req) == [k for k in req if k != "extracted_labs"]


def test_validator_requires_all_core_keys():
    # Stricter than before: a single-key object must FAIL; full skeleton passes.
    ok_partial, reason = live._validate_schema('{"extracted_labs":[]}')
    assert ok_partial is False and reason == "missing_expected_schema_fields"
    full = {k: [] for k in live.EXPECTED_TOP_LEVEL_FIELDS}
    full["needs_review"] = False
    ok_full, _ = live._validate_schema(json.dumps(full))
    assert ok_full is True


def test_prompt_hardened_for_required_fields():
    s = _summary()
    assert s["prompt_hardened_for_required_fields"] is True
    assert s["schema_skeleton_instruction_added"] is True
    assert s["required_fields_precheck_added"] is True


def test_ready_flag_consistency():
    s = _summary()
    if s["ready_to_resume_17c_r2_live"]:
        assert s["prompt_hardened_for_required_fields"] is True
        assert s["required_fields_precheck_added"] is True
        assert s["sealed_batch_valid"] is True
        assert s["credential_preflight_passed"] is True


def test_no_raw_response_or_credentials_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        for m in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key",
                  '"tokenized_content"', '"response"'):
            assert m not in t, (name, m)


def test_reports_pass_privacy_check():
    for name in PUBLIC_REPORTS:
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.raw_phi_logged_in_public_reports is False, name
        assert r.secret_leaks == 0, name


def test_no_ssn_pattern_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
