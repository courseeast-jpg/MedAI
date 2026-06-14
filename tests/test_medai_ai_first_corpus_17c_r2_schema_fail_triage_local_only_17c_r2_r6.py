"""Tests for the 17C-R2-R6 schema-fail triage. Inspect artifacts + the normalizer;
never call a provider model."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.strict_json import normalize_one_json_object
import scripts.run_medai_ai_first_corpus_17c_r2_schema_fail_triage_local_only_17c_r2_r6 as mod
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_schema_fail_triage_local_only_17c_r2_r6"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_SCHEMA_FAIL_TRIAGE_LOCAL_ONLY_17C_R2_R6"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "schema_failure_classification_public.json",
                  "local_replay_validation_public.json", "prompt_and_parser_hardening_public.md",
                  "live_resume_entry_gate.md", "safety_boundary_public.md")
ALLOWED_CLASSES = {
    "empty_response", "markdown_fenced_json", "prose_wrapped_json", "multiple_json_objects",
    "truncated_json", "invalid_json_escape", "safety_refusal_or_policy_text",
    "schema_shape_mismatch", "unexpected_non_json_text", "unknown_not_strict_json",
}


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_required_docs_exist():
    for d in (
        "MEDAI_17C_R2_SCHEMA_FAIL_TRIAGE_LOCAL_ONLY_17C_R2_R6.md",
        "MEDAI_17C_R2_STRICT_JSON_RESPONSE_POLICY_R6.md",
        "MEDAI_17C_R2_LIVE_RESUME_ENTRY_GATE_AFTER_R6.md",
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
    assert s["prior_failure_category"] == "not_strict_json"


def test_failure_class_allowed():
    s = _summary()
    assert s["schema_failure_class"] in ALLOWED_CLASSES


def test_schema_not_weakened_no_partial_no_inference():
    s = _summary()
    assert s["schema_weakened"] is False
    assert s["partial_json_accepted"] is False
    assert s["missing_fields_inferred"] is False


def test_normalizer_rejects_prose_multiple_truncated_recovers_fence():
    good = json.dumps({"extracted_labs": [], "needs_review": True})
    assert normalize_one_json_object(good)[1] == "ok"
    assert normalize_one_json_object("```json\n" + good + "\n```")[1] == "ok"
    assert normalize_one_json_object("Explanation:\n" + good)[0] is None        # prose
    assert normalize_one_json_object(good + "\n" + good)[0] is None             # multiple
    assert normalize_one_json_object('{"extracted_labs":[')[0] is None         # truncated
    assert normalize_one_json_object("")[0] is None                            # empty


def test_stale_cap_test_updated_and_caps():
    s = _summary()
    assert s["stale_cap_test_updated"] is True
    assert s["authorized_hard_cost_cap_total_usd"] == 0.40
    assert s["hard_cost_cap_per_chunk_usd"] == 0.05
    assert float(live.CAP_TOTAL) == 0.40
    assert float(live.CAP_PER_CHUNK) == 0.05


def test_ready_flag_consistency():
    s = _summary()
    if s["ready_to_resume_17c_r2_live"]:
        assert s["safe_normalizer_added"] is True
        assert s["prompt_hardened_for_strict_json"] is True
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
