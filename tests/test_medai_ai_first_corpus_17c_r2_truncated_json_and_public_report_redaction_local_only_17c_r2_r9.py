"""Tests for 17C-R2-R9 truncated-JSON triage + public-report redaction. Inspect
artifacts and exercise the redaction helper; never call a provider model."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.public_report_redaction import redact_report_file
import scripts.run_medai_ai_first_corpus_17c_r2_truncated_json_and_public_report_redaction_local_only_17c_r2_r9 as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_truncated_json_and_public_report_redaction_local_only_17c_r2_r9"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_TRUNCATED_JSON_AND_PUBLIC_REPORT_REDACTION_LOCAL_ONLY_17C_R2_R9"
LIVE_BATCH_REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_live_batch_vertex_17c_r2"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "truncated_json_classification_public.json",
                  "public_report_redaction_public.json", "failed_evidence_preservation_public.json",
                  "next_live_strategy_public.md", "safety_boundary_public.md")
ALLOWED_CLASSES = {
    "provider_empty_response", "provider_truncated_by_max_tokens", "provider_truncated_mid_json",
    "invalid_escaping", "invalid_control_character", "non_json_safety_or_refusal",
    "response_mime_json_but_invalid_body", "schema_validator_parse_bug",
    "unknown_truncated_or_invalid_json",
}
_WIN_PATH = re.compile(r"[A-Za-z]:\\")


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_required_docs_exist():
    for d in (
        "MEDAI_17C_R2_TRUNCATED_JSON_TRIAGE_LOCAL_ONLY_17C_R2_R9.md",
        "MEDAI_17C_R2_PUBLIC_REPORT_REDACTION_POLICY_R9.md",
        "MEDAI_17C_R2_NEXT_LIVE_STRATEGY_AFTER_R9.md",
    ):
        assert (DOC_DIR / d).exists(), d


def test_no_provider_no_live_no_mkb():
    s = _summary()
    for k in ("provider_model_call_made", "vertex_model_call_made", "gemini_call_made",
              "claude_call_made", "openai_call_made", "billing_api_call_made",
              "live_gate_set", "live_extraction_started", "mkb_db_opened", "active_mkb_write",
              "auto_accept_enabled", "medical_decision_made", "production_queue_mutated"):
        assert s[k] is False, k
    src = inspect.getsource(mod)
    assert "_default_http_post" not in src
    assert "generate_content" not in src


def test_prior_live_counts_preserved():
    s = _summary()
    assert s["prior_live_execution_result"] == "SCHEMA_FAIL"
    assert s["prior_request_count_loaded"] == 478
    assert s["prior_request_count_sent"] == 1
    assert s["prior_request_count_succeeded"] == 0
    assert s["prior_request_count_failed"] == 1
    assert s["prior_failure_stage"] == "schema"
    assert s["prior_failure_category"] == "truncated_or_invalid_json"


def test_failure_category_and_class():
    s = _summary()
    assert s["prior_failure_category"] == "truncated_or_invalid_json"
    assert s["truncated_json_failure_class"] in ALLOWED_CLASSES


def test_no_raw_ai_response_in_repo_reports():
    # Structural metadata only; never a raw body. The error head is a class label, not content.
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert '"response"' not in t
        assert '"candidates"' not in t
        assert '"tokenized_content"' not in t


def test_no_private_windows_path_in_r9_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert "MedAI_Private" not in t
        assert not _WIN_PATH.search(t), name


def test_no_secret_like_strings_in_r9_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        for m in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", "Bearer "):
            assert m not in t, (name, m)


def test_r9_reports_pass_privacy_check():
    assert _summary()["public_report_phi_leak_count"] == 0
    for name in PUBLIC_REPORTS:
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.passed, (name, getattr(r, "leak_examples_redacted", None))


def test_live_batch_reports_no_private_path_or_secret_after_redaction():
    # The committed 17C-R2 live-batch reports must carry no raw Windows path / secret.
    for f in sorted(LIVE_BATCH_REPORT_DIR.iterdir()):
        if f.suffix not in (".json", ".md", ".csv"):
            continue
        t = f.read_text(encoding="utf-8")
        assert "MedAI_Private" not in t, f.name
        assert not _WIN_PATH.search(t), f.name
        r = check_public_report_payload(t)
        assert r.passed, (f.name, r.private_filename_path_leaks, r.secret_leaks)


def test_redaction_summary_fields():
    s = _summary()
    assert s["public_report_redaction_applied"] is True
    assert s["private_filename_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
    assert s["privacy_checker_passed_after"] is True
    assert s["failed_evidence_public_path_redacted"] is True


def test_redaction_helper_behaviour():
    # The helper turns a private path / secret into safe labels and leaves clean text alone.
    leaky = json.dumps({"p": r"C:\Users\S1\AppData\Local\MedAI_Private\x\y.jsonl",
                        "sha": "a" * 64, "ok": "no_leak_here"})
    red, np, ns = redact_report_file(leaky, is_json=True)
    obj = json.loads(red)
    assert np == 1 and ns == 1
    assert "MedAI_Private" not in red and "C:\\" not in red
    assert obj["ok"] == "no_leak_here"
    assert check_public_report_payload(red).passed


def test_no_private_response_body_committed_flags():
    s = _summary()
    assert s["raw_failed_response_committed"] is False
    assert s["raw_failed_response_printed"] is False
    for k in ("private_responses_committed", "parsed_private_responses_committed",
              "tokenized_payloads_written_to_repo", "raw_ocr_written_to_repo",
              "token_maps_written_to_repo", "private_identifier_values_written_to_repo",
              "credential_or_token_written_to_repo"):
        assert s[k] is False, k


def test_retry_gate():
    s = _summary()
    assert s["ready_for_another_live_retry"] is False
    assert s["requires_runner_strategy_change_before_retry"] is True
    assert isinstance(s["recommended_next_live_strategy"], str) and s["recommended_next_live_strategy"]


def test_no_ssn_pattern_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
