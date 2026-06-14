"""Tests for Corpus 2 / P2 vault-coverage review + gated live extraction. Inspect PUBLIC
artifacts and pure helpers only; never call a provider, never open MKB, never read/print PI."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
import scripts.run_medai_ai_first_corpus2_p2_vault_coverage_review_and_live_extraction_01 as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus2_p2_vault_coverage_review_and_live_extraction_01"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "vault_coverage_public.json",
                  "live_entry_gate_public.md", "live_run_public_report.md", "failed_docs_public.json",
                  "safety_boundary_public.md")
_WIN_PATH = re.compile(r"[A-Za-z]:\\")
_LONG_HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])")


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_high_confidence_detector_flags_real_structured_pii():
    # Synthetic (non-real) structured PII must be detected; clean tokenized text must not.
    assert mod.high_confidence_uncovered_count("contact a@b.com or 555-123-4567") > 0
    assert mod.high_confidence_uncovered_count("labs [PERSON_1] value 12 within range") == 0


def test_vault_coverage_report_counts_only_no_pi():
    cov = json.loads((REPORT_DIR / "vault_coverage_public.json").read_text(encoding="utf-8"))
    assert cov["no_pi_values_in_this_report"] is True
    # Every reported category exposes only category/count/band/action — numbers, not values.
    for row in cov["high_confidence_categories"] + cov["low_confidence_categories"]:
        assert set(row.keys()) <= {"category", "count", "confidence_band", "action"}
        assert isinstance(row["count"], int)


def test_local_gate_made_no_provider_call():
    s = _summary()
    assert s["provider_model_call_made"] is False
    assert s["gemini_call_made"] is False
    assert s["live_run_started"] is False


def test_live_blocks_when_high_confidence_pi_or_gate_fails():
    s = _summary()
    # If high-confidence uncovered PI remains, the gate must not pass and live must not start.
    if s["high_confidence_uncovered_pi_count"] > 0:
        assert s["live_entry_gate_passed"] is False
        assert s["live_run_started"] is False
    # Source wiring: live runs only behind the gate, and the gate requires zero high-confidence PI.
    src = inspect.getsource(mod)
    assert "high_uncovered == 0" in src
    assert 'if g["gate_passed"]:' in src


def test_gate_blocked_outcome_consistent():
    s = _summary()
    if not s["live_entry_gate_passed"]:
        assert s["live_run_started"] is False
        assert s["provider_model_call_made"] is False
        assert s["run_result"] in ("BLOCKED", "LIVE_FAIL")


def test_tokenized_payloads_and_token_maps_private_only():
    s = _summary()
    assert s["tokenized_payloads_committed"] is False
    assert s["token_maps_committed"] is False
    # No private artifact files exist anywhere in the repo tree.
    for pat in ("outbound_requests_private.jsonl", "token_maps_private.jsonl",
                "person2_pi_vault_private.csv"):
        assert not list(REPO_ROOT.rglob(pat)), pat


def test_no_mkb_no_autoaccept_no_medical_decision():
    s = _summary()
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["future_mkb_import_started"] is False


def test_no_private_artifacts_committed_flags():
    s = _summary()
    for k in ("private_artifacts_committed", "raw_ai_response_committed",
              "tokenized_payloads_committed", "token_maps_committed", "pi_values_committed",
              "credentials_or_tokens_committed"):
        assert s[k] is False, k


def test_required_caps_and_counts():
    s = _summary()
    assert s["authorized_total_cap_usd"] == 2.00
    assert s["hard_cost_cap_per_chunk_usd"] == 0.05
    assert s["tokenized_request_count"] == 25
    assert s["target_model"] == "gemini-2.5-flash-lite"
    assert s["docs_loaded"] == 25


def test_corpus1_not_referenced_or_modified():
    s = _summary()
    assert s["corpus1_touched"] is False
    src = inspect.getsource(mod)
    # This block must not invoke the Corpus 1 autonomous engine or write Corpus 1 reports.
    assert "run_autonomous_recovery" not in src
    assert "autonomous_recovery_full_corpus" not in src
    assert "medai_ai_first_corpus_478_live_batch_vertex_17c_r2/" not in src
    assert "corpus2_p2" in str(REPORT_DIR)


def test_no_private_paths_or_secrets_in_public_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert "MedAI_Private" not in t, name
        assert not _WIN_PATH.search(t), name
        assert not _LONG_HEX.search(t), (name, "long-hex/secret-like token")
        for m in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", "Bearer "):
            assert m not in t, (name, m)


def test_public_reports_pass_privacy_check():
    s = _summary()
    assert s["public_report_phi_leak_count"] == 0
    assert s["private_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
    for name in PUBLIC_REPORTS:
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.passed, (name, getattr(r, "leak_examples_redacted", None))


def test_no_ssn_pattern_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
