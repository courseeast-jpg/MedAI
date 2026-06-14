"""Tests for Corpus 2 / P2 privacy-unblock + gated live extraction. Public artifacts and
pure helpers only; never call a provider, never open MKB, never read/print PI or candidates."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
import scripts.run_medai_corpus2_p2_privacy_unblock_and_live_extraction_01 as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_corpus2_p2_privacy_unblock_and_live_extraction_01"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "privacy_unblock_public.json",
                  "live_entry_gate_public.md", "live_run_public_report.md", "failed_docs_public.json",
                  "safety_boundary_public.md")
_WIN = re.compile(r"[A-Za-z]:\\")
_LONG_HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])")


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for n in PUBLIC_REPORTS:
        assert (REPORT_DIR / n).exists(), n


def test_high_confidence_detector_and_leak_scan_pure():
    assert mod.base_p2.high_confidence_uncovered_count("email a@b.com phone 555-123-4567") > 0
    # Pure leak scan: a provider name after a cue is detected before tokenization.
    high, pf, sd = mod._raw_leak_counts(["Referring Physician: John Smith on March 3, 2020"])
    assert pf >= 1 and sd >= 1
    # After tokenization the same spans are gone.
    high2, pf2, sd2 = mod._raw_leak_counts(["Referring Physician: [PROVIDER_1] on [DATE_1]"])
    assert pf2 == 0 and sd2 == 0


def test_unblock_and_gate_summary():
    s = _summary()
    assert s["supplemental_private_vault_created"] is True
    assert s["high_confidence_uncovered_pi_count_after"] == 0
    assert s["provider_facility_raw_leak_count_after"] == 0
    assert s["spelled_date_raw_leak_count_after"] == 0
    # 25 loaded docs reconcile to sendable + content_too_large corrections.
    assert s["tokenized_request_count"] + s["content_too_large_excluded_count"] == 25
    assert s["docs_loaded"] == 25


def test_local_gate_made_no_provider_call():
    s = _summary()
    if not s["live_run_started"]:
        assert s["provider_model_call_made"] is False
        assert s["gemini_call_made"] is False


def test_live_blocks_when_raw_pi_remains_source_wiring():
    src = inspect.getsource(mod)
    # Gate requires zero high/provider-facility/spelled-date raw leaks, and live runs only behind it.
    assert "high == 0" in src and "pf == 0" in src and "sd == 0" in src
    assert 'if args.live and g["gate_passed"]:' in src
    s = _summary()
    # When live did run, gate must have passed.
    if s["live_run_started"]:
        assert s["live_entry_gate_passed"] is True


def test_privacy_unblock_report_counts_only():
    pu = json.loads((REPORT_DIR / "privacy_unblock_public.json").read_text(encoding="utf-8"))
    assert pu["no_candidate_values_in_this_report"] is True
    assert isinstance(pu["supplemental_private_value_count"], int)
    assert pu["high_confidence_uncovered_pi_count_after"] == 0


def test_supplemental_vault_and_payloads_private_only():
    for pat in ("person2_pi_vault_supplemental_private.csv", "outbound_requests_private.jsonl",
                "token_maps_private.jsonl", "person2_pi_vault_private.csv"):
        assert not list(REPO_ROOT.rglob(pat)), pat
    s = _summary()
    assert s["tokenized_payloads_committed"] is False
    assert s["token_maps_committed"] is False
    assert s["pi_values_committed"] is False


def test_no_mkb_autoaccept_medical():
    s = _summary()
    for k in ("mkb_db_opened", "active_mkb_write", "auto_accept_enabled", "medical_decision_made",
              "future_mkb_import_started"):
        assert s[k] is False, k


def test_no_private_artifacts_committed_flags():
    s = _summary()
    for k in ("private_artifacts_committed", "raw_ai_response_committed",
              "tokenized_payloads_committed", "token_maps_committed", "pi_values_committed",
              "credentials_or_tokens_committed"):
        assert s[k] is False, k


def test_shared_budget_and_model():
    s = _summary()
    assert s["shared_budget_cap_usd"] == 50.00
    assert s["target_model"] == "gemini-2.5-flash-lite"
    assert float(s["estimated_cost_before_live_usd"]) <= 50.00


def test_corpus1_not_touched():
    s = _summary()
    assert s["corpus1_touched"] is False
    src = inspect.getsource(mod)
    assert "run_autonomous_recovery" not in src
    assert "corpus2_p2" in str(REPORT_DIR)


def test_no_private_paths_or_secrets_in_reports():
    for n in PUBLIC_REPORTS:
        t = (REPORT_DIR / n).read_text(encoding="utf-8")
        assert "MedAI_Private" not in t, n
        assert not _WIN.search(t), n
        assert not _LONG_HEX.search(t), (n, "long-hex")
        for m in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", "Bearer "):
            assert m not in t, (n, m)


def test_public_reports_pass_privacy_check():
    s = _summary()
    assert s["public_report_phi_leak_count"] == 0
    for n in PUBLIC_REPORTS:
        r = check_public_report_payload((REPORT_DIR / n).read_text(encoding="utf-8"))
        assert r.passed, (n, getattr(r, "leak_examples_redacted", None))


def test_no_ssn_pattern():
    for n in PUBLIC_REPORTS:
        t = (REPORT_DIR / n).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
