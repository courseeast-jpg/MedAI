"""Tests for R18 stronger-model residual fallback. Public artifacts + pure helpers;
never call a provider, never open MKB, never mutate main/R17 checkpoints."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution import live_checkpoint as lc
import scripts.run_medai_r18_residual_schema_failures_stronger_model_fallback as mod
import scripts.run_medai_fast_same_day_flash_contract_recovery_r17 as r17

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_r18_residual_schema_failures_stronger_model_fallback"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "stronger_model_route_public.json",
                  "residual_recovery_result_public.json", "safety_boundary_public.md")
_WIN = re.compile(r"[A-Za-z]:\\")
_LONG_HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])")
_C2_RTF_IDS = {"doc_8ab6a6eacd51d900", "doc_8f7ae945d57cf840"}


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for n in PUBLIC_REPORTS:
        assert (REPORT_DIR / n).exists(), n


def test_corpus1_residual_excludes_content_packages():
    residual, content_before = mod._corpus1_residual()
    assert content_before == 141  # 118 original + 23 R17
    assert len(residual) == 337
    done = set(lc.load_completed()) | set(lc.load_completed(base=r17.C1_CKPT))
    sel_ids = {str(r.get("document_id") or "") for r in residual}
    assert sel_ids.isdisjoint(done)  # no completed / R17-recovered docs reprocessed


def test_corpus2_residual_only_recoverable():
    residual = mod._corpus2_residual()
    assert len(residual) == 2
    ids = {str(r.get("document_id") or "") for r in residual}
    assert ids.isdisjoint(_C2_RTF_IDS)
    assert ids.isdisjoint(set(lc.load_completed(base=r17.C2_LIVE_CKPT)))  # 21 not resent


def test_stronger_model_route_configured_and_gated():
    s = _summary()
    assert s["stronger_model_route_available"] is True
    rt = json.loads((REPORT_DIR / "stronger_model_route_public.json").read_text(encoding="utf-8"))
    assert rt["stronger_model"] == "gemini-2.5-pro"
    assert rt["grounding_or_search_used"] is False
    assert rt["reuses_r17_salvage_skeleton_minimal"] is True
    src = inspect.getsource(mod)
    # Pro path runs only behind the gate; route is the existing safety-gated adapter.
    assert 'if args.live and g["gate_passed"]:' in src
    assert "_run_corpus" in src and "r17." in src


def test_cost_estimate_within_remaining_budget():
    s = _summary()
    assert s["prior_same_day_spend_usd"] == 3.033334
    assert float(s["estimated_cost_before_live_usd"]) + s["prior_same_day_spend_usd"] <= 50.00


def test_no_provider_call_during_local_gate():
    s = _summary()
    if not (s["corpus1_live_started"] or s["corpus2_live_started"]):
        assert s["provider_model_call_made"] is False
        assert s["gemini_call_made"] is False


def test_fast_fail_threshold():
    src = inspect.getsource(mod)
    assert "FAST_FAIL_THRESHOLD_DOCS = 20" in src


def test_no_mkb_autoaccept_medical_and_checkpoint_isolation():
    s = _summary()
    for k in ("mkb_db_opened", "active_mkb_write", "auto_accept_enabled", "medical_decision_made",
              "future_mkb_import_started", "corpus1_main_checkpoint_mutated", "r17_checkpoint_mutated"):
        assert s[k] is False, k
    assert "corpus1_r18_checkpoint" in str(mod.R18_C1_CKPT)
    assert "corpus2_r18_checkpoint" in str(mod.R18_C2_CKPT)


def test_no_grounding_search():
    s = _summary()
    assert s["grounding_or_search_used"] is False


def test_no_private_committed_flags():
    s = _summary()
    for k in ("private_artifacts_committed", "raw_ai_response_committed", "tokenized_payloads_committed",
              "token_maps_committed", "pi_values_committed", "credentials_or_tokens_committed",
              "corpus2_oversized_docs_sent", "corpus1_completed_docs_reprocessed",
              "corpus2_completed_docs_reprocessed"):
        assert s[k] is False, k


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
