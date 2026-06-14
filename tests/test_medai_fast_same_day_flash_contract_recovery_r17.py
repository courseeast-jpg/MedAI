"""Tests for R17 fast same-day flash contract recovery. Public artifacts + pure helpers;
never call a provider, never open MKB, never mutate main/R16 checkpoints."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution import flash_contract as fc
from execution.strict_json import missing_required_keys
import scripts.run_medai_fast_same_day_flash_contract_recovery_r17 as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_fast_same_day_flash_contract_recovery_r17"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "flash_contract_patch_public.json",
                  "recovery_result_public.json", "safety_boundary_public.md")
_WIN = re.compile(r"[A-Za-z]:\\")
_LONG_HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])")
_C2_RTF_IDS = {"doc_8ab6a6eacd51d900", "doc_8f7ae945d57cf840"}  # the 2 content_too_large containers


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for n in PUBLIC_REPORTS:
        assert (REPORT_DIR / n).exists(), n


def test_corpus1_selection_excludes_completed():
    sel, completed_count = mod._corpus1_selection()
    assert completed_count == 118
    assert len(sel) == 360
    completed = set()
    from execution import live_checkpoint as lc
    completed = set(lc.load_completed())
    sel_ids = {str(r.get("document_id") or "") for r in sel}
    assert sel_ids.isdisjoint(completed)


def test_corpus2_selection_only_recoverable():
    sel, completed, total = mod._corpus2_selection()
    assert completed == 21
    assert len(sel) == 2
    sel_ids = {str(r.get("document_id") or "") for r in sel}
    # 21 completed are not resent; the 2 RTF/signal containers are not in the outbound at all.
    assert sel_ids.isdisjoint(_C2_RTF_IDS)


def test_flash_json_salvage_handles_fenced_and_truncated():
    obj, _ = fc.salvage_flash_json('```json\n{"section":"x","items":[]}\n```')
    assert obj == {"section": "x", "items": []}
    obj2, reason2 = fc.salvage_flash_json('{"section":"labs","items":[{"a":1}')
    assert obj2 is not None and reason2 == "closed_truncated_tail"
    assert fc.salvage_flash_json("")[0] is None


def test_section_name_normalization_variants():
    s = fc.ALLOWED_SECTIONS[2]
    assert fc.normalize_section_name(s.replace("_", " ").upper()) == s
    assert fc.normalize_section_name(s) == s
    assert fc.normalize_section_name("totally_unknown_section") is None


def test_skeleton_retry_and_missing_keys_logic():
    req = ("section", "items")
    assert missing_required_keys({}, req) == list(req)
    assert missing_required_keys({"section": "x", "items": []}, req) == []
    src = inspect.getsource(mod)
    assert "skeleton_retry=True" in src  # skeleton retry wired into the per-doc ladder


def test_minimal_schema_fallback_is_review_bound():
    s = fc.ALLOWED_SECTIONS[0]
    m = fc.minimal_section_object(s)
    assert m["items"] == [] and m["needs_review"] is True and m["warnings"] == []
    assert fc.is_minimal_section(m, s) is True
    src = inspect.getsource(mod)
    assert "minimal_section_object" in src
    assert "do not promote" in src.lower() or "review-bound only" in src.lower() or "review_bound" in src.lower()


def test_no_provider_call_during_local_gate():
    s = _summary()
    if not (s["corpus1_live_started"] or s["corpus2_live_started"]):
        assert s["provider_model_call_made"] is False
        assert s["gemini_call_made"] is False
    src = inspect.getsource(mod)
    assert "if args.live" in src  # provider path only under --live


def test_fast_fail_threshold_and_no_proving_tranche():
    s = _summary()
    assert s["internal_fast_fail_threshold_docs"] == 25
    assert s["separate_proving_tranche_used"] is False


def test_no_mkb_autoaccept_medical_and_checkpoint_isolation():
    s = _summary()
    for k in ("mkb_db_opened", "active_mkb_write", "auto_accept_enabled", "medical_decision_made",
              "future_mkb_import_started", "corpus1_main_checkpoint_mutated", "r16_checkpoint_mutated"):
        assert s[k] is False, k
    assert "corpus1_r17_checkpoint" in str(mod.C1_CKPT)
    assert "corpus2_r17_checkpoint" in str(mod.C2_CKPT)


def test_no_private_committed_flags():
    s = _summary()
    for k in ("private_artifacts_committed", "raw_ai_response_committed", "tokenized_payloads_committed",
              "token_maps_committed", "pi_values_committed", "credentials_or_tokens_committed",
              "corpus2_oversized_docs_sent"):
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
