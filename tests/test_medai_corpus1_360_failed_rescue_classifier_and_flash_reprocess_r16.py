"""Tests for Corpus 1 R16 360-failed flash rescue. Public artifacts + pure helpers only;
never call a provider, never open MKB, never mutate the Corpus 1 main checkpoint."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
import scripts.run_medai_corpus1_360_failed_rescue_classifier_and_flash_reprocess_r16 as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_corpus1_360_failed_rescue_classifier_and_flash_reprocess_r16"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "failure_taxonomy_public.json",
                  "live_entry_gate_public.md", "live_run_public_report.md", "failed_docs_public.json",
                  "safety_boundary_public.md")
_WIN = re.compile(r"[A-Za-z]:\\")
_LONG_HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])")


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for n in PUBLIC_REPORTS:
        assert (REPORT_DIR / n).exists(), n


def test_model_and_counts():
    s = _summary()
    assert s["target_model"] == "gemini-2.5-flash"
    assert s["previous_model"] == "gemini-2.5-flash-lite"
    assert s["docs_total"] == 478
    assert s["docs_completed_before"] == 118
    assert s["docs_failed_for_review_before"] == 360
    assert s["docs_selected_for_rescue"] == 360


def test_completed_preserved_not_reprocessed():
    s = _summary()
    assert s["completed_docs_preserved"] is True
    assert s["completed_docs_reprocessed"] is False
    assert s["corpus1_main_checkpoint_mutated"] is False


def test_failure_taxonomy_completed():
    s = _summary()
    assert s["failure_taxonomy_completed"] is True
    tax = json.loads((REPORT_DIR / "failure_taxonomy_public.json").read_text(encoding="utf-8"))
    assert sum(tax["buckets"].values()) == 360
    assert set(tax["buckets"].keys()) <= set(mod.TAXONOMY)


def test_selected_excludes_completed():
    g = mod.evaluate_gate()
    completed = g["completed"]
    sel_ids = {str(r.get("document_id") or "") for r in g["selected"]}
    assert sel_ids.isdisjoint(completed)
    assert len(g["selected"]) == 360


def test_privacy_and_cost_gate():
    s = _summary()
    assert s["privacy_result"] == "passed"
    assert float(s["estimated_cost_before_live_usd"]) <= s["remaining_cap_before_part_b_usd"]
    assert s["remaining_cap_before_part_b_usd"] <= 50.00


def test_no_provider_call_during_local_gate():
    s = _summary()
    if not s["live_run_started"]:
        assert s["provider_model_call_made"] is False
        assert s["gemini_call_made"] is False
    src = inspect.getsource(mod)
    assert 'if args.live and g["gate_passed"]:' in src  # provider only behind gate


def test_no_mkb_autoaccept_medical():
    s = _summary()
    for k in ("mkb_db_opened", "active_mkb_write", "auto_accept_enabled", "medical_decision_made",
              "future_mkb_import_started"):
        assert s[k] is False, k


def test_separate_r16_checkpoint_not_main():
    # The rescue checkpoint/evidence must be R16-specific (not the canonical Corpus 1 dirs).
    assert "corpus1_r16_flash_rescue" in str(mod.R16_CHECKPOINT_DIR)
    assert "R16" in str(mod.R16_EVIDENCE_DIR)


def test_no_private_artifacts_committed_flags():
    s = _summary()
    for k in ("private_artifacts_committed", "raw_ai_response_committed",
              "tokenized_payloads_committed", "token_maps_committed", "pi_values_committed",
              "credentials_or_tokens_committed"):
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
