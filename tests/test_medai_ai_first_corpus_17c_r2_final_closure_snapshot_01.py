"""Tests for the Corpus 1 / 17C-R2 final closure snapshot (read-only). Inspect artifacts
only; never call a provider model and never open MKB."""
from __future__ import annotations

import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_final_closure_snapshot_01"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_FINAL_CLOSURE_SNAPSHOT_01"
SNAPSHOT_MD = REPO_ROOT / "docs" / "continuation_snapshots" / "MEDAI_AI_FIRST_CORPUS_17C_R2_FINAL_CLOSURE_SNAPSHOT_01.md"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "corpus1_final_state_public.json",
                  "completed_vs_failed_for_review_public.json", "reusable_architecture_assets_public.md",
                  "next_steps_policy_public.md", "safety_boundary_public.md")
_WIN_PATH = re.compile(r"[A-Za-z]:\\")


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_docs_exist():
    assert SNAPSHOT_MD.exists()
    assert any(DOC_DIR.glob("*.md")), "pilot_design closure dir must contain a design doc"


def test_no_provider_no_live_no_mkb():
    s = _summary()
    for k in ("provider_model_call_made", "live_extraction_started", "mkb_db_opened",
              "active_mkb_write", "auto_accept_enabled", "medical_decision_made"):
        assert s[k] is False, k
    assert s["local_only"] is True


def test_final_state_counts():
    s = _summary()
    assert s["corpus1_final_result"] == "LIVE_FAIL"
    assert s["docs_loaded"] == 478
    assert s["docs_completed"] == 118
    assert s["docs_failed_for_review"] == 360
    assert s["docs_unattempted"] == 0
    assert s["docs_completed"] + s["docs_failed_for_review"] == s["docs_loaded"]
    assert s["failure_stage"] == "provider_live_fail"
    assert s["failure_category"] == "api_disabled_or_permission"


def test_cost_and_caps():
    s = _summary()
    assert s["actual_total_token_count"] == 2504552
    assert s["actual_cost_public_if_available"] == "$0.383599"
    assert s["cost_cap_usd"] == 10.00
    assert s["per_chunk_cap_usd"] == 0.05


def test_reusable_assets_and_recommendation():
    s = _summary()
    assert s["checkpoint_resume_available"] is True
    assert s["sectioned_extraction_available"] is True
    assert s["autonomous_recovery_available"] is True
    assert s["failed_evidence_preserved"] is True
    assert s["ready_for_17d_mkb_import"] is False
    assert s["recommended_corpus1_action"] == "stop_live_retry_preserve_completed_and_failed_for_review"


def test_no_private_artifacts_or_leaks_flags():
    s = _summary()
    assert s["private_artifacts_committed"] is False
    assert s["public_report_phi_leak_count"] == 0
    assert s["private_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
    assert s["privacy_result"] == "passed"
    assert s["safety_result"] == "passed"


def test_no_private_windows_path_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert "MedAI_Private" not in t, name
        assert not _WIN_PATH.search(t), name


def test_no_secret_like_strings_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        for m in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", "Bearer "):
            assert m not in t, (name, m)


def test_reports_pass_privacy_check():
    for name in PUBLIC_REPORTS:
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.passed, (name, getattr(r, "leak_examples_redacted", None))


def test_no_ssn_pattern_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
