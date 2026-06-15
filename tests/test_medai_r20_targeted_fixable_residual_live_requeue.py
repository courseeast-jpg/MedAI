"""Tests for R20 targeted fixable residual live requeue."""
from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
import scripts.run_medai_r20_targeted_fixable_residual_live_requeue as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_r20_targeted_fixable_residual_live_requeue"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_local_gate_runs_and_selects_only_r19_targeted_candidates() -> None:
    result = subprocess.run(
        ["python", "scripts/run_medai_r20_targeted_fixable_residual_live_requeue.py", "--local-only"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    s = _summary()
    assert s["targeted_only"] is True
    assert s["targeted_candidates_selected"] == 97
    assert s["review_only_records_selected"] == 0
    assert s["completed_content_packages_selected"] == 0
    assert s["excluded_rtf_signal_selected"] == 0
    assert s["live_entry_gate_passed"] is True


def test_local_only_has_no_provider_mkb_or_auto_accept() -> None:
    s = _summary()
    assert s["local_only"] is True
    assert s["provider_model_call_made"] is False
    assert s["gemini_call_made"] is False
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["future_mkb_import_started"] is False


def test_private_queue_paths_are_outside_repo_and_not_reported() -> None:
    gate = mod.build_gate()
    assert len(gate["queue"]) == 97
    assert str(mod.PRIVATE_QUEUE).startswith(str(Path.home()).split("\\")[0]) or mod.PRIVATE_QUEUE.is_absolute()
    assert not str(mod.PRIVATE_QUEUE).startswith(str(REPO_ROOT))
    for report in REPORT_DIR.glob("*"):
        if report.suffix.lower() in {".json", ".md"}:
            text = report.read_text(encoding="utf-8", errors="ignore")
            assert "MedAI_Private" not in text
            assert "C:\\Users\\S1" not in text
            assert "tokenized_content" not in text


def test_public_reports_privacy_clean() -> None:
    for report in REPORT_DIR.glob("*"):
        if report.suffix.lower() in {".json", ".md"}:
            text = report.read_text(encoding="utf-8", errors="ignore")
            result = check_public_report_payload(text)
            assert result.passed, report.name
            assert "Bearer " not in text
            assert "ya29." not in text
            assert "GEMINI_API_KEY" not in text


def test_live_mode_uses_existing_vertex_adapter_and_r17_contract() -> None:
    src = inspect.getsource(mod)
    assert "build_vertex_generate_content_url" in src
    assert "acquire_google_cloud_access_token" in src
    assert "r17._attempt_doc" in src
    assert "response_schema" not in src.lower()
    assert "grounding" not in src.lower()


def test_summary_has_required_safety_flags_false() -> None:
    s = _summary()
    for key in (
        "private_artifacts_committed",
        "raw_ai_response_committed",
        "tokenized_payloads_committed",
        "token_maps_committed",
        "pi_values_committed",
        "credentials_or_tokens_committed",
        "cost_cap_exceeded",
    ):
        assert s[key] is False
    assert s["public_report_phi_leak_count"] == 0
    assert s["private_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
