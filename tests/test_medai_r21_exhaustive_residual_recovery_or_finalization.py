"""Tests for R21 exhaustive residual recovery or terminal finalization."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
import scripts.run_medai_r21_exhaustive_residual_recovery_or_finalization as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_r21_exhaustive_residual_recovery_or_finalization"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_local_gate_runs_and_creates_repaired_private_payloads() -> None:
    result = subprocess.run(
        ["python", "scripts/run_medai_r21_exhaustive_residual_recovery_or_finalization.py", "--local-only"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    s = _summary()
    assert s["r21_candidates_in_scope"] == 323
    assert s["private_repaired_payloads_created"] == 97
    assert s["finalized_review_only_before_provider"] == 226
    assert s["excluded_non_sendable_finalized"] == 2
    assert s["provider_model_call_made"] is False


def test_active_repairs_are_counted() -> None:
    s = _summary()
    assert s["hidden_text_dedup_repairs_applied"] == 75
    assert s["nonclinical_noise_repairs_applied"] == 75
    assert s["section_boundary_repairs_applied"] == 18
    assert s["oversized_request_repairs_applied"] == 4
    assert s["schema_repairs_applied"] >= 22


def test_all_local_review_and_non_sendable_records_have_terminal_states() -> None:
    terminal, states, reasons = mod._terminal_counts()
    assert states["finalized_review_only_with_reason"] >= 226
    assert states["excluded_non_sendable_with_reason"] == 2
    assert reasons["excluded_rtf_signal_container"] == 2
    assert "evidence_missing_review_only" in reasons


def test_no_provider_mkb_or_auto_accept_during_local_gate() -> None:
    s = _summary()
    assert s["local_only"] is True
    assert s["provider_model_call_made"] is False
    assert s["gemini_call_made"] is False
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False


def test_route_choice_includes_flash_and_pro() -> None:
    g = mod.build_local_gate()
    assert g["models"][mod.FLASH_MODEL] == 75
    assert g["models"][mod.PRO_MODEL] == 22


def test_timeout_and_schema_terminal_reason_constants_present() -> None:
    assert "provider_timeout_review_only" in mod.REVIEW_REASONS
    assert "schema_unrecoverable_review_only" in mod.REVIEW_REASONS
    assert "zero_yield_after_broad_repair" in mod.REVIEW_REASONS
    assert "blocked_by_hard_global_stop" in mod.TERMINAL_STATES


def test_public_reports_privacy_clean_and_no_private_payloads() -> None:
    for report in REPORT_DIR.glob("*"):
        if report.suffix.lower() in {".json", ".md"}:
            text = report.read_text(encoding="utf-8", errors="ignore")
            assert "tokenized_content" not in text
            assert "MedAI_Private" not in text
            assert "C:\\Users\\S1" not in text
            assert "Bearer " not in text
            assert "ya29." not in text
            result = check_public_report_payload(text)
            assert result.passed, report.name


def test_simulated_completion_has_no_unresolved_candidates() -> None:
    g = mod.build_local_gate()
    terminal = {str(item["document_id"]): item for item in g["terminal"]}
    for item in g["provider_queue"]:
        terminal[str(item["document_id"])] = {
            "document_id": item["document_id"],
            "terminal_state": "finalized_review_only_with_reason",
            "reason": "zero_yield_after_broad_repair",
            "provider_attempted": False,
        }
    in_scope_terminal = sum(1 for v in terminal.values() if v["reason"] != "excluded_rtf_signal_container")
    assert in_scope_terminal == 323
