"""Local-only tests for 17C-R2-R12 checkpoint unblock and redaction repair."""
from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
import scripts.run_medai_ai_first_corpus_17c_r2_checkpoint_unblock_and_report_redaction_local_only_17c_r2_r12 as mod
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_checkpoint_unblock_redaction_and_full_live_run_17c_r2_r12"
LIVE_REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_live_batch_vertex_17c_r2"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_script_exits_and_reports_ready() -> None:
    result = subprocess.run(
        ["python", "scripts/run_medai_ai_first_corpus_17c_r2_checkpoint_unblock_and_report_redaction_local_only_17c_r2_r12.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "MEDAI-AI-FIRST-CORPUS-17C-R2-CHECKPOINT-UNBLOCK-REDACTION-AND-FULL-LIVE-RUN-17C-R2-R12_PASS" in result.stdout
    assert _summary()["ready_for_full_live_run"] is True


def test_reports_exist_and_checkpoint_is_clean_or_resumable() -> None:
    for name in ("summary.json", "implementation_report.md", "checkpoint_unblock_public.json",
                 "redaction_repair_public.json", "live_run_gate_public.md"):
        assert (REPORT_DIR / name).exists(), name
    s = _summary()
    assert s["checkpoint_clean_or_resumable_before_live"] is True
    assert s["unresolved_failed_doc_after_unblock"] is False
    assert s["checkpoint_inconsistent"] is False


def test_no_provider_or_live_side_effects_in_local_script() -> None:
    src = inspect.getsource(mod)
    assert "_default_http_post" not in src
    assert "live.run(" not in src
    assert "os.environ[" not in src
    s = _summary()
    assert s["provider_model_call_made"] is False
    assert s["vertex_model_call_made"] is False
    assert s["gemini_call_made"] is False
    assert s["billing_api_call_made"] is False
    assert s["live_gate_set"] is False
    assert s["live_extraction_started"] is False


def test_live_runner_gates_are_active_for_future_run() -> None:
    s = _summary()
    assert s["canonical_batch_valid"] is True
    assert s["request_count_total"] == 478
    assert live.CAP_TOTAL == 10.00
    assert live.CAP_PER_CHUNK == 0.05
    assert live.MAX_OUTPUT_TOKENS == 8192
    assert s["selected_chunk_size"] == 17
    assert s["credential_preflight_passed"] is True
    assert s["ready_for_full_live_run"] is True


def test_redaction_and_public_privacy_reports_pass() -> None:
    s = _summary()
    assert s["private_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
    assert s["public_report_phi_leak_count"] == 0
    for path in LIVE_REPORT_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".csv"}:
            continue
        result = check_public_report_payload(path.read_text(encoding="utf-8", errors="ignore"))
        assert result.passed, (path.name, result.private_filename_path_leaks, result.secret_leaks)


def test_no_private_artifacts_committed_flags() -> None:
    s = _summary()
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["private_checkpoint_evidence_responses_committed"] is False
    assert s["raw_ai_response_committed"] is False
    assert s["tokenized_payloads_committed"] is False
    assert s["credentials_tokens_committed"] is False
