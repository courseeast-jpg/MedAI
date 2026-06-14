"""Local-only tests for R14 provider hard-stop diagnosis."""
from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
import scripts.run_medai_ai_first_corpus_17c_r2_provider_hard_stop_diagnosis_local_only_r14 as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_provider_hard_stop_diagnosis_local_only_r14"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_script_runs_without_provider_generation_call() -> None:
    result = subprocess.run(
        ["python", "scripts/run_medai_ai_first_corpus_17c_r2_provider_hard_stop_diagnosis_local_only_r14.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PROVIDER-HARD-STOP-DIAGNOSIS-LOCAL-ONLY-R14_PASS" in result.stdout
    src = inspect.getsource(mod)
    assert "_default_http_post" not in src
    assert "acquire_google_cloud_access_token" not in src
    assert "run_autonomous_recovery(live=True" not in src
    s = _summary()
    assert s["provider_generation_call_made"] is False
    assert s["billing_api_call_made"] is False


def test_no_mkb_write_or_private_artifact_commit_flags() -> None:
    s = _summary()
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["local_only"] is True


def test_provider_error_class_public_safe_and_explicit() -> None:
    s = _summary()
    assert s["exact_provider_error_class"] in {
        "api_disabled",
        "permission_denied",
        "quota_exhausted",
        "billing_or_credit_block",
        "model_unavailable",
        "region_unavailable",
        "rate_limit_or_transient",
        "malformed_request",
        "unknown_provider_hard_stop",
    }
    assert s["exact_provider_error_message_sanitized"]
    assert "Bearer " not in s["exact_provider_error_message_sanitized"]
    assert "ya29." not in s["exact_provider_error_message_sanitized"]


def test_checkpoint_status_contains_counts_only_and_resume_explicit() -> None:
    report = json.loads((REPORT_DIR / "checkpoint_resume_status_public.json").read_text(encoding="utf-8"))
    assert set(report) == {
        "completed_docs_checkpointed",
        "completed_doc_count_checkpointed",
        "remaining_docs_estimated",
        "resume_without_completed_doc_resend_supported",
        "ready_to_resume_after_provider_fix",
    }
    assert report["completed_doc_count_checkpointed"] == _summary()["completed_doc_count_checkpointed"]
    assert isinstance(report["resume_without_completed_doc_resend_supported"], bool)


def test_public_reports_have_no_private_paths_secrets_phi_or_raw_body_markers() -> None:
    for path in REPORT_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert '"response"' not in text
        assert "tokenized_content" not in text
        assert "raw provider response" not in text.lower()
        result = check_public_report_payload(text)
        assert result.passed, (path.name, result.private_filename_path_leaks, result.secret_leaks)


def test_recommendation_is_explicit() -> None:
    s = _summary()
    assert isinstance(s["requires_code_change_before_resume"], bool)
    assert isinstance(s["requires_provider_account_action"], bool)
    assert s["requires_code_change_before_resume"] is False
    assert s["requires_provider_account_action"] is True
