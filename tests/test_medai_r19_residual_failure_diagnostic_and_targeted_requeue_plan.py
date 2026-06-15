"""Tests for R19 residual failure diagnostic and targeted requeue plan."""
from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
import scripts.run_medai_r19_residual_failure_diagnostic_and_targeted_requeue_plan as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_r19_residual_failure_diagnostic_and_targeted_requeue_plan"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_script_runs_local_only_and_writes_reports() -> None:
    result = subprocess.run(
        ["python", "scripts/run_medai_r19_residual_failure_diagnostic_and_targeted_requeue_plan.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "MEDAI-R19-RESIDUAL-FAILURE-DIAGNOSTIC-AND-TARGETED-REQUEUE-PLAN_PASS" in result.stdout
    for name in (
        "summary.json",
        "implementation_report.md",
        "residual_failure_taxonomy_public.json",
        "corpus1_residual_diagnostic_public.json",
        "corpus2_residual_diagnostic_public.json",
        "targeted_requeue_plan_public.json",
        "review_only_plan_public.json",
        "safety_boundary_public.md",
    ):
        assert (REPORT_DIR / name).exists(), name


def test_no_provider_or_mkb_paths() -> None:
    src = inspect.getsource(mod)
    assert "acquire_google_cloud_access_token" not in src
    assert "_default_http_post" not in src
    assert "urllib.request" not in src
    s = _summary()
    assert s["local_only"] is True
    assert s["provider_model_call_made"] is False
    assert s["gemini_call_made"] is False
    assert s["vertex_call_made"] is False
    assert s["billing_api_call_made"] is False
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False


def test_expected_counts_and_exclusions() -> None:
    s = _summary()
    assert s["corpus1_content_packages_before"] == 159
    assert s["corpus1_residual_failed_before"] == 319
    assert s["corpus2_completed_before"] == 21
    assert s["corpus2_recoverable_failed_before"] == 2
    assert s["corpus2_excluded_rtf_signal"] == 2
    assert s["corpus1_residual_diagnosed"] == 319
    assert s["corpus2_residual_diagnosed"] == 2
    assert sum(s["failure_bucket_counts"].values()) == 321


def test_targeted_plan_only_failed_residuals_and_review_only_not_live() -> None:
    targeted = json.loads((REPORT_DIR / "targeted_requeue_plan_public.json").read_text(encoding="utf-8"))
    review = json.loads((REPORT_DIR / "review_only_plan_public.json").read_text(encoding="utf-8"))
    assert targeted["candidate_count"] == _summary()["targeted_requeue_candidate_count"]
    for doc in targeted["documents"]:
        assert doc["corpus"] in {"corpus1", "corpus2"}
        assert str(doc["doc_hash"]).startswith("doc_")
        assert doc["eligible_for_next_live"] is True
        assert doc["recommended_local_repair"] != "mark_review_only"
    for doc in review["documents"]:
        assert doc["eligible_for_next_live"] is False
        assert doc["recommended_local_repair"] == "mark_review_only"


def test_public_reports_no_raw_text_phi_private_paths_or_secrets() -> None:
    forbidden = ("tokenized_content", "raw OCR", "C:\\Users\\S1", "Bearer ", "ya29.")
    for path in REPORT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".json", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for marker in forbidden:
                assert marker not in text
            result = check_public_report_payload(text)
            assert result.passed, (path.name, result.private_filename_path_leaks, result.secret_leaks)


def test_private_artifact_flags_false() -> None:
    s = _summary()
    assert s["private_artifacts_committed"] is False
    assert s["raw_text_committed"] is False
    assert s["tokenized_payloads_committed"] is False
    assert s["token_maps_committed"] is False
    assert s["pi_values_committed"] is False
    assert s["credentials_or_tokens_committed"] is False
