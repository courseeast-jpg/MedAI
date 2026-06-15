"""Tests for R22 review package consolidation and extraction closure."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
import scripts.run_medai_r22_review_package_consolidation_and_extraction_closure as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_r22_review_package_consolidation_and_extraction_closure"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_script_runs_and_writes_review_package() -> None:
    result = subprocess.run(
        ["python", "scripts/run_medai_r22_review_package_consolidation_and_extraction_closure.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for name in (
        "summary.json",
        "implementation_report.md",
        "extraction_closure_manifest_public.json",
        "operator_review_queue_public.json",
        "review_only_finalized_public.json",
        "minimal_review_bound_public.json",
        "full_schema_content_public.json",
        "non_sendable_exclusions_public.json",
        "next_action_matrix_public.md",
        "safety_boundary_public.md",
    ):
        assert (REPORT_DIR / name).exists(), name


def test_required_counts_and_live_stopped() -> None:
    s = _summary()
    assert s["overall_result"] == "PASS"
    assert s["live_extraction_stopped"] is True
    assert s["total_content_packages_available"] == 163
    assert s["full_schema_package_count"] == 161
    assert s["minimal_review_bound_package_count"] == 2
    assert s["review_only_finalized_count"] == 315
    assert s["non_sendable_excluded_count"] == 2
    assert s["remaining_failed_for_review_or_review_only"] == 315
    assert s["unresolved_candidates_after_r22"] == 0


def test_no_provider_mkb_auto_accept_or_medical_decision() -> None:
    s = _summary()
    for key in (
        "provider_model_call_made",
        "gemini_call_made",
        "vertex_call_made",
        "billing_api_call_made",
        "mkb_db_opened",
        "active_mkb_write",
        "future_mkb_import_started",
        "auto_accept_enabled",
        "medical_decision_made",
    ):
        assert s[key] is False


def test_operator_queue_records_have_public_allowed_fields_only() -> None:
    queue = json.loads((REPORT_DIR / "operator_review_queue_public.json").read_text(encoding="utf-8"))
    allowed = {
        "corpus_id", "document_id", "source_phase", "terminal_state", "package_type",
        "review_priority", "reason_code", "evidence_anchor_count", "section_count",
        "warnings_count", "allowed_next_action",
    }
    assert len(queue["records"]) == 480
    for record in queue["records"]:
        assert set(record) == allowed
        assert record["review_priority"] in {"high", "normal", "low"}
        assert record["allowed_next_action"] in mod.ALLOWED_ACTIONS
        assert "tokenized" not in json.dumps(record).lower()


def test_priority_counts_match_queue() -> None:
    s = _summary()
    queue = json.loads((REPORT_DIR / "operator_review_queue_public.json").read_text(encoding="utf-8"))
    counts = {"high": 0, "normal": 0, "low": 0}
    for record in queue["records"]:
        counts[record["review_priority"]] += 1
    assert counts["high"] == s["review_priority_high_count"]
    assert counts["normal"] == s["review_priority_normal_count"]
    assert counts["low"] == s["review_priority_low_count"]
    assert sum(counts.values()) == 480


def test_next_action_matrix_disallows_mkb_and_provider_retry() -> None:
    text = (REPORT_DIR / "next_action_matrix_public.md").read_text(encoding="utf-8")
    for action in mod.ALLOWED_ACTIONS:
        assert action in text
    for action in mod.DISALLOWED_ACTIONS:
        assert action in text
    assert "send_review_only_records_to_provider_again" in text
    assert "write_to_mkb_now" in text


def test_public_reports_privacy_clean() -> None:
    for report in REPORT_DIR.glob("*"):
        if report.suffix.lower() in {".json", ".md"}:
            text = report.read_text(encoding="utf-8", errors="ignore")
            assert "C:\\Users\\S1" not in text
            assert "MedAI_Private" not in text
            assert "tokenized_content" not in text
            assert "Bearer " not in text
            assert "ya29." not in text
            result = check_public_report_payload(text)
            assert result.passed, report.name


def test_private_artifact_flags_false() -> None:
    s = _summary()
    for key in (
        "private_artifacts_committed",
        "raw_ai_response_committed",
        "raw_text_committed",
        "tokenized_payloads_committed",
        "token_maps_committed",
        "pi_values_committed",
        "credentials_or_tokens_committed",
    ):
        assert s[key] is False
    assert s["public_report_phi_leak_count"] == 0
    assert s["private_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
