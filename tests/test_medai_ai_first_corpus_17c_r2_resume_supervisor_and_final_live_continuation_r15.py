"""Local-only tests for R15 resume supervisor."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.resume_supervisor import (
    EXPECTED_COMPLETED_FOR_R15,
    EXPECTED_DOCS,
    build_preflight_matrix,
    classify_checkpoint_state,
    simulate_resume,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_resume_supervisor_and_final_live_continuation_r15"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_local_only_script_passes_and_writes_reports() -> None:
    result = subprocess.run(
        ["python", "scripts/run_medai_ai_first_corpus_17c_r2_resume_supervisor_and_final_live_continuation_r15.py", "--local-only"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "RESUME-SUPERVISOR-AND-FINAL-LIVE-CONTINUATION-R15_PASS" in result.stdout
    for name in (
        "summary.json",
        "implementation_report.md",
        "checkpoint_state_public.json",
        "preflight_matrix_public.json",
        "dry_run_resume_simulation_public.json",
        "live_continuation_public_report.md",
        "failed_docs_public.json",
        "safety_boundary_public.md",
    ):
        assert (REPORT_DIR / name).exists(), name


def test_clean_partial_checkpoint_is_resumable_and_counts_preserved() -> None:
    s = _summary()
    assert s["checkpoint_state_before"] == "clean_partial_checkpoint_resumable"
    assert s["docs_completed_before"] == EXPECTED_COMPLETED_FOR_R15
    assert s["docs_remaining_selected_before"] == EXPECTED_DOCS - EXPECTED_COMPLETED_FOR_R15
    assert s["completed_docs_preserved"] is True
    assert s["completed_docs_skipped_on_resume"] is True
    assert s["completed_sections_skipped_on_resume"] is True


def test_classifier_blocks_unresolved_failed_corrupt_and_sha_mismatch() -> None:
    order = [f"doc_{i}" for i in range(5)]
    assert classify_checkpoint_state(
        batch_sha="sha",
        order=order,
        total=5,
        completed=["doc_0"],
        failed={"failed_doc_id": "doc_1", "resolved": False},
        state={"canonical_batch_sha256": "sha"},
        failed_review_count=0,
    ).state == "unresolved_failed_checkpoint"
    assert classify_checkpoint_state(
        batch_sha="sha",
        order=order,
        total=5,
        completed=["doc_0", "doc_0"],
        failed=None,
        state={"canonical_batch_sha256": "sha"},
        failed_review_count=0,
    ).state == "corrupted_checkpoint"
    assert classify_checkpoint_state(
        batch_sha="sha",
        order=order,
        total=5,
        completed=["doc_0"],
        failed=None,
        state={"canonical_batch_sha256": "other"},
        failed_review_count=0,
    ).state == "sha_mismatch_checkpoint"


def test_provider_free_simulation_does_not_call_gemini_and_selects_remaining() -> None:
    sim = json.loads((REPORT_DIR / "dry_run_resume_simulation_public.json").read_text(encoding="utf-8"))
    assert sim["provider_generation_call_made"] is False
    assert sim["completed_docs_skipped"] == EXPECTED_COMPLETED_FOR_R15
    assert sim["remaining_docs_selected"] == EXPECTED_DOCS - EXPECTED_COMPLETED_FOR_R15
    assert sim["completed_doc_resent"] is False
    assert sim["simulated_provider_failure_preserves_evidence"] is True
    assert sim["simulated_success_updates_checkpoint_without_mkb_write"] is True
    assert sim["mkb_db_opened"] is False
    assert sim["active_mkb_write"] is False


def test_preflight_matrix_and_cost_caps_pass() -> None:
    matrix = json.loads((REPORT_DIR / "preflight_matrix_public.json").read_text(encoding="utf-8"))
    assert matrix["canonical_batch_valid"] is True
    assert matrix["credential_preflight_passed"] is True
    assert matrix["project_is_sot_knowledge_ocr"] is True
    assert matrix["vertex_route_configured"] is True
    assert matrix["checkpoint_state_resumable"] is True
    assert matrix["cost_caps_passed"] is True
    assert matrix["total_cap_usd"] == 10.0
    assert matrix["per_chunk_cap_usd"] == 0.05
    assert matrix["preflight_matrix_passed"] is True


def test_safety_flags_and_public_report_redaction() -> None:
    s = _summary()
    assert s["provider_model_call_made_during_live_phase"] is False
    assert s["gemini_call_made_during_live_phase"] is False
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["future_17d_mkb_import_not_started"] is True
    for path in REPORT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".json", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert '"response"' not in text
            assert "tokenized_content" not in text
            result = check_public_report_payload(text)
            assert result.passed, (path.name, result.private_filename_path_leaks, result.secret_leaks)
