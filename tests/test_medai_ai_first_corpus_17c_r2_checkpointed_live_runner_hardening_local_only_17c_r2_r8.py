"""Tests for 17C-R2-R8 checkpointed live-runner hardening. Exercise the checkpoint
module + inspect artifacts; never call a provider model and never set a live gate."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution import live_checkpoint as lc
import scripts.run_medai_ai_first_corpus_17c_r2_checkpointed_live_runner_hardening_local_only_17c_r2_r8 as mod
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_checkpointed_live_runner_hardening_local_only_17c_r2_r8"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_CHECKPOINTED_LIVE_RUNNER_HARDENING_LOCAL_ONLY_17C_R2_R8"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "checkpoint_policy_public.json",
                  "simulation_result_public.json", "failed_evidence_preservation_public.json",
                  "live_entry_gate_public.md", "safety_boundary_public.md")


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_required_docs_exist():
    for d in (
        "MEDAI_17C_R2_CHECKPOINTED_LIVE_RUNNER_HARDENING_LOCAL_ONLY_17C_R2_R8.md",
        "MEDAI_17C_R2_CHECKPOINT_AND_RESUME_POLICY_R8.md",
        "MEDAI_17C_R2_FAILED_EVIDENCE_PRESERVATION_POLICY_R8.md",
        "MEDAI_17C_R2_LIVE_ENTRY_GATE_AFTER_R8.md",
    ):
        assert (DOC_DIR / d).exists(), d


def test_no_provider_no_live_no_mkb():
    s = _summary()
    for k in ("provider_model_call_made", "vertex_model_call_made", "gemini_call_made",
              "claude_call_made", "openai_call_made", "billing_api_call_made",
              "live_gate_set", "live_extraction_started", "mkb_db_opened", "active_mkb_write",
              "auto_accept_enabled", "medical_decision_made", "production_queue_mutated",
              "simulation_provider_call_made"):
        assert s[k] is False, k
    # The R8 driver itself must not reach the provider transport.
    src = inspect.getsource(mod)
    assert "_default_http_post" not in src
    assert "generate_content" not in src


def test_checkpoint_folder_outside_repo():
    assert lc.is_outside_repo(lc.CHECKPOINT_DIR, REPO_ROOT)
    assert lc.is_outside_repo(lc.CHECKPOINT_DIR_LABEL, REPO_ROOT)
    s = _summary()
    assert s["checkpoint_folder_outside_repo"] is True
    assert "MedAI_Private" in s["checkpoint_folder_private"]


def test_evidence_folder_outside_repo():
    assert lc.is_outside_repo(lc.EVIDENCE_DIR, REPO_ROOT)
    assert lc.is_outside_repo(lc.EVIDENCE_DIR_LABEL, REPO_ROOT)
    s = _summary()
    assert s["failed_evidence_preserve_folder_outside_repo"] is True
    assert "Downloads" in s["failed_evidence_preserve_folder"]


def test_checkpointing_added_and_runner_wired():
    s = _summary()
    assert s["checkpointing_added"] is True
    assert s["checkpoint_resume_supported"] is True
    assert all(s["runner_integration"].values())
    src = inspect.getsource(live)
    assert "live_checkpoint" in src
    assert "lc.decide_resume(" in src
    assert "lc.mark_completed(" in src
    assert "lc.mark_failed(" in src
    assert "lc.preserve_failed_evidence(" in src


def test_resume_skips_completed_and_resumes_at_next(tmp_path):
    base = tmp_path / "ck"
    order = ["docA", "docB", "docC"]
    sha = "a" * 64
    # Fresh -> start at first.
    start, blocked, reason, _ = lc.decide_resume(sha, order, 3, base=base)
    assert (start, blocked, reason) == (0, False, "no_checkpoint_start_from_first")
    lc.init_checkpoint("r", sha, "m", 0.40, 0.05, 3, base=base)
    lc.mark_completed("docA", 0, base=base)
    start, blocked, reason, completed = lc.decide_resume(sha, order, 3, base=base)
    assert start == 1                      # resume at request #2 (0-based index 1)
    assert blocked is False
    assert "docA" in completed             # completed doc is skipped, never re-sent
    assert "docB" not in completed


def test_failed_doc_blocks_until_triaged(tmp_path):
    base = tmp_path / "ck"
    order = ["docA", "docB"]
    sha = "b" * 64
    lc.init_checkpoint("r", sha, "m", 0.40, 0.05, 2, base=base)
    lc.mark_completed("docA", 0, base=base)
    lc.mark_failed("docB", 1, "missing_expected_schema_fields", base=base)
    start, blocked, reason, completed = lc.decide_resume(sha, order, 2, base=base)
    assert blocked is True
    assert reason == "failed_doc_unresolved_requires_triage_or_reset"
    assert start == 1                      # would resume at the failed request once cleared
    assert "docA" in completed
    # Operator triage clears the block.
    assert lc.resolve_failed(base=base) is True
    _, blocked2, _, _ = lc.decide_resume(sha, order, 2, base=base)
    assert blocked2 is False


def test_sha256_mismatch_blocks(tmp_path):
    base = tmp_path / "ck"
    order = ["docA"]
    lc.init_checkpoint("r", "c" * 64, "m", 0.40, 0.05, 1, base=base)
    _, blocked, reason, _ = lc.decide_resume("d" * 64, order, 1, base=base)
    assert blocked is True
    assert reason == "canonical_batch_sha256_mismatch"


def test_inconsistent_checkpoint_blocks(tmp_path):
    base = tmp_path / "ck"
    order = ["docA", "docB"]
    sha = "e" * 64
    lc.init_checkpoint("r", sha, "m", 0.40, 0.05, 2, base=base)
    # Completed contains a doc not in the batch order -> inconsistent.
    (base / lc.COMPLETED_FILE).write_text(json.dumps(["docZZZ"]), encoding="utf-8")
    _, blocked, reason, _ = lc.decide_resume(sha, order, 2, base=base)
    assert blocked is True
    assert reason == "checkpoint_inconsistent_unknown_completed_doc"


def test_evidence_preservation_creates_run_folder(tmp_path):
    stg = tmp_path / "stg"
    stg.mkdir()
    (stg / "live_responses_private.jsonl").write_text('{"x":1}\n', encoding="utf-8")
    ev = tmp_path / "ev"
    ck = tmp_path / "ck"
    lc.init_checkpoint("r", "f" * 64, "m", 0.40, 0.05, 1, base=ck)
    preserved, path, copied = lc.preserve_failed_evidence("sim", staging_dir=stg, base=ck, evidence_base=ev)
    assert preserved is True
    run_dir = Path(path)
    assert run_dir.is_dir()
    assert (run_dir / "README_PRIVATE_DO_NOT_SHARE.txt").is_file()
    assert (run_dir / "live_responses_private.jsonl").is_file()
    assert copied >= 1


def test_simulation_summary_fields():
    s = _summary()
    assert s["simulation_mode_run"] is True
    assert s["simulation_provider_call_made"] is False
    assert s["simulation_completed_request_count"] == 1
    assert s["simulation_failed_request_count"] == 1
    assert s["simulation_resume_next_index"] == 2          # resumes at request #2
    assert s["simulation_resends_completed_request"] is False
    assert s["simulation_evidence_preserved"] is True
    assert s["simulation_evidence_readme_present"] is True
    assert s["completed_requests_skipped_on_resume"] is True
    assert s["checkpoint_requires_sha256_match"] is True
    assert s["failed_doc_blocks_until_triaged"] is True
    assert s["failed_evidence_preservation_added"] is True


def test_entry_gate_consistency():
    s = _summary()
    assert s["authorized_hard_cost_cap_total_usd"] == 0.40
    assert s["hard_cost_cap_per_chunk_usd"] == 0.05
    assert s["resume_policy"] == "resume_from_next_unsent_request"
    if s["ready_to_resume_17c_r2_live"]:
        assert s["canonical_batch_valid"] is True
        assert s["credential_preflight_passed"] is True
        assert s["remaining_cost_estimate_within_cap"] is True
        assert float(s["remaining_cost_estimate_usd"]) <= 0.40


def test_no_private_body_or_credentials_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        for m in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key",
                  "Bearer ", '"tokenized_content"'):
            assert m not in t, (name, m)
    # The private-body sentinel string used inside the isolated sim must not surface here.
    for name in PUBLIC_REPORTS:
        assert "private body redacted" not in (REPORT_DIR / name).read_text(encoding="utf-8")


def test_no_committed_private_flags():
    s = _summary()
    for k in ("private_checkpoint_committed", "private_responses_committed",
              "parsed_private_responses_committed", "tokenized_payloads_written_to_repo",
              "raw_ocr_written_to_repo", "token_maps_written_to_repo",
              "private_identifier_values_written_to_repo", "credential_or_token_written_to_repo",
              "raw_failed_response_committed", "private_response_bodies_committed"):
        assert s[k] is False, k


def test_reports_pass_privacy_check():
    s = _summary()
    assert s["public_report_phi_leak_count"] == 0
    for name in PUBLIC_REPORTS:
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.raw_phi_logged_in_public_reports is False, name
        assert r.secret_leaks == 0, name


def test_no_ssn_pattern_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
