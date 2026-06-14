"""Tests for the 17C-R2-R4 operator-context canonical batch restore. Inspect artifacts
and operator-context filesystem state only; never call a provider model."""
from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.canonical_batch_paths import resolve_canonical_batch
import scripts.run_medai_ai_first_corpus_operator_context_canonical_batch_restore_17c_r2_r4 as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_operator_context_canonical_batch_restore_17c_r2_r4"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_OPERATOR_CONTEXT_CANONICAL_BATCH_RESTORE_17C_R2_R4"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md",
                  "operator_context_path_diagnostics_public.json", "restored_batch_integrity_public.json",
                  "credential_preflight_public.json", "rerun_17c_r2_entry_gate.md")
PATH_BEARING = {"summary.json", "implementation_report.md",
                "operator_context_path_diagnostics_public.json", "rerun_17c_r2_entry_gate.md"}


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_required_docs_exist():
    for d in (
        "MEDAI_OPERATOR_CONTEXT_CANONICAL_BATCH_RESTORE_17C_R2_R4.md",
        "MEDAI_PRIVATE_ARTIFACT_VISIBILITY_POLICY_17C_R2_R4.md",
        "MEDAI_17C_R2_LIVE_RERUN_ENTRY_GATE_AFTER_R4.md",
    ):
        assert (DOC_DIR / d).exists(), d


def test_no_provider_model_call_no_live_gate_no_mkb_in_source():
    src = inspect.getsource(mod)
    for marker in ("generate_content", ":generateContent", "_default_http_post",
                   "import sqlite3", "chromadb", "from clinical_knowledge.mkb"):
        assert marker not in src, marker
    # No live-gate env is set in this preflight/restore.
    assert "os.environ[" not in src


def test_summary_safe_no_model_call():
    s = _summary()
    assert s["block"] == "MEDAI-AI-FIRST-CORPUS-OPERATOR-CONTEXT-CANONICAL-BATCH-RESTORE-17C-R2-R4"
    assert s["local_preflight_only"] is True
    assert s["provider_model_call_made"] is False
    assert s["vertex_model_call_made"] is False
    assert s["gemini_call_made"] is False
    assert s["claude_call_made"] is False
    assert s["openai_call_made"] is False
    assert s["billing_api_call_made"] is False
    assert s["live_gate_set"] is False
    assert s["live_extraction_started"] is False
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["operator_context_checked"] is True
    assert s["private_outbound_requests_committed"] is False
    assert s["tokenized_payloads_written_to_repo"] is False
    assert s["raw_ocr_written_to_repo"] is False
    assert s["token_maps_written_to_repo"] is False
    assert s["private_identifier_values_written_to_repo"] is False
    assert s["credential_or_token_written_to_repo"] is False
    assert s["public_report_phi_leak_count"] == 0


def test_canonical_batch_exists_after_and_readable():
    s = _summary()
    assert s["canonical_batch_exists_after"] is True
    assert s["canonical_batch_is_file_after"] is True
    # Read-only file must be treated as readable, not missing.
    assert s["canonical_batch_readable_after"] is True


def test_sealed_batch_diagnostics():
    s = _summary()
    assert s["sha256_sidecar_verified"] is True
    assert s["sealed_batch_valid"] is True
    assert s["physical_newline_record_count"] == 478
    assert s["parseable_record_count"] == 478
    assert s["unique_doc_ids"] == 478
    assert s["malformed_record_count"] == 0
    assert s["request_validation_passed"] is True
    assert s["residual_pi_failures"] == 0
    assert s["old_12_added_separately"] is False
    assert s["combined_batch_count"] == 478


def test_r3_resolver_passes_after_restore():
    # Same operator-context assertions the R3 resolver tests make.
    path, exists = resolve_canonical_batch()
    assert exists is True
    assert path.is_file() is True
    saved = os.environ.pop("LOCALAPPDATA", None)
    try:
        p2, ok2 = resolve_canonical_batch()
        assert ok2 is True and p2.is_file() is True
    finally:
        if saved is not None:
            os.environ["LOCALAPPDATA"] = saved


def test_credential_reported_without_token():
    cp = json.loads((REPORT_DIR / "credential_preflight_public.json").read_text(encoding="utf-8"))
    assert "credential_preflight_passed" in cp
    assert cp["vertex_model_call_made"] is False
    for m in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key"):
        assert m not in json.dumps(cp)


def test_ready_flag_consistency():
    s = _summary()
    if s["sealed_batch_valid"] and s["credential_preflight_passed"]:
        assert s["ready_to_rerun_17c_r2_live"] is True
    if not s["credential_preflight_passed"]:
        assert s["ready_to_rerun_17c_r2_live"] is False


def test_no_credentials_or_payload_markers_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        for m in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", '"tokenized_content"'):
            assert m not in t, (name, m)


def test_reports_no_phi_or_secret():
    for name in PUBLIC_REPORTS:
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.raw_phi_logged_in_public_reports is False, name
        assert r.secret_leaks == 0, name
        if name not in PATH_BEARING:
            assert r.passed, (name, getattr(r, "leak_examples_redacted", None))


def test_no_ssn_pattern_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
