"""No-live tests for the 17C-R2-R1 478-batch integrity restore. Inspect public
artifacts only; never call a provider."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_ai_first_corpus_478_batch_integrity_restore_local_only_17c_r2_r1 as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_batch_integrity_restore_local_only_17c_r2_r1"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_478_BATCH_INTEGRITY_RESTORE_LOCAL_ONLY_17C_R2_R1"

PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "integrity_restore_matrix.md",
                  "private_batch_integrity_public.json", "mutation_risk_public.md",
                  "rerun_17c_r2_entry_gate.md")


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_required_docs_exist():
    for d in (
        "MEDAI_AI_FIRST_CORPUS_478_BATCH_INTEGRITY_RESTORE_LOCAL_ONLY_17C_R2_R1.md",
        "MEDAI_PRIVATE_BATCH_INTEGRITY_SEAL_POLICY_17C_R2_R1.md",
        "MEDAI_MUTATION_RISK_AND_READONLY_POLICY_17C_R2_R1.md",
        "MEDAI_17C_R2_RERUN_ENTRY_CRITERIA_AFTER_INTEGRITY_RESTORE.md",
    ):
        assert (DOC_DIR / d).exists(), d


def test_no_provider_network_billing_imports():
    src = inspect.getsource(mod)
    for marker in (
        "import requests", "import httpx", "import aiohttp", "import urllib",
        "import google", "from google", "import vertexai", "generativeai",
        "import anthropic", "import openai", "generate_content", "cloudbilling",
        "billing_v1", "import sqlite3", "chromadb", "from clinical_knowledge.mkb",
        "_default_http_post", "acquire_google_cloud_access_token",
    ):
        assert marker not in src, marker


def test_no_live_gate_activation():
    src = inspect.getsource(mod)
    assert "os.environ[" not in src


def test_summary_integrity_fields():
    s = _summary()
    assert s["block"] == "MEDAI-AI-FIRST-CORPUS-478-BATCH-INTEGRITY-RESTORE-LOCAL-ONLY-17C-R2-R1"
    assert s["local_only"] is True
    assert s["provider_call_made"] is False
    assert s["vertex_live_execution"] is False
    assert s["gemini_call_made"] is False
    assert s["claude_call_made"] is False
    assert s["openai_call_made"] is False
    assert s["billing_api_call_made"] is False
    assert s["live_gate_set"] is False
    assert s["live_extraction_started"] is False
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["production_queue_mutated"] is False
    assert s["expected_request_count"] == 478
    assert s["rebuilt_request_count"] == 478
    assert s["parseable_after"] == 478
    assert s["unique_doc_ids_after"] == 478
    assert s["malformed_after"] == 0
    assert s["request_validation_passed_after"] is True
    assert s["residual_pi_failures_after"] == 0
    assert s["old_12_added_separately"] is False
    assert s["combined_batch_count"] == 478
    assert s["private_integrity_sidecar_written"] is True
    assert s["sha256_sidecar_written"] is True
    assert s["doc_id_manifest_written"] is True
    assert s["private_outbound_requests_committed"] is False
    assert s["tokenized_payloads_written_to_repo"] is False
    assert s["raw_ocr_written_to_repo"] is False
    assert s["token_maps_written_to_repo"] is False
    assert s["private_identifier_values_written_to_repo"] is False
    assert s["public_report_phi_leak_count"] == 0
    assert s["future_17c_r2_live_not_started"] is True
    assert s["privacy_result"] == "passed"
    assert s["safety_result"] == "passed"


def test_canonical_batch_path_outside_repo():
    s = _summary()
    p = Path(s["canonical_batch_path"])
    assert REPO_ROOT not in p.parents
    assert "MedAI_Private" in str(p)


def test_public_reports_no_request_bodies_or_secrets():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert '"tokenized_content"' not in t
        assert "/home/" not in t
        assert "Bearer " not in t
        assert "ya29." not in t
    for name in PUBLIC_REPORTS:
        if name == "summary.json":
            continue
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.passed, (name, getattr(r, "leak_examples_redacted", None))
    rs = check_public_report_payload((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert rs.raw_phi_logged_in_public_reports is False
    assert rs.secret_leaks == 0


def test_no_ssn_pattern_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
