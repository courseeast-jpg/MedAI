"""Tests for Corpus 2 / P2 private prep (local-only). Inspect PUBLIC artifacts only;
never call a provider, never open MKB, never read or print PI."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
import scripts.run_medai_ai_first_corpus2_p2_private_prep_local_only_01 as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus2_p2_private_prep_local_only_01"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS2_P2_PRIVATE_PREP_LOCAL_ONLY_01"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "corpus2_p2_inventory_public.json",
                  "corpus2_p2_privacy_validation_public.json", "corpus2_p2_outbound_package_public.json",
                  "corpus2_p2_live_readiness_public.md", "safety_boundary_public.md")
_WIN_PATH = re.compile(r"[A-Za-z]:\\")
_LONG_HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])")


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_doc_exists():
    assert (DOC_DIR / "MEDAI_AI_FIRST_CORPUS2_P2_PRIVATE_PREP_LOCAL_ONLY_01.md").exists()


def test_no_provider_no_live_no_mkb():
    s = _summary()
    for k in ("provider_model_call_made", "vertex_call_made", "gemini_call_made",
              "billing_api_call_made", "live_gate_set", "mkb_db_opened", "active_mkb_write",
              "auto_accept_enabled", "medical_decision_made"):
        assert s[k] is False, k
    assert s["local_only"] is True
    src = inspect.getsource(mod)
    # No provider transport in this local-only prep script.
    assert "_default_http_post" not in src
    assert "generate_content" not in src


def test_redacted_path_labels_only():
    s = _summary()
    assert s["input_folder_label"] == "CORPUS2_P2_INPUT_FOLDER_REDACTED"
    assert s["pi_vault_template_label"] == "PERSON2_PI_VAULT_TEMPLATE_REDACTED"
    assert s["private_output_root_label"] == "CORPUS2_P2_PRIVATE_OUTPUT_ROOT_REDACTED"
    assert s["person_label"] == "P2"


def test_inventory_counts_consistent():
    s = _summary()
    assert s["files_discovered_total"] >= 1
    assert s["pdf_files_discovered"] >= 1
    assert s["pdf_files_discovered"] <= s["files_discovered_total"]
    assert s["supported_files"] + s["unsupported_files"] == s["files_discovered_total"]
    inv = json.loads((REPORT_DIR / "corpus2_p2_inventory_public.json").read_text(encoding="utf-8"))
    assert inv["files_discovered_total"] == s["files_discovered_total"]
    assert len(inv["documents"]) == s["files_discovered_total"]


def test_pi_vault_and_tokenization():
    s = _summary()
    assert s["pi_vault_loaded"] is True
    assert s["private_value_count"] >= 1
    assert s["tokenized_payloads_private_only"] is True
    assert s["token_maps_private_only"] is True
    assert s["extraction_succeeded"] >= 1


def test_privacy_validation_clean_payloads():
    s = _summary()
    assert s["total_residual_pi_pattern_count"] == 0
    assert s["total_vault_value_leak_count"] == 0
    pv = json.loads((REPORT_DIR / "corpus2_p2_privacy_validation_public.json").read_text(encoding="utf-8"))
    assert pv["all_outbound_payloads_clean"] is True
    assert sum(r["vault_value_leak_count"] for r in pv["per_document"]) == 0


def test_outbound_package_public():
    s = _summary()
    pkg = json.loads((REPORT_DIR / "corpus2_p2_outbound_package_public.json").read_text(encoding="utf-8"))
    if s["tokenized_request_count"] > 0:
        assert s["outbound_requests_private_jsonl_created"] is True
        assert s["outbound_requests_sha256_created"] is True
        assert s["doc_id_manifest_private_created"] is True
        assert pkg["integrity_ok"] is True
    assert pkg["outbound_package_committed_to_repo"] is False


def test_readiness_gate_consistency():
    s = _summary()
    if s["ready_for_corpus2_p2_live_ai_extraction"]:
        assert s["pi_vault_loaded"] is True
        assert s["tokenized_request_count"] > 0
        assert s["total_residual_pi_pattern_count"] == 0
        assert s["total_vault_value_leak_count"] == 0
        assert s["extraction_failed"] == 0
    assert s["ready_for_corpus2_p2_live_ai_extraction"] == (not s["requires_human_fix_before_live"])


def test_no_private_committed_flags():
    s = _summary()
    for k in ("private_artifacts_committed", "raw_ocr_committed", "tokenized_payloads_committed",
              "token_maps_committed", "pi_values_committed", "credentials_or_tokens_committed"):
        assert s[k] is False, k


def test_no_private_paths_or_secrets_in_public_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert "MedAI_Private" not in t, name
        assert not _WIN_PATH.search(t), name
        assert not _LONG_HEX.search(t), (name, "long-hex/secret-like token present")
        for m in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", "Bearer "):
            assert m not in t, (name, m)


def test_public_reports_pass_privacy_check():
    s = _summary()
    assert s["public_report_phi_leak_count"] == 0
    assert s["private_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
    for name in PUBLIC_REPORTS:
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.passed, (name, getattr(r, "leak_examples_redacted", None))


def test_no_ssn_pattern_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
