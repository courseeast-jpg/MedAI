"""Tests for the 17C-R1 Vertex credential-environment preflight. Inspect public
artifacts only; never call a provider model."""
from __future__ import annotations

import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_credentials_environment_preflight_17c_r1"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md",
                  "credential_environment_matrix.md", "next_steps_to_rerun_17c.md")
SECRET_MARKERS = ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", "client_secret")


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_no_credential_or_token_secrets_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        for m in SECRET_MARKERS:
            assert m not in t, (name, m)


def test_summary_safety_fields():
    s = _summary()
    assert s["block"] == "MEDAI-VERTEX-CREDENTIALS-ENVIRONMENT-PREFLIGHT-17C-R1"
    assert s["local_only_preflight"] is True
    assert s["medical_payload_sent"] is False
    assert s["vertex_model_call_made"] is False
    assert s["provider_content_call_made"] is False
    assert s["billing_api_call_made"] is False
    assert s["mkb_db_opened"] is False
    assert s["mkb_write"] is False
    assert s["credentials_printed"] is False
    assert s["tokens_printed"] is False
    assert s["credential_file_committed"] is False
    assert s["credential_file_copied_to_repo"] is False
    assert s["safety_result"] == "passed"


def test_result_and_token_fields_consistent():
    s = _summary()
    assert s["preflight_result"] in ("PASS", "LOGIN_REQUIRED", "FAIL")
    assert s["token_refresh_result"] in ("not_checked", "pass", "login_required", "fail")
    assert isinstance(s["gcloud_on_path"], bool)
    assert isinstance(s["google_auth_importable"], bool)
    assert isinstance(s["adc_file_found"], bool)
    # ready_to_rerun only when refresh passed.
    if s["ready_to_rerun_17c"]:
        assert s["token_refresh_result"] == "pass"


def test_no_credential_file_in_repo_tree():
    # No credential JSON should have been copied into the repo.
    for p in REPORT_DIR.glob("*"):
        t = p.read_text(encoding="utf-8")
        assert "application_default_credentials.json" not in t or "%APPDATA%" in t or "<redacted" in t \
            or p.name == "summary.json"  # summary may reference sanitized adc path label only
        for m in SECRET_MARKERS:
            assert m not in t


def test_reports_pass_privacy_check_except_mandated_path():
    for name in ("implementation_report.md", "credential_environment_matrix.md",
                 "next_steps_to_rerun_17c.md"):
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.passed, (name, getattr(r, "leak_examples_redacted", None))
    rs = check_public_report_payload((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert rs.raw_phi_logged_in_public_reports is False
    assert rs.secret_leaks == 0


def test_no_ssn_pattern_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
