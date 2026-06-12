"""Focused tests for MEDAI-AI-GEMINI-LIVE-SMOKE-GATED-15M."""
from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from clinical_knowledge.privacy import check_public_report_payload
from execution.gemini_live_smoke import (
    GEMINI_CREDENTIAL_ENV_VAR,
    ALLOW_SMOKE_ENV,
    OPERATOR_APPROVED_ENV,
    SYNTHETIC_PAYLOAD_CLASS,
    FakeGeminiLiveClient,
    GeminiLiveSmokeAuditRecord,
    GeminiLiveSmokeDecision,
    GeminiLiveSmokePolicy,
    GeminiLiveSmokeRequest,
    GeminiLiveSmokeResult,
    build_live_smoke_operator_preview,
    evaluate_live_smoke_gates,
    run_gemini_live_smoke,
    synthetic_payload_check,
)
from scripts.medai_ai_flat_validation_harness import (
    DEFAULT_DESELECT_PATTERNS,
    FlatHarnessConfig,
    run_flat_validation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTRINE = REPO_ROOT / "docs/architecture/MEDAI_CAPABILITY_BOUNDARY_DOCTRINE.md"
GEMINI_SECRET = "gemini-live-smoke-secret-value-15m"
LIVE_ENV = {
    ALLOW_SMOKE_ENV: "1",
    OPERATOR_APPROVED_ENV: "1",
    GEMINI_CREDENTIAL_ENV_VAR: GEMINI_SECRET,
}
RAW_FIXTURE_VALUES = ["Jane Example", "jane.example@example.com", "INS-ABC-12345", "01/02/1970", "Park Medical Center"]
DOCTRINE_PHRASES = [
    "Do not assign semantic/layout understanding to OCR or deterministic rules.",
    (
        "A MedAI extraction block is not successful because records were created. It is "
        "successful only if the operator can compare a source-faithful package against the "
        "original document quickly and safely."
    ),
]
PRIOR_FOCUSED_FILES = [
    "tests/test_medai_ai_extraction_workflow_seam_15a.py",
    "tests/test_medai_ai_extraction_privacy_gate_15b.py",
    "tests/test_medai_ai_provider_adapter_stub_15c.py",
    "tests/test_medai_ai_provider_selection_ui_15d.py",
    "tests/test_medai_ai_external_call_dry_run_15e.py",
    "tests/test_medai_ai_real_provider_enablement_gate_15f.py",
    "tests/test_medai_ai_gemini_adapter_disabled_15g.py",
    "tests/test_medai_ai_premium_adapters_disabled_15i.py",
    "tests/test_medai_ai_local_ollama_adapter_disabled_15j.py",
    "tests/test_medai_ai_provider_enablement_operator_control_15k.py",
    "tests/test_medai_ai_validation_harness_derecursion_15l.py",
]
DIRECT_COMMANDS = [
    ("regression_12a_uat", [sys.executable, "scripts/run_medai_operator_workflow_uat_12a.py"]),
    ("regression_13c_pytest", [sys.executable, "-m", "pytest", "tests/test_medai_operator_usability_polish_13c.py", "-p", "no:cacheprovider", "-q"]),
]


@functools.lru_cache(maxsize=1)
def _flat_run_prior() -> dict:
    config = FlatHarnessConfig(
        test_files=PRIOR_FOCUSED_FILES,
        deselect_patterns=DEFAULT_DESELECT_PATTERNS,
        direct_commands=DIRECT_COMMANDS,
        per_command_timeout_s=600,
        total_timeout_s=1800,
        allow_nested_scripts=False,
    )
    return run_flat_validation(config)


# 1. Policy classes exist.
def test_policy_classes_exist() -> None:
    assert GeminiLiveSmokePolicy is not None
    assert GeminiLiveSmokeRequest is not None
    assert GeminiLiveSmokeDecision is not None
    assert GeminiLiveSmokeResult is not None
    assert GeminiLiveSmokeAuditRecord is not None


# 2, 22-24. Default mode: no external call.
def test_default_mode_makes_no_external_call() -> None:
    result = run_gemini_live_smoke(environ={})
    assert result.gemini_real_call_attempted is False
    assert result.external_api_used is False
    assert result.real_network_call_used is False
    assert result.live_call_status.startswith("blocked")
    assert result.review_required is True
    assert result.active_written_count == 0


# 3. Missing GEMINI_API_KEY blocks live smoke.
def test_missing_credential_blocks() -> None:
    result = run_gemini_live_smoke(environ={ALLOW_SMOKE_ENV: "1", OPERATOR_APPROVED_ENV: "1"})
    assert result.live_call_status == "blocked_missing_credential"
    assert result.gemini_real_call_attempted is False


# 4-5. Missing approval env vars block live smoke.
def test_missing_allow_env_blocks() -> None:
    result = run_gemini_live_smoke(environ={OPERATOR_APPROVED_ENV: "1", GEMINI_CREDENTIAL_ENV_VAR: GEMINI_SECRET})
    assert result.live_call_status == "blocked_missing_operator_approval"


def test_missing_operator_approved_env_blocks() -> None:
    result = run_gemini_live_smoke(environ={ALLOW_SMOKE_ENV: "1", GEMINI_CREDENTIAL_ENV_VAR: GEMINI_SECRET})
    assert result.live_call_status == "blocked_missing_operator_approval"


# 6-15. Config-gate failures block (no call attempted).
@pytest.mark.parametrize(
    "overrides,expected_in_missing",
    [
        ({"operator_enablement_request_state": "not_requested"}, "operator_staged_request_required"),
        ({"privacy_gate_status": "fail_closed"}, "privacy_gate_not_passed"),
        ({"payload_policy_allowed": False}, "payload_policy_failed"),
        ({"budget_allowed": False}, "budget_exceeded"),
        ({"dry_run_status": "blocked"}, "dry_run_not_passed"),
        ({"call_limit": 2}, "call_limit_must_be_one"),
        ({"payload_class": "real_patient_document"}, "payload_class_not_synthetic_redacted_live_smoke"),
        ({"payload_type": "raw_pdf"}, "payload_type_not_redacted_text_layout_summary"),
        ({"payload_type": "raw_image"}, "payload_type_not_redacted_text_layout_summary"),
        ({"payload_type": "external_vision_payload"}, "payload_type_not_redacted_text_layout_summary"),
    ],
)
def test_config_gate_failures_block_live_smoke(overrides, expected_in_missing) -> None:
    fake = FakeGeminiLiveClient()
    result = run_gemini_live_smoke(GeminiLiveSmokeRequest(**overrides), environ=LIVE_ENV, gemini_client=fake)
    assert result.gemini_real_call_attempted is False
    assert fake.call_count == 0
    assert expected_in_missing in result.missing_gates


# 16-18. Credential value / token map / PII never in reports.
def test_no_credential_or_pii_in_result() -> None:
    fake = FakeGeminiLiveClient()
    result = run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=fake)
    serialized = json.dumps(
        {
            "result": result.__dict__,
            "audit": __import__("execution.gemini_live_smoke", fromlist=["build_live_smoke_audit"]).build_live_smoke_audit(result).__dict__,
            "preview": build_live_smoke_operator_preview(result),
        },
        default=str,
    )
    assert GEMINI_SECRET not in serialized
    assert '"token_map":' not in serialized
    for raw_value in RAW_FIXTURE_VALUES:
        assert raw_value not in serialized
    assert check_public_report_payload(json.loads(json.dumps(result.__dict__, default=str))).passed


# 19-21. No active write, no auto-accept, review-bound.
def test_no_active_write_no_auto_accept_review_bound() -> None:
    fake = FakeGeminiLiveClient()
    result = run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=fake)
    assert result.active_written_count == 0
    assert result.auto_accept is False
    assert result.review_required is True


# 25-28. Live mode with injectable fake client: one call, schema validated, draft.
def test_live_mode_fake_client_one_call_and_draft() -> None:
    fake = FakeGeminiLiveClient()
    result = run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=fake)
    assert result.gemini_real_call_attempted is True
    assert fake.call_count == 1
    assert result.provider_response_received is True
    assert result.schema_valid is True
    assert result.live_call_status == "succeeded"
    assert result.review_bound_package_count >= 1
    assert result.provider_client_kind == "fake"
    # fake client is not network
    assert result.external_api_used is False
    assert result.real_network_call_used is False


# 29-31. Other providers' real-call flags false; no local model / subprocess.
def test_other_provider_flags_false() -> None:
    fake = FakeGeminiLiveClient()
    result = run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=fake)
    assert result.claude_real_call_attempted is False
    assert result.openai_real_call_attempted is False
    assert result.ollama_real_call_attempted is False
    assert result.local_model_call_used is False
    assert result.subprocess_call_used is False


# 32-33. Operator preview: blocked in default; one-call only when fake live ran.
def test_operator_preview_blocked_default_and_one_call_when_live() -> None:
    blocked = build_live_smoke_operator_preview(run_gemini_live_smoke(environ={}))
    assert blocked["no_external_call_notice"] == "No external AI call was made"
    assert blocked["blocked_notice"] == "Live smoke blocked until explicit operator approval"
    assert blocked["one_call_notice"] == ""
    live = build_live_smoke_operator_preview(run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=FakeGeminiLiveClient()))
    assert live["one_call_notice"] == "One Gemini live smoke call was made"
    assert live["effective_provider"] == "gemini"


# Synthetic payload fixture is obviously synthetic and identifier-free.
def test_synthetic_payload_is_safe() -> None:
    check = synthetic_payload_check()
    assert check["is_obviously_synthetic"] is True
    assert check["contains_no_identifier_markers"] is True
    assert check["is_cost_bounded_short"] is True
    assert check["payload_class"] == SYNTHETIC_PAYLOAD_CLASS


# No provider SDK hard import / no raw HTTP in the live-smoke source.
def test_no_hard_sdk_or_http_in_source() -> None:
    source = (REPO_ROOT / "execution/gemini_live_smoke.py").read_text(encoding="utf-8")
    forbidden = [
        "import " + "google.generativeai",
        "from " + "google.generativeai",
        "google.generativeai",  # contiguous literal must be absent (assembled dynamically)
        "requests" + ".get",
        "requests" + ".post",
        "urllib" + ".request",
        "httpx" + ".",
        "import " + "subprocess",
    ]
    assert not any(item in source for item in forbidden)


# 34. 15L flat harness used; no recursive prior-block scripts.
def test_flat_harness_used_no_recursive_scripts() -> None:
    run = _flat_run_prior()
    assert run["nested_prior_block_scripts_invoked"] is False
    assert run["recursive_tests_deselected"] is True


# 35. Doctrine phrases remain present.
def test_doctrine_phrases_present() -> None:
    assert DOCTRINE.exists()
    text = DOCTRINE.read_text(encoding="utf-8")
    for phrase in DOCTRINE_PHRASES:
        assert phrase in text


# 16-18 (reports). 15M reports public-safe.
def test_15m_reports_are_public_safe(monkeypatch) -> None:
    monkeypatch.setenv(GEMINI_CREDENTIAL_ENV_VAR, GEMINI_SECRET)
    env = dict(os.environ)
    env["MEDAI_15M_SKIP_FLAT"] = "1"  # skip the heavy flat suite for the report test
    # ensure no real live gates so the script stays in blocked/no-call mode
    env.pop(ALLOW_SMOKE_ENV, None)
    env.pop(OPERATOR_APPROVED_ENV, None)
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_gemini_live_smoke_gated_15m.py"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report_dir = REPO_ROOT / "reports" / "medai_ai_gemini_live_smoke_gated_15m"
    for path in report_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert GEMINI_SECRET not in text
        assert '"token_map":' not in text
        assert "C:\\" not in text
        assert "sk-" not in text
        for raw_value in RAW_FIXTURE_VALUES:
            assert raw_value not in text
        payload = json.loads(text) if path.suffix == ".json" else text
        check = check_public_report_payload(payload)
        assert check.passed, f"{path.name}: {check.leak_examples_redacted}"


# 36-46. Each prior block passes under the flat harness.
@pytest.mark.parametrize("test_file", PRIOR_FOCUSED_FILES)
def test_prior_block_passes_under_flat_harness(test_file) -> None:
    run = _flat_run_prior()
    assert run["flat_suite"]["status"] == "passed", run["flat_suite"].get("stdout_tail", "")
    assert int(run["flat_suite"]["pytest_counts"].get("failed", 0)) == 0
    assert test_file not in run["missing_test_files"]


# 47-48. 12A and 13C pass.
def test_12a_and_13c_pass() -> None:
    run = _flat_run_prior()
    direct = {item["name"]: item["status"] for item in run["direct_commands"]}
    assert direct["regression_12a_uat"] == "passed"
    assert direct["regression_13c_pytest"] == "passed"
