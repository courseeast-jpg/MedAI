"""Focused tests for MEDAI-AI-GEMINI-LIVE-SMOKE-SDK-DIAGNOSTIC-REPAIR-15N-R1.

No live call, no network. Provider failures are simulated with injectable fake
clients that raise; the repaired runner classifies + sanitizes the error.
"""
from __future__ import annotations

import functools
import json
import sys
import types
from pathlib import Path

import pytest

from clinical_knowledge.privacy import check_public_report_payload
import execution.gemini_live_smoke as gls
from execution.gemini_live_smoke import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_MODEL_ENV,
    SYNTHETIC_REDACTED_SMOKE_SUMMARY,
    GeminiLiveSmokeAuditRecord,
    GeminiLiveSmokeResult,
    build_live_smoke_audit,
    classify_provider_error,
    detect_gemini_sdk,
    resolve_gemini_model,
    run_gemini_live_smoke,
    sanitize_provider_error_message,
    _acquire_real_gemini_client,
)
from scripts.medai_ai_flat_validation_harness import (
    DEFAULT_DESELECT_PATTERNS,
    FlatHarnessConfig,
    run_flat_validation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTRINE = REPO_ROOT / "docs/architecture/MEDAI_CAPABILITY_BOUNDARY_DOCTRINE.md"
FAKE_KEY = "AIzaFAKEKEY1234567890abcdef"
LIVE_ENV = {
    "MEDAI_ALLOW_REAL_PROVIDER_SMOKE": "1",
    "MEDAI_OPERATOR_APPROVED_LIVE_SMOKE": "1",
    "GEMINI_API_KEY": FAKE_KEY,
}
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
    "tests/test_medai_ai_gemini_live_smoke_gated_15m.py",
]
DIRECT_COMMANDS = [
    ("regression_12a_uat", [sys.executable, "scripts/run_medai_operator_workflow_uat_12a.py"]),
    ("regression_13c_pytest", [sys.executable, "-m", "pytest", "tests/test_medai_operator_usability_polish_13c.py", "-p", "no:cacheprovider", "-q"]),
]


# Fake provider exception classes (names match google.api_core.exceptions).
class InvalidArgument(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.status_code = 400
        self.code = "INVALID_ARGUMENT"


class PermissionDenied(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.status_code = 403


class NotFound(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.status_code = 404


class ResourceExhausted(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.status_code = 429


class DeadlineExceeded(Exception):
    pass


class _FailClient:
    def __init__(self, exc):
        self.exc = exc
        self.call_count = 0

    def generate_extraction(self, *, prompt: str = "", **_kwargs):
        self.call_count += 1
        raise self.exc


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


# 1. Provider exception capture fields exist.
def test_capture_fields_exist() -> None:
    for fname in (
        "provider_error_type",
        "provider_error_message_sanitized",
        "provider_error_status_code",
        "provider_error_code",
        "provider_error_category",
        "provider_error_source",
        "provider_error_retryable",
        "provider_error_redaction_applied",
        "provider_error_message_truncated",
        "provider_error_available",
        "sdk_package_detected",
        "sdk_package_version",
        "sdk_call_style",
        "gemini_model_name",
        "sdk_call_kwargs_keys",
    ):
        assert fname in GeminiLiveSmokeResult.__dataclass_fields__
        assert fname in GeminiLiveSmokeAuditRecord.__dataclass_fields__


# 2. Broad exception no longer collapses to undiagnosable generic-only result.
def test_failure_is_diagnosable() -> None:
    result = run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=_FailClient(TypeError("boom")))
    assert result.live_call_status == "failed"
    assert result.provider_error_available is True
    assert result.provider_error_category != ""
    assert result.provider_response_received is False


# 3-9. Category classification.
@pytest.mark.parametrize(
    "exc,expected",
    [
        (TypeError("unexpected keyword argument 'request_options'"), "sdk_signature_mismatch"),
        (InvalidArgument("400 invalid request"), "request_config_rejected"),
        (NotFound("404 model not found"), "model_not_found"),
        (PermissionDenied("403 SERVICE_DISABLED not enabled"), "api_disabled_or_permission"),
        (ResourceExhausted("429 quota exceeded"), "quota_or_billing"),
        (DeadlineExceeded("deadline exceeded"), "timeout"),
        (ValueError("totally unknown boom"), "unknown_provider_error"),
    ],
)
def test_error_categories(exc, expected) -> None:
    result = run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=_FailClient(exc))
    assert result.provider_error_category == expected
    assert result.provider_error_available is True
    assert result.gemini_real_call_attempted is True
    assert result.provider_response_received is False


# 10-11. API-key-like strings + GEMINI_API_KEY value redacted.
def test_api_key_redacted() -> None:
    msg = f"auth failed for key={FAKE_KEY} and AIzaOTHERKEY0987654321zzzz"
    result = run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=_FailClient(RuntimeError(msg)))
    sanitized = result.provider_error_message_sanitized
    assert FAKE_KEY not in sanitized
    assert "AIzaOTHERKEY" not in sanitized
    assert "<redacted-credential>" in sanitized
    assert result.provider_error_redaction_applied is True


# 12. Prompt/payload fragments redacted.
def test_payload_fragment_redacted() -> None:
    msg = f"rejected content: {SYNTHETIC_REDACTED_SMOKE_SUMMARY}"
    result = run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=_FailClient(RuntimeError(msg)))
    assert SYNTHETIC_REDACTED_SMOKE_SUMMARY not in result.provider_error_message_sanitized
    assert "<redacted-payload>" in result.provider_error_message_sanitized


# 13. Windows paths redacted.
def test_windows_path_redacted() -> None:
    msg = r"file error at C:\Users\S1\.codex\secret_config.py line 42"
    result = run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=_FailClient(RuntimeError(msg)))
    assert "C:\\" not in result.provider_error_message_sanitized
    assert "<redacted-path>" in result.provider_error_message_sanitized


# 14. Error message truncated safely.
def test_message_truncated() -> None:
    sanitized, _redacted, truncated = sanitize_provider_error_message("x" * 5000)
    assert truncated is True
    assert len(sanitized) <= 320


# 15-16. status_code / code extracted when available.
def test_status_code_and_code_extracted() -> None:
    result = run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=_FailClient(InvalidArgument("bad")))
    assert result.provider_error_status_code == "400"
    assert result.provider_error_code in {"INVALID_ARGUMENT", "400"}


# 17-19. SDK detection + kwargs keys (names only).
def test_sdk_detection_safe() -> None:
    sdk = detect_gemini_sdk()
    assert sdk["sdk_call_style"] in {"google_genai", "legacy_generativeai", "none"}
    assert isinstance(sdk["sdk_package_version"], str)
    result = run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=_FailClient(TypeError("x")))
    assert all(isinstance(k, str) for k in result.sdk_call_kwargs_keys)
    # kwargs keys are names only; never values
    assert FAKE_KEY not in json.dumps(result.sdk_call_kwargs_keys)


# 20-22. Model configurable, override, text model.
def test_model_configurable_and_text_default() -> None:
    assert resolve_gemini_model({}) == DEFAULT_GEMINI_MODEL
    assert resolve_gemini_model({GEMINI_MODEL_ENV: "gemini-2.5-flash"}) == "gemini-2.5-flash"
    lowered = DEFAULT_GEMINI_MODEL.lower()
    for banned in ("vision", "image", "imagen", "tts", "audio", "veo"):
        assert banned not in lowered


# 23. google.genai support path constructed with import shim (no network).
def test_genai_path_constructible_with_shim(monkeypatch) -> None:
    fake_response = types.SimpleNamespace(text=json.dumps(gls.SYNTHETIC_SMOKE_RESPONSE))

    class _Models:
        def generate_content(self, *, model, contents, config):
            return fake_response

    class _Client:
        def __init__(self, *, api_key):
            assert api_key  # value used only to construct; never logged
            self.models = _Models()

    fake_mod = types.SimpleNamespace(Client=_Client)
    monkeypatch.setitem(sys.modules, "google.genai", fake_mod)
    monkeypatch.setattr(gls, "detect_gemini_sdk", lambda: {
        "google_genai_present": True, "legacy_generativeai_present": False,
        "sdk_call_style": "google_genai", "sdk_package_detected": "google-genai", "sdk_package_version": "x",
    })
    client = _acquire_real_gemini_client(environ=LIVE_ENV, policy=gls.GeminiLiveSmokePolicy())
    assert client is not None
    assert client.call_kwargs_keys == ["model", "contents", "config"]
    parsed = client.generate_extraction(prompt="p")
    assert parsed["document_type"]  # parsed JSON, no network


# 24. legacy google.generativeai support path constructed with import shim (no network).
def test_legacy_path_constructible_with_shim(monkeypatch) -> None:
    class _Model:
        def __init__(self, name):
            self.name = name

        def generate_content(self, prompt, generation_config=None):
            return types.SimpleNamespace(text=json.dumps(gls.SYNTHETIC_SMOKE_RESPONSE))

    fake_mod = types.SimpleNamespace(
        configure=lambda **k: None,
        GenerativeModel=_Model,
    )
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_mod)
    monkeypatch.setattr(gls, "detect_gemini_sdk", lambda: {
        "google_genai_present": False, "legacy_generativeai_present": True,
        "sdk_call_style": "legacy_generativeai", "sdk_package_detected": "google-generativeai", "sdk_package_version": "0.8.6",
    })
    client = _acquire_real_gemini_client(environ=LIVE_ENV, policy=gls.GeminiLiveSmokePolicy())
    assert client is not None
    assert client.call_kwargs_keys == ["contents", "generation_config"]


# 25. No hard SDK dependency in default tests (module imports without any SDK).
def test_no_hard_sdk_dependency() -> None:
    source = (REPO_ROOT / "execution/gemini_live_smoke.py").read_text(encoding="utf-8")
    for forbidden in ("import " + "google.generativeai", "from " + "google.generativeai", "google.generativeai", "import " + "google.genai"):
        assert forbidden not in source


# 26-32. No live/network/subprocess; invariants on every failure.
@pytest.mark.parametrize("exc", [TypeError("x"), InvalidArgument("y"), DeadlineExceeded("z"), ValueError("w")])
def test_failure_invariants(exc) -> None:
    client = _FailClient(exc)
    result = run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=client)
    assert client.call_count == 1
    assert result.provider_response_received is False
    assert result.schema_valid is False
    assert result.review_bound_package_count == 0
    assert result.active_written_count == 0
    assert result.auto_accept is False
    assert result.review_required is True
    assert result.local_model_call_used is False
    assert result.subprocess_call_used is False
    assert result.claude_real_call_attempted is False
    assert result.openai_real_call_attempted is False
    assert result.ollama_real_call_attempted is False


# 33-35. No token map / PII / credential in result+audit+sanitized output.
def test_no_leak_in_outputs() -> None:
    payload_secret = SYNTHETIC_REDACTED_SMOKE_SUMMARY
    msg = f"err {FAKE_KEY} {payload_secret} C:\\Users\\S1\\x.py"
    result = run_gemini_live_smoke(environ=LIVE_ENV, gemini_client=_FailClient(RuntimeError(msg)))
    audit = build_live_smoke_audit(result)
    blob = json.dumps({"r": result.__dict__, "a": audit.__dict__}, default=str)
    assert FAKE_KEY not in blob
    assert '"token_map":' not in blob
    assert "C:\\Users" not in blob
    assert "Jane Example" not in blob
    assert check_public_report_payload(json.loads(json.dumps(result.__dict__, default=str))).passed


# 36. Failed live-run summary preserved safely by the repair script.
def test_failed_live_run_summary_preserved() -> None:
    from scripts.run_medai_ai_gemini_live_smoke_sdk_diagnostic_repair_15n_r1 import (
        FAILED_LIVE_RUN_SUMMARY,
    )

    s = FAILED_LIVE_RUN_SUMMARY
    assert s["external_api_used"] is True
    assert s["real_network_call_used"] is True
    assert s["gemini_real_call_attempted"] is True
    assert s["provider_response_received"] is False
    assert s["schema_validation_ran"] is False
    assert s["review_bound_package_count"] == 0
    assert s["active_written_count"] == 0
    assert s["auto_accept"] is False
    assert s["additional_live_call_made_during_15n_r1"] is False


# 37. Doctrine phrases present.
def test_doctrine_phrases_present() -> None:
    assert DOCTRINE.exists()
    text = DOCTRINE.read_text(encoding="utf-8")
    for phrase in DOCTRINE_PHRASES:
        assert phrase in text


# 38-50. Prior blocks + 12A + 13C pass under the flat harness.
@pytest.mark.parametrize("test_file", PRIOR_FOCUSED_FILES)
def test_prior_block_passes_under_flat_harness(test_file) -> None:
    run = _flat_run_prior()
    assert run["flat_suite"]["status"] == "passed", run["flat_suite"].get("stdout_tail", "")
    assert int(run["flat_suite"]["pytest_counts"].get("failed", 0)) == 0
    assert test_file not in run["missing_test_files"]


def test_12a_and_13c_pass_under_flat_harness() -> None:
    run = _flat_run_prior()
    direct = {item["name"]: item["status"] for item in run["direct_commands"]}
    assert direct["regression_12a_uat"] == "passed"
    assert direct["regression_13c_pytest"] == "passed"
    assert run["nested_prior_block_scripts_invoked"] is False
