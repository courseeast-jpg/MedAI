"""Focused tests for MEDAI-AI-PROVIDER-ENABLEMENT-OPERATOR-CONTROL-15K."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_extraction_adapter import ExtractionWorkflowContext
from execution.extraction_workflow import run_ai_extraction_workflow
from execution.ai_provider_operator_control import (
    CONTROLLED_PROVIDERS,
    AIProviderOperatorAuditRecord,
    AIProviderOperatorControlState,
    AIProviderOperatorEnablementDecision,
    AIProviderOperatorEnablementRequest,
    AIProviderOperatorReadinessSummary,
    build_provider_operator_control,
    evaluate_operator_enablement_request,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTRINE = REPO_ROOT / "docs/architecture/MEDAI_CAPABILITY_BOUNDARY_DOCTRINE.md"
PRIVATE_TEXT = (
    "Patient Jane Example; DOB 01/02/1970; MRN 123456; Accession CY-2026-0001; "
    "Facility Park Medical Center; Provider Dr. Alice Clinician; "
    "123 Main Street, Springfield, NY 10001; 555-123-4567; jane.example@example.com; "
    "Insurance INS-ABC-12345; Collected 06/10/2026"
)
CLAUDE_SECRET = "claude-15k-secret-value"
RAW_FIXTURE_VALUES = [
    "Jane Example",
    "01/02/1970",
    "123456",
    "CY-2026-0001",
    "Park Medical Center",
    "Alice Clinician",
    "123 Main Street",
    "555-123-4567",
    "jane.example@example.com",
    "INS-ABC-12345",
    "06/10/2026",
]
DOCTRINE_PHRASES = [
    "Do not assign semantic/layout understanding to OCR or deterministic rules.",
    (
        "A MedAI extraction block is not successful because records were created. It is "
        "successful only if the operator can compare a source-faithful package against the "
        "original document quickly and safely."
    ),
]

PASSING_GATES = dict(
    privacy_gate_result={"privacy_gate_status": "redacted_payload_ready"},
    payload_policy_result={"payload_policy_allowed": True},
    budget_guard_result={"budget_allowed": True},
    dry_run_decision_result={"dry_run_external_call_allowed": True},
)


def _stage(provider="claude", credential_present=True, is_cloud=True, **override):
    gates = {**PASSING_GATES, **override}
    return evaluate_operator_enablement_request(
        provider_name=provider,
        requested_state="staged",
        credential_present=credential_present,
        is_cloud_provider=is_cloud,
        **gates,
    )


# 1. Operator-control classes exist.
def test_operator_control_classes_exist() -> None:
    assert AIProviderOperatorControlState is not None
    assert AIProviderOperatorEnablementRequest is not None
    assert AIProviderOperatorEnablementDecision is not None
    assert AIProviderOperatorReadinessSummary is not None
    assert AIProviderOperatorAuditRecord is not None


def _control(**override):
    gates = {**PASSING_GATES, **override}
    return build_provider_operator_control(
        requested_provider=override.pop("requested_provider", "claude") if "requested_provider" in override else "claude",
        selected_provider="claude",
        effective_provider="fake_local",
        enablement_request=override.pop("enablement_request", None) if "enablement_request" in override else None,
        **{k: v for k, v in gates.items() if k in {"privacy_gate_result", "payload_policy_result", "budget_guard_result", "dry_run_decision_result"}},
    )


# 2. All providers appear in readiness summary.
def test_all_providers_in_readiness_summary() -> None:
    state = _control()
    names = [row["provider_name"] for row in state.providers]
    for provider in CONTROLLED_PROVIDERS:
        assert provider in names


# 3-4. fake_local enabled; real providers disabled by policy.
def test_fake_local_enabled_real_providers_disabled() -> None:
    state = _control()
    rows = {row["provider_name"]: row for row in state.providers}
    assert rows["fake_local"]["provider_enabled_by_policy"] is True
    for provider in ("gemini", "claude", "openai", "local_ollama"):
        assert rows[provider]["provider_enabled_by_policy"] is False
        assert rows[provider]["execution_allowed"] is False
        assert rows[provider]["real_call_attempted"] is False


# 5-6. Staged request does not enable execution / final external call.
def test_staged_request_does_not_enable_execution() -> None:
    decision = _stage()
    assert decision.operator_enablement_request_state == "staged"
    assert decision.operator_enablement_request_allowed is True
    assert decision.real_provider_execution_enabled is False
    assert decision.final_external_call_allowed is False
    assert decision.staged_only is True
    # Also through the workflow control state
    state = build_provider_operator_control(
        requested_provider="claude",
        selected_provider="claude",
        effective_provider="fake_local",
        enablement_request={"provider": "claude", "requested_state": "staged"},
        **PASSING_GATES,
    )
    assert state.real_provider_execution_enabled is False
    assert state.final_external_call_allowed is False


# 7. Missing privacy gate blocks staging.
def test_missing_privacy_gate_blocks_staging() -> None:
    decision = _stage(privacy_gate_result={})
    assert decision.operator_enablement_request_allowed is False
    assert decision.operator_enablement_request_block_reason == "missing_privacy_gate"


# 8. Failed payload policy blocks staging.
def test_failed_payload_policy_blocks_staging() -> None:
    decision = _stage(payload_policy_result={"payload_policy_allowed": False})
    assert decision.operator_enablement_request_allowed is False
    assert decision.operator_enablement_request_block_reason == "payload_policy_failed"


# 9. Budget failure blocks staging.
def test_budget_failure_blocks_staging() -> None:
    decision = _stage(budget_guard_result={"budget_allowed": False})
    assert decision.operator_enablement_request_allowed is False
    assert decision.operator_enablement_request_block_reason == "budget_exceeded"


# 10. Missing dry-run blocks staging.
def test_missing_dry_run_blocks_staging() -> None:
    decision = _stage(dry_run_decision_result={"dry_run_external_call_allowed": False})
    assert decision.operator_enablement_request_allowed is False
    assert decision.operator_enablement_request_block_reason == "dry_run_required"


# 11. Missing credential blocks cloud-provider staging.
def test_missing_credential_blocks_cloud_staging() -> None:
    decision = _stage(provider="openai", credential_present=False, is_cloud=True)
    assert decision.operator_enablement_request_allowed is False
    assert decision.operator_enablement_request_block_reason == "credential_missing"
    # local_ollama is not cloud: gates pass without credential
    local = _stage(provider="local_ollama", credential_present=False, is_cloud=False)
    assert local.operator_enablement_request_allowed is True
    assert local.real_provider_execution_enabled is False


# 12. Credential values never appear in control output.
def test_credential_values_never_appear(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", CLAUDE_SECRET)
    state = build_provider_operator_control(
        requested_provider="claude",
        selected_provider="claude",
        effective_provider="fake_local",
        **PASSING_GATES,
    )
    serialized = json.dumps(state.providers, sort_keys=True)
    assert CLAUDE_SECRET not in serialized
    claude_row = next(r for r in state.providers if r["provider_name"] == "claude")
    assert claude_row["credential_present"] is True
    assert claude_row["credential_env_var_name"] == "ANTHROPIC_API_KEY"


# 13-16. local_ollama readiness: no localhost, no subprocess; no SDK; no network.
def test_local_ollama_readiness_no_runtime_call() -> None:
    state = _control(requested_provider="local_ollama")
    row = next(r for r in state.providers if r["provider_name"] == "local_ollama")
    assert row["local_model_call_used"] is False
    assert row["subprocess_call_used"] is False
    assert row["credential_required"] is False
    assert row["real_call_attempted"] is False
    for module in ("ollama", "anthropic", "openai", "google.generativeai"):
        assert module not in sys.modules
    # operator-control source has no network/subprocess/localhost call code
    source = (REPO_ROOT / "execution/ai_provider_operator_control.py").read_text(encoding="utf-8")
    for marker in (
        "import " + "subprocess",
        "subprocess" + ".run",
        "import " + "socket",
        "import " + "requests",
        "requests" + ".",
        "urllib" + ".request",
        "httpx" + ".",
        "urlopen",
        "http" + ".client",
    ):
        assert marker not in source, marker


# 17-27. Workflow invariants across all providers.
def test_workflow_invariants_all_flags_false() -> None:
    result = run_ai_extraction_workflow(_context())
    control = result.operator_control_result
    assert control["external_api_used"] is False
    assert control["real_network_call_used"] is False
    assert control["local_model_call_used"] is False
    assert control["subprocess_call_used"] is False
    assert control["final_external_call_allowed"] is False
    assert control["real_provider_execution_enabled"] is False
    assert control["active_written_count"] == 0
    assert control["auto_accept"] is False
    assert control["review_required"] is True
    for row in control["providers"]:
        assert row["real_call_attempted"] is False
        assert row["real_provider_execution_enabled"] is False


def test_raw_payloads_remain_forbidden() -> None:
    from execution.premium_adapter_contract import FORBIDDEN_PAYLOAD_TYPES
    from execution.claude_extraction_adapter import ClaudeAdapterRequest, ClaudeExtractionAdapter

    for payload_type in FORBIDDEN_PAYLOAD_TYPES:
        result = ClaudeExtractionAdapter().evaluate(
            ClaudeAdapterRequest(
                payload_type=payload_type,
                redacted_payload_hash="abc",
                privacy_gate_status="redacted_payload_ready",
                payload_policy_allowed=True,
                budget_allowed=True,
                real_provider_execution_enabled=True,
                final_external_call_allowed=True,
            )
        )
        assert result.fail_closed_reason == "forbidden_payload_type"


# 28-31. fake_local path still produces visible review-bound packages.
def test_fake_local_path_review_bound_preserved() -> None:
    result = run_ai_extraction_workflow(_context(provider_name="fake_local", provider_mode="fake_local"))
    assert result.review_bound_package_count == 1
    assert result.active_written_count == 0
    assert result.auto_accept is False
    assert result.review_required is True
    assert result.operator_preview["visible"] is True


# 32-33. Workflow preview: no token map / PII.
def test_workflow_preview_has_no_pii_or_token_map(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", CLAUDE_SECRET)
    preview = run_ai_extraction_workflow(_context()).operator_preview
    serialized = json.dumps(preview, sort_keys=True)
    assert '"token_map":' not in serialized
    assert CLAUDE_SECRET not in serialized
    for raw_value in RAW_FIXTURE_VALUES:
        assert raw_value not in serialized
    assert check_public_report_payload(preview).passed


# 34-36. Operator preview notices.
def test_operator_preview_notices() -> None:
    preview = run_ai_extraction_workflow(_context()).operator_preview
    assert preview["operator_control_no_external_call_notice"] == "No external AI call was made"
    assert preview["operator_control_no_local_model_call_notice"] == "No local model call was made"
    assert preview["operator_control_staged_request_notice"] == "Staged request does not enable execution"
    assert preview["operator_control_real_provider_execution_enabled"] is False


# 37. Doctrine file exists and required exact phrases remain present.
def test_capability_boundary_doctrine_phrases_present() -> None:
    assert DOCTRINE.exists()
    text = DOCTRINE.read_text(encoding="utf-8")
    for phrase in DOCTRINE_PHRASES:
        assert phrase in text


# 32-33 (reports). 15K reports are public-safe.
def test_15k_reports_are_public_safe(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", CLAUDE_SECRET)
    env = dict(os.environ)
    env["MEDAI_15K_SKIP_PYTEST"] = "1"
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_provider_enablement_operator_control_15k.py"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    report_dir = REPO_ROOT / "reports" / "medai_ai_provider_enablement_operator_control_15k"
    for path in report_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert CLAUDE_SECRET not in text
        assert '"token_map":' not in text
        assert "C:\\" not in text
        assert "sk-" not in text
        for raw_value in RAW_FIXTURE_VALUES:
            assert raw_value not in text
        payload = json.loads(text) if path.suffix == ".json" else text
        check = check_public_report_payload(payload)
        assert check.passed, f"{path.name}: {check.leak_examples_redacted}"


# 38-48. Prior-block regressions still pass.
def test_prior_block_regressions_still_pass() -> None:
    commands = [
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_local_ollama_adapter_disabled_15j.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_premium_adapters_disabled_15i.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_gemini_adapter_disabled_15g.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_real_provider_enablement_gate_15f.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_external_call_dry_run_15e.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_provider_selection_ui_15d.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_provider_adapter_stub_15c.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_privacy_gate_15b.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_workflow_seam_15a.py"],
        [sys.executable, "scripts/run_medai_operator_workflow_uat_12a.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_operator_usability_polish_13c.py"],
    ]
    for command in commands:
        proc = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr


def _context(**overrides):
    values = {
        "source_class": "urinalysis_table",
        "safe_source_document_id": "source_fake_15k",
        "selected_document_category": "AI-assisted extraction",
        "selected_specialty_domain": "urology",
        "source_modality": "fake_local_adapter",
        "raw_text_local_only": PRIVATE_TEXT,
        "provider_name": "claude",
        "provider_mode": "disabled",
        "operator_approval_state": "approved_for_dry_run",
        "external_call_mode": "dry_run",
        "real_provider_enablement_mode": "readiness_check",
    }
    values.update(overrides)
    return ExtractionWorkflowContext(**values)
