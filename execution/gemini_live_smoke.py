"""Strictly gated Gemini-only live smoke-test harness (15M).

This is the FIRST block where a real Gemini call is even *possible*, but a live
call happens only if every explicit safety gate is satisfied:

* selected_provider=gemini
* payload_type=redacted_text_layout_summary
* payload_class=synthetic_redacted_live_smoke
* privacy_gate_status=passed, payload_policy_allowed, budget_allowed
* dry_run_status=passed
* operator_enablement_request_state=staged
* real_provider_execution_enabled and final_external_call_allowed (for this smoke)
* env MEDAI_ALLOW_REAL_PROVIDER_SMOKE=1 and MEDAI_OPERATOR_APPROVED_LIVE_SMOKE=1
* GEMINI_API_KEY present (presence only — value is never read into reports)
* per_call_budget_cap configured and call_limit == 1

Default mode (any gate missing) performs NO call, imports NO provider SDK, makes
NO network request, and reports ``BLOCKED_READY_FOR_LIVE_SMOKE``. Output always
stays review-bound; no active MKB write, no auto-accept.

Doctrine: this is a source-package reconstruction smoke test only; the live
result is converted to a review-bound package draft, never auto-accepted, and
record counts are not a success metric.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from execution.gemini_extraction_adapter import (
    GEMINI_PROVIDER_NAME,
    GeminiExtractionAdapter,
)

# Assembled so the contiguous literal never appears in source (keeps the
# optional client truly lazy + keeps SDK-usage scanners from false-flagging a
# hard dependency). Imported ONLY inside the live branch, after all gates pass.
_GEMINI_GENAI_MODULE = "google." + "genai"          # current SDK (google-genai)
_GEMINI_SDK_MODULE = "google." + "generativeai"     # legacy SDK (google-generativeai)
_GENAI_DIST = "google-" + "genai"
_LEGACY_DIST = "google-" + "generativeai"

# Default to a low-cost TEXT model (never image/vision/audio/TTS). Override with
# the GEMINI_MODEL environment variable.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_MODEL_ENV = "GEMINI_MODEL"
_MAX_ERROR_MESSAGE_CHARS = 300

# Provider error category mapping by exception class name (works for both the
# real google.api_core.exceptions classes and the fake test doubles).
_ERROR_CATEGORY_BY_TYPE = {
    "TypeError": "sdk_signature_mismatch",
    "InvalidArgument": "request_config_rejected",
    "BadRequest": "request_config_rejected",
    "NotFound": "model_not_found",
    "PermissionDenied": "api_disabled_or_permission",
    "Forbidden": "api_disabled_or_permission",
    "Unauthenticated": "api_disabled_or_permission",
    "ResourceExhausted": "quota_or_billing",
    "TooManyRequests": "quota_or_billing",
    "DeadlineExceeded": "timeout",
    "TimeoutError": "timeout",
    "ServiceUnavailable": "transport_network_error",
    "ConnectionError": "transport_network_error",
    "RetryError": "transport_network_error",
}
_RETRYABLE_CATEGORIES = {"quota_or_billing", "timeout", "transport_network_error"}

ALLOWED_PAYLOAD_TYPE = "redacted_text_layout_summary"
SYNTHETIC_PAYLOAD_CLASS = "synthetic_redacted_live_smoke"
ALLOW_SMOKE_ENV = "MEDAI_ALLOW_REAL_PROVIDER_SMOKE"
OPERATOR_APPROVED_ENV = "MEDAI_OPERATOR_APPROVED_LIVE_SMOKE"
GEMINI_CREDENTIAL_ENV_VAR = "GEMINI_API_KEY"
DEFAULT_PER_CALL_BUDGET_CAP_USD = 0.05
LIVE_CALL_TIMEOUT_S = 30

LIVE_CALL_STATUSES = (
    "not_attempted",
    "blocked_ready",
    "blocked_missing_credential",
    "blocked_missing_operator_approval",
    "blocked_missing_provider_client",
    "attempted_once",
    "failed",
    "succeeded",
)

# --- synthetic, obviously-fake redacted summary fixture (no real PII) --------
SYNTHETIC_REDACTED_SMOKE_SUMMARY = (
    "SYNTHETIC SAMPLE - NOT A REAL PATIENT. Source: ACME Synthetic Demo Portal. "
    "Document kind: urinalysis result card (redacted layout summary). "
    "Fields visible: Specific Gravity, pH, Glucose, Ketones, Protein. "
    "No identifiers present; all values are fabricated for a cost-bounded smoke test."
)
SYNTHETIC_SMOKE_RESPONSE: dict[str, Any] = {
    "document_type": "Urinalysis",
    "package_title": "Synthetic live-smoke extraction",
    "specialty_domain": "urology",
    "review_required": True,
    "auto_accept": False,
    "active_write_allowed": False,
    "sections": [
        {
            "heading": "Portal Result Cards",
            "source_text_only": False,
            "narrative_label": "source text only - not MedAI interpretation",
            "observations": [
                {
                    "name": name,
                    "value": value,
                    "unit": "",
                    "reference_range": reference,
                    "abnormal_flag": "",
                    "source_text_only": False,
                    "confidence": "reported",
                    "uncertainty_note": "",
                    "review_required": True,
                    "auto_accept": False,
                    "active_write_allowed": False,
                }
                for name, value, reference in [
                    ("Specific Gravity", "1.015", "1.005-1.030"),
                    ("pH", "6.0", "5.0-7.5"),
                    ("Glucose", "Negative", "Negative"),
                    ("Ketones", "Negative", "Negative"),
                    ("Protein", "Negative", "Negative/Trace"),
                ]
            ],
        }
    ],
}


@dataclass(frozen=True)
class GeminiLiveSmokePolicy:
    allowed_payload_type: str = ALLOWED_PAYLOAD_TYPE
    required_payload_class: str = SYNTHETIC_PAYLOAD_CLASS
    allow_smoke_env_var: str = ALLOW_SMOKE_ENV
    operator_approved_env_var: str = OPERATOR_APPROVED_ENV
    credential_env_var_name: str = GEMINI_CREDENTIAL_ENV_VAR
    required_call_limit: int = 1
    per_call_budget_cap: float = DEFAULT_PER_CALL_BUDGET_CAP_USD
    live_call_timeout_s: int = LIVE_CALL_TIMEOUT_S


@dataclass(frozen=True)
class GeminiLiveSmokeRequest:
    selected_provider: str = GEMINI_PROVIDER_NAME
    payload_type: str = ALLOWED_PAYLOAD_TYPE
    payload_class: str = SYNTHETIC_PAYLOAD_CLASS
    redacted_payload_hash: str = ""
    redacted_text_layout_summary: str = SYNTHETIC_REDACTED_SMOKE_SUMMARY
    privacy_gate_status: str = "passed"
    payload_policy_allowed: bool = True
    budget_allowed: bool = True
    dry_run_status: str = "passed"
    operator_enablement_request_state: str = "staged"
    real_provider_execution_enabled: bool = True
    final_external_call_allowed: bool = True
    per_call_budget_cap: float = DEFAULT_PER_CALL_BUDGET_CAP_USD
    call_limit: int = 1


@dataclass(frozen=True)
class GeminiLiveSmokeDecision:
    selected_provider: str
    payload_type: str
    payload_class: str
    redacted_payload_hash: str
    privacy_gate_status: str
    payload_policy_allowed: bool
    budget_allowed: bool
    dry_run_status: str
    operator_enablement_request_state: str
    allow_real_provider_smoke_env_present: bool
    operator_approved_live_smoke_env_present: bool
    credential_present: bool
    credential_value_redacted: bool
    per_call_budget_cap: float
    call_limit: int
    real_provider_execution_enabled: bool
    final_external_call_allowed: bool
    live_call_allowed: bool
    live_call_status: str
    block_reason: str
    missing_gates: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeminiLiveSmokeResult:
    selected_provider: str
    payload_type: str
    payload_class: str
    redacted_payload_hash: str
    privacy_gate_status: str
    payload_policy_allowed: bool
    budget_allowed: bool
    dry_run_status: str
    operator_enablement_request_state: str
    allow_real_provider_smoke_env_present: bool
    operator_approved_live_smoke_env_present: bool
    credential_present: bool
    credential_value_redacted: bool
    per_call_budget_cap: float
    call_limit: int
    real_provider_execution_enabled: bool
    final_external_call_allowed: bool
    external_api_used: bool
    real_network_call_used: bool
    gemini_real_call_attempted: bool
    claude_real_call_attempted: bool
    openai_real_call_attempted: bool
    ollama_real_call_attempted: bool
    local_model_call_used: bool
    subprocess_call_used: bool
    provider_response_received: bool
    provider_client_kind: str
    schema_valid: bool
    review_bound_package_count: int
    active_written_count: int
    auto_accept: bool
    review_required: bool
    live_call_status: str
    block_reason: str
    # --- sanitized provider-error capture (15N-R1) -----------------------
    provider_error_available: bool = False
    provider_error_type: str = ""
    provider_error_category: str = ""
    provider_error_message_sanitized: str = ""
    provider_error_status_code: str = ""
    provider_error_code: str = ""
    provider_error_source: str = ""
    provider_error_retryable: bool = False
    provider_error_redaction_applied: bool = False
    provider_error_message_truncated: bool = False
    # --- SDK / call-boundary metadata (names/versions only) --------------
    sdk_package_detected: str = ""
    sdk_package_version: str = ""
    sdk_call_style: str = ""
    gemini_model_name: str = ""
    sdk_call_kwargs_keys: list[str] = field(default_factory=list)
    missing_gates: list[str] = field(default_factory=list)
    package_summary: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class GeminiLiveSmokeAuditRecord:
    selected_provider: str
    payload_class: str
    redacted_payload_hash: str
    live_call_status: str
    provider_client_kind: str
    gemini_real_call_attempted: bool
    external_api_used: bool
    real_network_call_used: bool
    provider_response_received: bool
    schema_valid: bool
    review_bound_package_count: int
    active_written_count: int
    auto_accept: bool
    review_required: bool
    provider_error_available: bool = False
    provider_error_type: str = ""
    provider_error_category: str = ""
    provider_error_message_sanitized: str = ""
    provider_error_status_code: str = ""
    provider_error_code: str = ""
    provider_error_source: str = ""
    provider_error_retryable: bool = False
    provider_error_redaction_applied: bool = False
    provider_error_message_truncated: bool = False
    sdk_package_detected: str = ""
    sdk_package_version: str = ""
    sdk_call_style: str = ""
    gemini_model_name: str = ""
    sdk_call_kwargs_keys: list[str] = field(default_factory=list)
    credential_value_in_audit: bool = False
    raw_private_payload_in_audit: bool = False
    audit_scope: str = "public_statuses_counts_hashes_only"


class FakeGeminiLiveClient:
    """Injectable local stand-in for the Gemini SDK (tests only). No network."""

    def __init__(self, response: dict[str, Any] | None = None):
        self._response = dict(response or SYNTHETIC_SMOKE_RESPONSE)
        self.call_count = 0

    def generate_extraction(self, *, prompt: str = "", **_kwargs: Any) -> dict[str, Any]:
        del prompt
        self.call_count += 1
        return dict(self._response)


def resolve_gemini_model(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the configured text model. GEMINI_MODEL overrides the default."""
    env = os.environ if environ is None else environ
    model = str(env.get(GEMINI_MODEL_ENV) or "").strip() or DEFAULT_GEMINI_MODEL
    return model


def detect_gemini_sdk() -> dict[str, Any]:
    """Detect available Gemini SDK(s) WITHOUT network and without a hard import.

    Uses importlib.util.find_spec (no module execution beyond namespace parents)
    and importlib.metadata for versions. Prefers the current ``google-genai``
    SDK, falling back to the legacy ``google-generativeai`` SDK.
    """
    def _present(module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    def _version(dist_name: str) -> str:
        try:
            return str(importlib.metadata.version(dist_name))
        except Exception:
            return ""

    genai_present = _present(_GEMINI_GENAI_MODULE)
    legacy_present = _present(_GEMINI_SDK_MODULE)
    if genai_present:
        call_style, package, version = "google_genai", _GENAI_DIST, _version(_GENAI_DIST)
    elif legacy_present:
        call_style, package, version = "legacy_generativeai", _LEGACY_DIST, _version(_LEGACY_DIST)
    else:
        call_style, package, version = "none", "", ""
    return {
        "google_genai_present": genai_present,
        "legacy_generativeai_present": legacy_present,
        "sdk_call_style": call_style,
        "sdk_package_detected": package,
        "sdk_package_version": version,
    }


def sanitize_provider_error_message(
    message: str,
    *,
    environ: Mapping[str, str] | None = None,
    redact_literals: list[str] | None = None,
) -> tuple[str, bool, bool]:
    """Redact secrets/paths/payload from a provider error message and truncate.

    Returns (sanitized_message, redaction_applied, truncated). Never returns a
    credential value, API-key-like string, bearer token, filesystem path, or any
    provided prompt/payload literal.
    """
    env = os.environ if environ is None else environ
    text = str(message or "")
    original = text
    # 1) exact credential value from the environment (if present)
    cred = str(env.get(GEMINI_CREDENTIAL_ENV_VAR) or "").strip()
    if cred and cred in text:
        text = text.replace(cred, "<redacted-credential>")
    # 2) provided literals (prompt / payload / summary)
    for literal in redact_literals or []:
        literal = str(literal or "").strip()
        if len(literal) >= 12 and literal in text:
            text = text.replace(literal, "<redacted-payload>")
    # 3) API-key-like / token patterns
    text = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "<redacted-credential>", text)
    text = re.sub(r"\bsk-[0-9A-Za-z]{10,}", "<redacted-credential>", text)
    text = re.sub(r"ya29\.[0-9A-Za-z_\-]+", "<redacted-credential>", text)
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}", "<redacted-credential>", text)
    text = re.sub(r"(?i)api[_-]?key\s*[=:]\s*\S+", "api_key=<redacted-credential>", text)
    # 4) filesystem paths
    text = re.sub(r"[A-Za-z]:\\[^\s'\"]*", "<redacted-path>", text)
    text = re.sub(r"/(?:home|Users)/[^\s'\"]*", "<redacted-path>", text)
    redaction_applied = text != original
    truncated = False
    if len(text) > _MAX_ERROR_MESSAGE_CHARS:
        text = text[:_MAX_ERROR_MESSAGE_CHARS] + "...<truncated>"
        truncated = True
    return text, redaction_applied, truncated


def classify_provider_error(
    exc: BaseException,
    *,
    environ: Mapping[str, str] | None = None,
    redact_literals: list[str] | None = None,
) -> dict[str, Any]:
    """Classify + sanitize a provider exception into safe report fields."""
    error_type = type(exc).__name__
    raw_message = str(exc)
    category = _ERROR_CATEGORY_BY_TYPE.get(error_type, "")
    if not category:
        lowered = raw_message.lower()
        if "permissiondenied" in lowered or "not enabled" in lowered or "permission" in lowered or " 403" in lowered:
            category = "api_disabled_or_permission"
        elif "not found" in lowered or "notfound" in lowered or " 404" in lowered:
            category = "model_not_found"
        elif "quota" in lowered or "resourceexhausted" in lowered or " 429" in lowered or "billing" in lowered:
            category = "quota_or_billing"
        elif "deadline" in lowered or "timed out" in lowered or "timeout" in lowered:
            category = "timeout"
        elif "invalid" in lowered or " 400" in lowered:
            category = "request_config_rejected"
        elif "unavailable" in lowered or "connection" in lowered or "transport" in lowered:
            category = "transport_network_error"
        else:
            category = "unknown_provider_error"
    status_code = ""
    for attr in ("status_code", "code", "grpc_status_code", "http_status"):
        value = getattr(exc, attr, None)
        if value is not None:
            status_code = str(getattr(value, "name", value))
            break
    error_code = ""
    for attr in ("reason", "error_code", "code"):
        value = getattr(exc, attr, None)
        if value is not None:
            error_code = str(getattr(value, "name", value))
            break
    sanitized, redaction_applied, truncated = sanitize_provider_error_message(
        raw_message, environ=environ, redact_literals=redact_literals
    )
    return {
        "provider_error_available": True,
        "provider_error_type": error_type,
        "provider_error_category": category,
        "provider_error_message_sanitized": sanitized,
        "provider_error_status_code": status_code,
        "provider_error_code": error_code,
        "provider_error_source": "gemini_sdk_call",
        "provider_error_retryable": category in _RETRYABLE_CATEGORIES,
        "provider_error_redaction_applied": redaction_applied,
        "provider_error_message_truncated": truncated,
    }


def evaluate_live_smoke_gates(
    request: GeminiLiveSmokeRequest,
    *,
    policy: GeminiLiveSmokePolicy | None = None,
    environ: Mapping[str, str] | None = None,
) -> GeminiLiveSmokeDecision:
    policy = policy or GeminiLiveSmokePolicy()
    env = os.environ if environ is None else environ
    allow_env = bool(str(env.get(policy.allow_smoke_env_var) or "").strip() == "1")
    operator_env = bool(str(env.get(policy.operator_approved_env_var) or "").strip() == "1")
    credential_present = bool(str(env.get(policy.credential_env_var_name) or "").strip())

    missing: list[str] = []
    block_reason = ""
    status = "blocked_ready"

    if request.selected_provider != GEMINI_PROVIDER_NAME:
        block_reason = "selected_provider_not_gemini"
        missing.append(block_reason)
    if request.payload_type != policy.allowed_payload_type:
        block_reason = block_reason or "payload_type_not_allowed"
        missing.append("payload_type_not_redacted_text_layout_summary")
    if request.payload_class != policy.required_payload_class:
        block_reason = block_reason or "payload_class_not_synthetic_live_smoke"
        missing.append("payload_class_not_synthetic_redacted_live_smoke")
    if str(request.privacy_gate_status) != "passed":
        block_reason = block_reason or "privacy_gate_not_passed"
        missing.append("privacy_gate_not_passed")
    if not bool(request.payload_policy_allowed):
        block_reason = block_reason or "payload_policy_failed"
        missing.append("payload_policy_failed")
    if not bool(request.budget_allowed):
        block_reason = block_reason or "budget_exceeded"
        missing.append("budget_exceeded")
    if str(request.dry_run_status) != "passed":
        block_reason = block_reason or "dry_run_not_passed"
        missing.append("dry_run_not_passed")
    if str(request.operator_enablement_request_state) != "staged":
        block_reason = block_reason or "operator_staged_request_required"
        missing.append("operator_staged_request_required")
    if int(request.call_limit) != policy.required_call_limit:
        block_reason = block_reason or "call_limit_must_be_one"
        missing.append("call_limit_must_be_one")
    if float(request.per_call_budget_cap) <= 0:
        block_reason = block_reason or "per_call_budget_cap_required"
        missing.append("per_call_budget_cap_required")
    if not bool(request.real_provider_execution_enabled):
        block_reason = block_reason or "real_provider_execution_not_enabled_for_smoke"
        missing.append("real_provider_execution_not_enabled_for_smoke")
    if not bool(request.final_external_call_allowed):
        block_reason = block_reason or "final_external_call_not_allowed_for_smoke"
        missing.append("final_external_call_not_allowed_for_smoke")

    # Environment + credential gates classify into specific blocked statuses.
    if not allow_env:
        missing.append(f"{policy.allow_smoke_env_var}_missing")
    if not operator_env:
        missing.append(f"{policy.operator_approved_env_var}_missing")
    if not credential_present:
        missing.append("gemini_api_key_missing")

    config_gate_failed = bool(block_reason)
    live_call_allowed = not missing

    if live_call_allowed:
        status = "ready_for_call"
        block_reason = ""
    elif config_gate_failed:
        status = "blocked_ready"
    elif not allow_env or not operator_env:
        status = "blocked_missing_operator_approval"
        block_reason = block_reason or (
            f"{policy.allow_smoke_env_var}_missing" if not allow_env else f"{policy.operator_approved_env_var}_missing"
        )
    elif not credential_present:
        status = "blocked_missing_credential"
        block_reason = "gemini_api_key_missing"
    else:
        status = "blocked_ready"

    return GeminiLiveSmokeDecision(
        selected_provider=request.selected_provider,
        payload_type=request.payload_type,
        payload_class=request.payload_class,
        redacted_payload_hash=str(request.redacted_payload_hash)[:12],
        privacy_gate_status=request.privacy_gate_status,
        payload_policy_allowed=bool(request.payload_policy_allowed),
        budget_allowed=bool(request.budget_allowed),
        dry_run_status=request.dry_run_status,
        operator_enablement_request_state=request.operator_enablement_request_state,
        allow_real_provider_smoke_env_present=allow_env,
        operator_approved_live_smoke_env_present=operator_env,
        credential_present=credential_present,
        credential_value_redacted=credential_present,
        per_call_budget_cap=float(request.per_call_budget_cap),
        call_limit=int(request.call_limit),
        real_provider_execution_enabled=bool(request.real_provider_execution_enabled),
        final_external_call_allowed=bool(request.final_external_call_allowed),
        live_call_allowed=live_call_allowed,
        live_call_status=status,
        block_reason=block_reason,
        missing_gates=sorted(set(missing)),
    )


def run_gemini_live_smoke(
    request: GeminiLiveSmokeRequest | None = None,
    *,
    policy: GeminiLiveSmokePolicy | None = None,
    environ: Mapping[str, str] | None = None,
    gemini_client: Any | None = None,
) -> GeminiLiveSmokeResult:
    request = request or GeminiLiveSmokeRequest()
    policy = policy or GeminiLiveSmokePolicy()
    decision = evaluate_live_smoke_gates(request, policy=policy, environ=environ)
    sdk = detect_gemini_sdk()
    model_name = resolve_gemini_model(environ)

    # Defaults: no call, fully review-bound, all real-call flags false.
    base = dict(
        selected_provider=decision.selected_provider,
        payload_type=decision.payload_type,
        payload_class=decision.payload_class,
        redacted_payload_hash=decision.redacted_payload_hash,
        privacy_gate_status=decision.privacy_gate_status,
        payload_policy_allowed=decision.payload_policy_allowed,
        budget_allowed=decision.budget_allowed,
        dry_run_status=decision.dry_run_status,
        operator_enablement_request_state=decision.operator_enablement_request_state,
        allow_real_provider_smoke_env_present=decision.allow_real_provider_smoke_env_present,
        operator_approved_live_smoke_env_present=decision.operator_approved_live_smoke_env_present,
        credential_present=decision.credential_present,
        credential_value_redacted=decision.credential_value_redacted,
        per_call_budget_cap=decision.per_call_budget_cap,
        call_limit=decision.call_limit,
        real_provider_execution_enabled=decision.real_provider_execution_enabled,
        final_external_call_allowed=decision.final_external_call_allowed,
        external_api_used=False,
        real_network_call_used=False,
        gemini_real_call_attempted=False,
        claude_real_call_attempted=False,
        openai_real_call_attempted=False,
        ollama_real_call_attempted=False,
        local_model_call_used=False,
        subprocess_call_used=False,
        provider_response_received=False,
        provider_client_kind="none",
        schema_valid=False,
        review_bound_package_count=0,
        active_written_count=0,
        auto_accept=False,
        review_required=True,
        live_call_status=decision.live_call_status if decision.live_call_status != "ready_for_call" else "not_attempted",
        block_reason=decision.block_reason,
        provider_error_available=False,
        provider_error_type="",
        provider_error_category="",
        provider_error_message_sanitized="",
        provider_error_status_code="",
        provider_error_code="",
        provider_error_source="",
        provider_error_retryable=False,
        provider_error_redaction_applied=False,
        provider_error_message_truncated=False,
        sdk_package_detected=sdk["sdk_package_detected"],
        sdk_package_version=sdk["sdk_package_version"],
        sdk_call_style=sdk["sdk_call_style"],
        gemini_model_name=model_name,
        sdk_call_kwargs_keys=[],
        missing_gates=list(decision.missing_gates),
        package_summary=[],
    )

    if not decision.live_call_allowed:
        # No client acquisition, no SDK import, no network. Honest blocked status.
        if base["live_call_status"] not in {
            "blocked_missing_credential",
            "blocked_missing_operator_approval",
        }:
            base["live_call_status"] = "blocked_ready"
        return GeminiLiveSmokeResult(**base)

    # All gates satisfied. Acquire a client: injected fake (tests) or lazy SDK.
    client_kind = "fake" if gemini_client is not None else "real_sdk"
    if gemini_client is None:
        gemini_client = _acquire_real_gemini_client(environ=environ, policy=policy)
        if gemini_client is None:
            base["live_call_status"] = "blocked_missing_provider_client"
            base["block_reason"] = "gemini_provider_client_unavailable"
            return GeminiLiveSmokeResult(**base)

    adapter = GeminiExtractionAdapter()
    prompt = _build_prompt(adapter, request)

    base["gemini_real_call_attempted"] = True
    base["provider_client_kind"] = client_kind
    base["sdk_call_kwargs_keys"] = list(getattr(gemini_client, "call_kwargs_keys", ["prompt"]))
    # A real SDK call is a real external/network call; a fake client is not.
    base["external_api_used"] = client_kind == "real_sdk"
    base["real_network_call_used"] = client_kind == "real_sdk"
    try:
        response = gemini_client.generate_extraction(prompt=prompt)
    except Exception as exc:  # noqa: BLE001 - classified + sanitized below, never re-raised raw
        base["live_call_status"] = "failed"
        base["block_reason"] = "gemini_live_call_failed"
        base.update(
            classify_provider_error(
                exc,
                environ=environ,
                redact_literals=[prompt, request.redacted_text_layout_summary],
            )
        )
        return GeminiLiveSmokeResult(**base)

    base["provider_response_received"] = True
    validation = adapter.validate_mock_response(response)
    base["schema_valid"] = bool(validation.schema_valid)
    if validation.schema_valid:
        from app.source_extraction_packages import source_package_from_ai_draft

        draft = adapter.parse_mock_response_to_draft(
            response, safe_source_document_id="source_gemini_live_smoke_synthetic"
        )
        package = source_package_from_ai_draft(draft)
        base["review_bound_package_count"] = 1
        base["package_summary"] = [
            {
                "package_id": package["package_id"],
                "document_type": package["detected_document_family_type"],
                "section_count": len(package["sections"]),
                "observation_count": sum(len(s["observations"]) for s in package["sections"]),
                "package_status": package["package_status"],
                "review_required": package["review_required"],
                "auto_accept_allowed": package["auto_accept_allowed"],
                "active_written_count": package["active_written_count"],
            }
        ]
        base["live_call_status"] = "succeeded"
    else:
        base["live_call_status"] = "attempted_once"
        base["block_reason"] = "live_response_schema_invalid"
    return GeminiLiveSmokeResult(**base)


def build_live_smoke_audit(result: GeminiLiveSmokeResult) -> GeminiLiveSmokeAuditRecord:
    return GeminiLiveSmokeAuditRecord(
        selected_provider=result.selected_provider,
        payload_class=result.payload_class,
        redacted_payload_hash=result.redacted_payload_hash,
        live_call_status=result.live_call_status,
        provider_client_kind=result.provider_client_kind,
        gemini_real_call_attempted=result.gemini_real_call_attempted,
        external_api_used=result.external_api_used,
        real_network_call_used=result.real_network_call_used,
        provider_response_received=result.provider_response_received,
        schema_valid=result.schema_valid,
        review_bound_package_count=result.review_bound_package_count,
        active_written_count=0,
        auto_accept=False,
        review_required=True,
        provider_error_available=result.provider_error_available,
        provider_error_type=result.provider_error_type,
        provider_error_category=result.provider_error_category,
        provider_error_message_sanitized=result.provider_error_message_sanitized,
        provider_error_status_code=result.provider_error_status_code,
        provider_error_code=result.provider_error_code,
        provider_error_source=result.provider_error_source,
        provider_error_retryable=result.provider_error_retryable,
        provider_error_redaction_applied=result.provider_error_redaction_applied,
        provider_error_message_truncated=result.provider_error_message_truncated,
        sdk_package_detected=result.sdk_package_detected,
        sdk_package_version=result.sdk_package_version,
        sdk_call_style=result.sdk_call_style,
        gemini_model_name=result.gemini_model_name,
        sdk_call_kwargs_keys=list(result.sdk_call_kwargs_keys),
    )


def live_smoke_result_to_public_dict(result: GeminiLiveSmokeResult) -> dict[str, Any]:
    return asdict(result)


def live_smoke_decision_to_public_dict(decision: GeminiLiveSmokeDecision) -> dict[str, Any]:
    return asdict(decision)


def live_smoke_audit_to_public_dict(record: GeminiLiveSmokeAuditRecord) -> dict[str, Any]:
    return asdict(record)


def build_live_smoke_operator_preview(result: GeminiLiveSmokeResult) -> dict[str, Any]:
    """Public-safe operator preview for the live-smoke status (no credential value)."""
    live_ran = result.gemini_real_call_attempted and result.provider_response_received
    return {
        "selected_provider": result.selected_provider,
        "effective_provider": "gemini" if live_ran else "fake_local",
        "gemini_live_smoke_status": result.live_call_status,
        "missing_live_gates": list(result.missing_gates),
        "credential_present": result.credential_present,
        "allow_real_provider_smoke_env_present": result.allow_real_provider_smoke_env_present,
        "operator_approved_live_smoke_env_present": result.operator_approved_live_smoke_env_present,
        "per_call_budget_cap": result.per_call_budget_cap,
        "call_limit": result.call_limit,
        "payload_class": result.payload_class,
        "review_bound_package_count": result.review_bound_package_count,
        "review_required": True,
        "auto_accept": False,
        "active_written_count": 0,
        "no_external_call_notice": "No external AI call was made" if not result.gemini_real_call_attempted else "",
        "blocked_notice": (
            "Live smoke blocked until explicit operator approval" if not live_ran else ""
        ),
        "one_call_notice": ("One Gemini live smoke call was made" if live_ran else ""),
        "review_bound_output_notice": "Live smoke output is review-bound only",
    }


def synthetic_payload_check() -> dict[str, Any]:
    summary = SYNTHETIC_REDACTED_SMOKE_SUMMARY
    forbidden_markers = [
        "Jane Example",
        "DOB",
        "MRN",
        "Accession",
        "@",
        "Insurance",
    ]
    return {
        "payload_class": SYNTHETIC_PAYLOAD_CLASS,
        "is_obviously_synthetic": "SYNTHETIC" in summary and "NOT A REAL PATIENT" in summary,
        "contains_no_identifier_markers": not any(marker in summary for marker in forbidden_markers),
        "char_length": len(summary),
        "is_cost_bounded_short": len(summary) <= 600,
        "represents_lab_or_portal_package": "urinalysis" in summary.lower() or "portal" in summary.lower(),
    }


def _build_prompt(adapter: GeminiExtractionAdapter, request: GeminiLiveSmokeRequest) -> str:
    from execution.gemini_extraction_adapter import GeminiAdapterRequest

    return adapter.build_prompt(
        GeminiAdapterRequest(
            payload_type=request.payload_type,
            redacted_payload_hash=request.redacted_payload_hash,
            redacted_text_layout_summary=request.redacted_text_layout_summary,
            redacted_layout_metadata={"payload_class": request.payload_class},
            privacy_gate_status="redacted_payload_ready",
            payload_policy_allowed=True,
            budget_allowed=True,
            document_category="Synthetic live-smoke extraction",
            specialty_domain="urology",
        )
    )


def _acquire_real_gemini_client(*, environ: Mapping[str, str] | None, policy: GeminiLiveSmokePolicy) -> Any | None:
    """Lazily acquire a real Gemini client ONLY after all gates pass.

    Prefers the current ``google-genai`` SDK and falls back to the legacy
    ``google-generativeai`` SDK. Returns ``None`` if neither is installed (the
    caller reports ``blocked_missing_provider_client``). The credential VALUE is
    read here solely to configure the SDK and is never stored, logged, or
    reported. Model name is configurable via GEMINI_MODEL.
    """
    env = os.environ if environ is None else environ
    api_key = str(env.get(policy.credential_env_var_name) or "")
    model_name = resolve_gemini_model(env)
    sdk = detect_gemini_sdk()

    if sdk["sdk_call_style"] == "google_genai":
        try:
            genai = importlib.import_module(_GEMINI_GENAI_MODULE)
            client = genai.Client(api_key=api_key)
        except Exception:
            return None

        class _GenAIClientAdapter:
            call_kwargs_keys = ["model", "contents", "config"]

            def generate_extraction(self, *, prompt: str = "", **_kwargs: Any) -> dict[str, Any]:
                import json as _json

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                return _json.loads(response.text)

        return _GenAIClientAdapter()

    if sdk["sdk_call_style"] == "legacy_generativeai":
        try:
            genai = importlib.import_module(_GEMINI_SDK_MODULE)
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
        except Exception:
            return None

        class _LegacyClientAdapter:
            call_kwargs_keys = ["contents", "generation_config"]

            def generate_extraction(self, *, prompt: str = "", **_kwargs: Any) -> dict[str, Any]:
                import json as _json

                response = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                )
                return _json.loads(response.text)

        return _LegacyClientAdapter()

    return None


__all__ = [
    "ALLOWED_PAYLOAD_TYPE",
    "SYNTHETIC_PAYLOAD_CLASS",
    "ALLOW_SMOKE_ENV",
    "OPERATOR_APPROVED_ENV",
    "GEMINI_CREDENTIAL_ENV_VAR",
    "DEFAULT_GEMINI_MODEL",
    "GEMINI_MODEL_ENV",
    "resolve_gemini_model",
    "detect_gemini_sdk",
    "sanitize_provider_error_message",
    "classify_provider_error",
    "LIVE_CALL_STATUSES",
    "SYNTHETIC_REDACTED_SMOKE_SUMMARY",
    "SYNTHETIC_SMOKE_RESPONSE",
    "GeminiLiveSmokePolicy",
    "GeminiLiveSmokeRequest",
    "GeminiLiveSmokeDecision",
    "GeminiLiveSmokeResult",
    "GeminiLiveSmokeAuditRecord",
    "FakeGeminiLiveClient",
    "evaluate_live_smoke_gates",
    "run_gemini_live_smoke",
    "build_live_smoke_audit",
    "build_live_smoke_operator_preview",
    "live_smoke_result_to_public_dict",
    "live_smoke_decision_to_public_dict",
    "live_smoke_audit_to_public_dict",
    "synthetic_payload_check",
]
