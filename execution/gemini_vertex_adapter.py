"""Google Cloud Vertex Gemini route for the gated 15N-R4 smoke.

This module is separate from the AI Studio Gemini API-key adapter. Defaults are
disabled; a live call is possible only through the explicit 15N-R4 smoke gate.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping
from urllib.parse import quote


VERTEX_PROVIDER_ROUTE = "vertex"
VERTEX_PROVIDER_NAME = "gemini_vertex"
VERTEX_PROJECT_ID = "sot-knowledge-ocr"
VERTEX_LOCATION = "global"
VERTEX_MODEL = "gemini-2.5-flash-lite"
VERTEX_ENDPOINT = "https://aiplatform.googleapis.com"
VERTEX_ENDPOINT_HOST = "aiplatform.googleapis.com"
VERTEX_LIVE_SMOKE_ENV = "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED"
VERTEX_SMOKE_PROMPT = 'Return exactly this JSON and nothing else: {"status":"ok","route":"vertex"}'
VERTEX_MAX_OUTPUT_TOKENS = 32
VERTEX_TEMPERATURE = 0

FORBIDDEN_PAYLOAD_MARKERS = (
    "DOB",
    "MRN",
    "Accession",
    "Patient ",
    "Provider ",
    "Insurance",
    "C:\\",
    ".pdf",
    ".png",
    ".jpg",
    "raw OCR",
    "raw_pdf",
    "raw_image",
)


@dataclass(frozen=True)
class GeminiVertexConfig:
    provider_route: str = VERTEX_PROVIDER_ROUTE
    provider_name: str = VERTEX_PROVIDER_NAME
    vertex_project_id: str = VERTEX_PROJECT_ID
    vertex_location: str = VERTEX_LOCATION
    vertex_model: str = VERTEX_MODEL
    vertex_endpoint: str = VERTEX_ENDPOINT
    vertex_enabled: bool = False
    vertex_live_smoke_allowed: bool = False


@dataclass(frozen=True)
class GeminiVertexSmokeResult:
    provider_route: str
    provider_name: str
    model: str
    location: str
    endpoint_host: str
    live_call_made: bool
    external_api_used: bool
    real_network_call_used: bool
    gemini_real_call_attempted: bool
    provider_response_received: bool
    schema_valid: bool
    response_sanitized: dict[str, Any]
    prompt_token_count: int
    candidates_token_count: int
    total_token_count: int
    active_written_count: int
    auto_accept: bool
    review_required: bool
    billing_check_pending: bool
    live_call_status: str
    block_reason: str
    provider_error_available: bool = False
    provider_error_category: str = ""
    provider_error_type: str = ""
    provider_error_status_code: str = ""
    provider_error_code: str = ""
    provider_error_message_sanitized: str = ""
    provider_error_source: str = ""
    provider_error_retryable: bool = False
    live_retry_allowed: bool = False
    safety_notes: dict[str, bool] = field(default_factory=dict)


def build_vertex_config(environ: Mapping[str, str] | None = None) -> GeminiVertexConfig:
    env = os.environ if environ is None else environ
    return GeminiVertexConfig(
        vertex_project_id=str(env.get("MEDAI_VERTEX_PROJECT_ID") or VERTEX_PROJECT_ID).strip(),
        vertex_location=str(env.get("MEDAI_VERTEX_LOCATION") or VERTEX_LOCATION).strip(),
        vertex_model=str(env.get("MEDAI_VERTEX_MODEL") or VERTEX_MODEL).strip(),
        vertex_endpoint=str(env.get("MEDAI_VERTEX_ENDPOINT") or VERTEX_ENDPOINT).strip().rstrip("/"),
        vertex_enabled=str(env.get("MEDAI_VERTEX_ENABLED") or "").strip().upper() == "YES",
        vertex_live_smoke_allowed=is_vertex_live_smoke_allowed(env),
    )


def is_vertex_live_smoke_allowed(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(VERTEX_LIVE_SMOKE_ENV) or "").strip().upper() == "YES"


def build_vertex_generate_content_url(config: GeminiVertexConfig | None = None) -> str:
    config = config or GeminiVertexConfig()
    project = quote(config.vertex_project_id, safe="")
    location = quote(config.vertex_location, safe="")
    model = quote(config.vertex_model, safe="")
    return (
        f"{config.vertex_endpoint.rstrip('/')}/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model}:generateContent"
    )


def build_vertex_smoke_payload(prompt: str = VERTEX_SMOKE_PROMPT) -> dict[str, Any]:
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": VERTEX_TEMPERATURE,
            "maxOutputTokens": VERTEX_MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
        },
    }


def evaluate_vertex_payload_privacy(prompt: str = VERTEX_SMOKE_PROMPT) -> dict[str, Any]:
    text = str(prompt or "")
    marker_hits = [marker for marker in FORBIDDEN_PAYLOAD_MARKERS if marker.lower() in text.lower()]
    exact_prompt = text == VERTEX_SMOKE_PROMPT
    return {
        "privacy_result": "passed" if exact_prompt and not marker_hits else "failed",
        "synthetic_redacted_payload_only": exact_prompt,
        "no_real_medical_docs_sent": True,
        "no_raw_pdf_image_sent": True,
        "no_private_ocr_payload_sent": True,
        "forbidden_marker_hits": marker_hits,
        "prompt_body_written_to_report": False,
    }


def build_vertex_public_status(config: GeminiVertexConfig | None = None) -> dict[str, Any]:
    config = config or GeminiVertexConfig()
    return {
        **asdict(config),
        "endpoint_host": VERTEX_ENDPOINT_HOST,
        "live_call_made": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "billing_check_pending": True,
    }


def run_gemini_vertex_live_smoke(
    *,
    environ: Mapping[str, str] | None = None,
    token_provider: Callable[[], str] | None = None,
    http_post: Callable[[str, dict[str, Any], str], dict[str, Any]] | None = None,
) -> GeminiVertexSmokeResult:
    env = os.environ if environ is None else environ
    config = build_vertex_config(env)
    privacy = evaluate_vertex_payload_privacy()
    base = _base_result(config, live_call_status="blocked", block_reason="")
    if not config.vertex_live_smoke_allowed:
        return _replace_result(
            base,
            live_call_status="blocked_missing_vertex_live_smoke_gate",
            block_reason=f"{VERTEX_LIVE_SMOKE_ENV}_must_equal_YES",
            safety_notes=privacy,
        )
    if privacy["privacy_result"] != "passed":
        return _replace_result(
            base,
            live_call_status="blocked_privacy_gate",
            block_reason="vertex_smoke_payload_privacy_gate_failed",
            safety_notes=privacy,
        )

    try:
        token = (token_provider or acquire_google_cloud_access_token)()
        if not str(token or "").strip():
            raise RuntimeError("google_cloud_access_token_unavailable")
        response = (http_post or _default_http_post)(
            build_vertex_generate_content_url(config),
            build_vertex_smoke_payload(),
            token,
        )
    except Exception as exc:  # noqa: BLE001 - sanitized below
        error = classify_vertex_provider_error(exc)
        return _replace_result(
            base,
            live_call_made=True,
            external_api_used=True,
            real_network_call_used=True,
            gemini_real_call_attempted=True,
            live_call_status="failed",
            block_reason="vertex_live_call_failed",
            safety_notes=privacy,
            **error,
        )

    parsed = sanitize_vertex_response(response)
    schema_valid = parsed.get("response_sanitized") == {"status": "ok", "route": "vertex"}
    return _replace_result(
        base,
        live_call_made=True,
        external_api_used=True,
        real_network_call_used=True,
        gemini_real_call_attempted=True,
        provider_response_received=True,
        schema_valid=schema_valid,
        response_sanitized=parsed["response_sanitized"],
        prompt_token_count=parsed["prompt_token_count"],
        candidates_token_count=parsed["candidates_token_count"],
        total_token_count=parsed["total_token_count"],
        live_call_status="succeeded" if schema_valid else "failed_schema_invalid",
        block_reason="" if schema_valid else "vertex_response_schema_invalid",
        safety_notes=privacy,
    )


def acquire_google_cloud_access_token() -> str:
    try:
        import google.auth  # type: ignore
        import google.auth.transport.requests  # type: ignore

        credentials, _project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        return str(credentials.token or "")
    except Exception:
        pass

    proc = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return ""
    return str(proc.stdout or "").strip()


def sanitize_vertex_response(response: Mapping[str, Any]) -> dict[str, Any]:
    text = ""
    try:
        text = str(response["candidates"][0]["content"]["parts"][0]["text"])
    except Exception:
        text = ""
    parsed: dict[str, Any] = {}
    if text:
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                parsed = {
                    "status": str(loaded.get("status") or ""),
                    "route": str(loaded.get("route") or ""),
                }
        except Exception:
            parsed = {}
    usage = response.get("usageMetadata") if isinstance(response, Mapping) else {}
    usage = usage if isinstance(usage, Mapping) else {}
    return {
        "response_sanitized": parsed,
        "prompt_token_count": int(usage.get("promptTokenCount") or 0),
        "candidates_token_count": int(usage.get("candidatesTokenCount") or 0),
        "total_token_count": int(usage.get("totalTokenCount") or 0),
    }


def classify_vertex_provider_error(exc: BaseException) -> dict[str, Any]:
    message = sanitize_error_message(str(exc))
    error_type = type(exc).__name__
    lower = str(exc).lower()
    category = "unknown_provider_error"
    if "permission" in lower or "unauth" in lower or "forbidden" in lower or " 403" in lower:
        category = "api_disabled_or_permission"
    elif "quota" in lower or "billing" in lower or "resourceexhausted" in lower or " 429" in lower:
        category = "quota_or_billing"
    elif "not found" in lower or " 404" in lower:
        category = "model_or_endpoint_not_found"
    elif "timeout" in lower or "timed out" in lower:
        category = "timeout"
    elif "invalid" in lower or " 400" in lower:
        category = "request_config_rejected"
    status = str(getattr(exc, "status", "") or getattr(exc, "status_code", "") or "")
    code = str(getattr(exc, "code", "") or "")
    return {
        "provider_error_available": True,
        "provider_error_category": category,
        "provider_error_type": error_type,
        "provider_error_status_code": status,
        "provider_error_code": code,
        "provider_error_message_sanitized": message,
        "provider_error_source": "vertex_generate_content_rest_call",
        "provider_error_retryable": False,
    }


def sanitize_error_message(message: str) -> str:
    text = str(message or "")
    text = re.sub(r"ya29\.[0-9A-Za-z_\-\.]+", "<redacted-credential>", text)
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}", "<redacted-credential>", text)
    text = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "<redacted-credential>", text)
    text = re.sub(r"(?i)authorization\s*[:=]\s*\S+", "authorization=<redacted-credential>", text)
    text = re.sub(r"[A-Za-z]:\\[^\s'\"]*", "<redacted-path>", text)
    if len(text) > 300:
        text = text[:300] + "...<truncated>"
    return text


def vertex_result_to_public_dict(result: GeminiVertexSmokeResult) -> dict[str, Any]:
    return asdict(result)


def _default_http_post(url: str, payload: dict[str, Any], token: str) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vertex_http_error status={exc.code} body={detail}") from exc


def _base_result(config: GeminiVertexConfig, *, live_call_status: str, block_reason: str) -> GeminiVertexSmokeResult:
    return GeminiVertexSmokeResult(
        provider_route=config.provider_route,
        provider_name=config.provider_name,
        model=config.vertex_model,
        location=config.vertex_location,
        endpoint_host=VERTEX_ENDPOINT_HOST,
        live_call_made=False,
        external_api_used=False,
        real_network_call_used=False,
        gemini_real_call_attempted=False,
        provider_response_received=False,
        schema_valid=False,
        response_sanitized={},
        prompt_token_count=0,
        candidates_token_count=0,
        total_token_count=0,
        active_written_count=0,
        auto_accept=False,
        review_required=True,
        billing_check_pending=True,
        live_call_status=live_call_status,
        block_reason=block_reason,
        safety_notes=evaluate_vertex_payload_privacy(),
    )


def _replace_result(result: GeminiVertexSmokeResult, **updates: Any) -> GeminiVertexSmokeResult:
    values = asdict(result)
    values.update(updates)
    return GeminiVertexSmokeResult(**values)


__all__ = [
    "VERTEX_PROVIDER_ROUTE",
    "VERTEX_PROVIDER_NAME",
    "VERTEX_PROJECT_ID",
    "VERTEX_LOCATION",
    "VERTEX_MODEL",
    "VERTEX_ENDPOINT",
    "VERTEX_ENDPOINT_HOST",
    "VERTEX_LIVE_SMOKE_ENV",
    "VERTEX_SMOKE_PROMPT",
    "VERTEX_MAX_OUTPUT_TOKENS",
    "GeminiVertexConfig",
    "GeminiVertexSmokeResult",
    "build_vertex_config",
    "build_vertex_generate_content_url",
    "build_vertex_smoke_payload",
    "build_vertex_public_status",
    "evaluate_vertex_payload_privacy",
    "is_vertex_live_smoke_allowed",
    "run_gemini_vertex_live_smoke",
    "sanitize_vertex_response",
    "vertex_result_to_public_dict",
]
