"""No-live tests for MEDAI-GEMINI-VERTEX-CREDIT-ROUTE-SMOKE-15N-R4."""
from __future__ import annotations

import json
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.gemini_extraction_adapter import GEMINI_DEFAULT_MODEL, GEMINI_PROVIDER_NAME
from execution.gemini_vertex_adapter import (
    VERTEX_ENDPOINT_HOST,
    VERTEX_LIVE_SMOKE_ENV,
    VERTEX_MAX_OUTPUT_TOKENS,
    VERTEX_MODEL,
    VERTEX_SMOKE_PROMPT,
    build_vertex_config,
    build_vertex_generate_content_url,
    build_vertex_smoke_payload,
    evaluate_vertex_payload_privacy,
    run_gemini_vertex_live_smoke,
    vertex_result_to_public_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET_TOKEN = "redacted-unit-test-token-15n-r4"


def test_vertex_config_defaults_disabled() -> None:
    config = build_vertex_config({})
    assert config.provider_route == "vertex"
    assert config.provider_name == "gemini_vertex"
    assert config.vertex_project_id == "sot-knowledge-ocr"
    assert config.vertex_location == "global"
    assert config.vertex_model == VERTEX_MODEL
    assert config.vertex_endpoint == "https://aiplatform.googleapis.com"
    assert config.vertex_enabled is False
    assert config.vertex_live_smoke_allowed is False


def test_endpoint_construction_uses_vertex_rest_route() -> None:
    url = build_vertex_generate_content_url(build_vertex_config({}))
    assert url == (
        "https://aiplatform.googleapis.com/v1/projects/sot-knowledge-ocr/locations/global"
        "/publishers/google/models/gemini-2.5-flash-lite:generateContent"
    )
    assert VERTEX_ENDPOINT_HOST in url


def test_request_payload_shape_is_cost_bounded_json_only() -> None:
    payload = build_vertex_smoke_payload()
    assert payload["contents"][0]["parts"][0]["text"] == VERTEX_SMOKE_PROMPT
    assert payload["generationConfig"]["temperature"] == 0
    assert payload["generationConfig"]["maxOutputTokens"] <= VERTEX_MAX_OUTPUT_TOKENS
    assert payload["generationConfig"]["responseMimeType"] == "application/json"


def test_no_gate_blocks_without_network_or_token_provider() -> None:
    called = {"token": 0, "post": 0}

    def token_provider() -> str:
        called["token"] += 1
        return SECRET_TOKEN

    def http_post(_url, _payload, _token):
        called["post"] += 1
        return {}

    result = run_gemini_vertex_live_smoke(environ={}, token_provider=token_provider, http_post=http_post)
    assert result.live_call_made is False
    assert result.external_api_used is False
    assert result.real_network_call_used is False
    assert result.live_call_status == "blocked_missing_vertex_live_smoke_gate"
    assert called == {"token": 0, "post": 0}


def test_live_gate_fake_transport_makes_exactly_one_call_and_sanitizes_response() -> None:
    calls = []

    def token_provider() -> str:
        return SECRET_TOKEN

    def http_post(url, payload, token):
        calls.append({"url": url, "payload": payload, "token": token})
        return {
            "candidates": [{"content": {"parts": [{"text": '{"status":"ok","route":"vertex"}'}]}}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 14, "totalTokenCount": 26},
        }

    result = run_gemini_vertex_live_smoke(
        environ={VERTEX_LIVE_SMOKE_ENV: "YES"},
        token_provider=token_provider,
        http_post=http_post,
    )
    assert len(calls) == 1
    assert result.live_call_made is True
    assert result.external_api_used is True
    assert result.real_network_call_used is True
    assert result.provider_response_received is True
    assert result.schema_valid is True
    assert result.response_sanitized == {"status": "ok", "route": "vertex"}
    assert result.total_token_count == 26
    serialized = json.dumps(vertex_result_to_public_dict(result), sort_keys=True)
    assert SECRET_TOKEN not in serialized
    assert "Authorization" not in serialized
    assert "Bearer " not in serialized


def test_privacy_gate_blocks_real_or_private_payloads() -> None:
    assert evaluate_vertex_payload_privacy()["privacy_result"] == "passed"
    private = evaluate_vertex_payload_privacy("Patient Jane Example DOB 01/02/1970 MRN: 123456 raw OCR")
    assert private["privacy_result"] == "failed"
    assert private["forbidden_marker_hits"]


def test_no_active_writes_auto_accept_false_review_required_true() -> None:
    result = run_gemini_vertex_live_smoke(environ={})
    assert result.active_written_count == 0
    assert result.auto_accept is False
    assert result.review_required is True


def test_old_ai_studio_gemini_adapter_contract_unchanged() -> None:
    assert GEMINI_PROVIDER_NAME == "gemini"
    assert GEMINI_DEFAULT_MODEL == "gemini-disabled-15g"
    old_source = (REPO_ROOT / "execution" / "gemini_extraction_adapter.py").read_text(encoding="utf-8")
    assert "VERTEX_PROVIDER_ROUTE" not in old_source
    assert "aiplatform.googleapis.com" not in old_source


def test_public_result_is_report_safe() -> None:
    result = run_gemini_vertex_live_smoke(environ={})
    payload = vertex_result_to_public_dict(result)
    check = check_public_report_payload(payload)
    assert check.passed, check.leak_examples_redacted
