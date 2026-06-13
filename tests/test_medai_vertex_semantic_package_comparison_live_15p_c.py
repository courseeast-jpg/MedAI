"""No-live tests for MEDAI-VERTEX-SEMANTIC-PACKAGE-COMPARISON-LIVE-15P-C.

Never performs a real provider call. Uses injectable fake http_post /
token_provider (client_kind="fake_injected"). Verifies the corrected request
shape (only contents+generationConfig), gate refusal, comparison/validation
logic, hallucination detection, and sanitization.
"""
from __future__ import annotations

import json

from app.ai_package_run_review_preview import build_run_review_package_previews
from execution.vertex_semantic_package_contract import build_fake_vertex_semantic_response
from scripts.run_medai_vertex_semantic_package_comparison_live_15p_c import (
    ALLOWED_TOP_LEVEL_KEYS,
    LIVE_COMPARE_ENV,
    MAX_OUTPUT_TOKENS,
    PREFERRED_PACKAGE_FAMILY,
    build_vertex_api_body,
    is_live_compare_allowed,
    run_vertex_semantic_live_comparison,
    select_comparison_preview,
)

FAKE_TOKEN = "fake-access-token-not-logged-15p-c"
GATE_ON = {LIVE_COMPARE_ENV: "YES"}


class RecordingHttpPost:
    def __init__(self, response: dict):
        self.calls: list[tuple[str, dict, str]] = []
        self._response = response

    def __call__(self, url: str, payload: dict, token: str) -> dict:
        self.calls.append((url, payload, token))
        return self._response


def _vertex_envelope(body: dict, *, prompt_tokens: int = 40, output_tokens: int = 60) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": output_tokens,
            "totalTokenCount": prompt_tokens + output_tokens,
        },
    }


def _portal_preview():
    return select_comparison_preview()


def _faithful_response(preview) -> dict:
    return build_fake_vertex_semantic_response(preview)


def _run(http_post, *, environ=GATE_ON):
    return run_vertex_semantic_live_comparison(
        environ=environ,
        token_provider=lambda: FAKE_TOKEN,
        http_post=http_post,
        client_kind="fake_injected",
    )


def test_selects_shortest_portal_fixture() -> None:
    preview = select_comparison_preview()
    assert preview.package_family == PREFERRED_PACKAGE_FAMILY
    portal = [p for p in build_run_review_package_previews() if p.package_family == PREFERRED_PACKAGE_FAMILY]
    assert preview.source_visible_body == min(portal, key=lambda p: len(p.source_visible_body)).source_visible_body


def test_gate_refuses_without_env() -> None:
    recorder = RecordingHttpPost(_vertex_envelope(_faithful_response(_portal_preview())))
    result = run_vertex_semantic_live_comparison(
        environ={}, token_provider=lambda: FAKE_TOKEN, http_post=recorder, client_kind="fake_injected"
    )
    assert result["status"] == "BLOCKED_READY_FOR_LIVE_COMPARE"
    assert result["live_call_made"] is False
    assert recorder.calls == []
    assert is_live_compare_allowed({}) is False


def test_does_not_reuse_smoke_gate_env() -> None:
    recorder = RecordingHttpPost(_vertex_envelope(_faithful_response(_portal_preview())))
    result = run_vertex_semantic_live_comparison(
        environ={"MEDAI_VERTEX_LIVE_SMOKE_ALLOWED": "YES"},
        token_provider=lambda: FAKE_TOKEN,
        http_post=recorder,
        client_kind="fake_injected",
    )
    assert result["status"] == "BLOCKED_READY_FOR_LIVE_COMPARE"
    assert recorder.calls == []


def test_build_vertex_api_body_strips_metadata() -> None:
    from execution.vertex_semantic_package_contract import build_vertex_semantic_request_payload

    payload = build_vertex_semantic_request_payload(_portal_preview())
    api_body = build_vertex_api_body(payload)
    assert set(api_body.keys()) == {"contents", "generationConfig"}
    for forbidden in ("provider_route", "provider_name", "model", "location", "endpoint_host", "live_call_made"):
        assert forbidden not in api_body


def test_posted_body_has_only_vertex_api_keys() -> None:
    preview = _portal_preview()
    recorder = RecordingHttpPost(_vertex_envelope(_faithful_response(preview)))
    result = _run(recorder)
    _url, posted, _token = recorder.calls[0]
    assert set(posted.keys()) == {"contents", "generationConfig"}
    assert set(posted.keys()) <= ALLOWED_TOP_LEVEL_KEYS
    assert result["posted_body_allowed_top_level_keys_only"] is True
    assert result["posted_body_top_level_keys"] == ["contents", "generationConfig"]
    for forbidden in ("provider_route", "provider_name", "model", "location", "endpoint_host", "live_call_made", "metadata", "project_id", "package_family"):
        assert forbidden not in posted


def test_faithful_response_passes_contract_single_call() -> None:
    preview = _portal_preview()
    recorder = RecordingHttpPost(_vertex_envelope(_faithful_response(preview)))
    result = _run(recorder)
    assert len(recorder.calls) == 1
    assert result["status"] == "PASS"
    assert result["provider_response_received"] is True
    assert result["schema_validation_pass"] is True
    assert result["hallucinated_field_count"] == 0
    assert result["evidence_anchor_preserved"] is True
    assert result["unknown_values_explicit"] is True
    assert result["uncertainty_flags_visible"] is True
    assert result["source_visible_body_preserved"] is True
    assert result["review_required"] is True
    assert result["auto_accept"] is False
    assert result["active_written_count"] == 0
    assert result["external_api_used"] is False
    assert result["real_network_call_used"] is False


def test_payload_constraints_enforced() -> None:
    preview = _portal_preview()
    recorder = RecordingHttpPost(_vertex_envelope(_faithful_response(preview)))
    _run(recorder)
    _url, payload, token = recorder.calls[0]
    gc = payload["generationConfig"]
    assert gc["temperature"] == 0
    assert gc["maxOutputTokens"] <= 512 and gc["maxOutputTokens"] == MAX_OUTPUT_TOKENS
    assert gc["responseMimeType"] == "application/json"
    assert token == FAKE_TOKEN


def test_hallucinated_field_fails() -> None:
    preview = _portal_preview()
    body = _faithful_response(preview)
    body["semantic_findings"].append(
        {
            "label": "Fabricated Diagnosis",
            "value": "invented",
            "source_section": "Nonexistent Section",
            "evidence_text": "not in source",
            "uncertainty": "",
            "unknown_value": False,
            "source_faithful": True,
        }
    )
    result = _run(RecordingHttpPost(_vertex_envelope(body)))
    assert result["hallucinated_field_count"] >= 1
    assert result["status"] == "FAIL_HALLUCINATED_FIELDS"
    assert result["active_written_count"] == 0


def test_auto_accept_true_rejected() -> None:
    preview = _portal_preview()
    body = _faithful_response(preview)
    body["auto_accept"] = True
    result = _run(RecordingHttpPost(_vertex_envelope(body)))
    assert result["status"] == "FAIL_AUTO_ACCEPT"
    assert result["auto_accept"] is True
    assert result["active_written_count"] == 0


def test_non_json_response_fails() -> None:
    envelope = {
        "candidates": [{"content": {"parts": [{"text": "this is not json"}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
    }
    result = _run(RecordingHttpPost(envelope))
    assert result["status"] == "FAIL_RESPONSE_NOT_JSON"
    assert result["schema_validation_pass"] is False
    assert result["active_written_count"] == 0


def test_schema_invalid_response_fails() -> None:
    preview = _portal_preview()
    body = _faithful_response(preview)
    for finding in body["semantic_findings"]:
        finding.pop("evidence_text", None)
    result = _run(RecordingHttpPost(_vertex_envelope(body)))
    assert result["status"] in {"FAIL_SCHEMA_INVALID", "FAIL_HALLUCINATED_FIELDS"}
    assert result["schema_validation_pass"] is False


def test_token_usage_captured() -> None:
    preview = _portal_preview()
    recorder = RecordingHttpPost(_vertex_envelope(_faithful_response(preview), prompt_tokens=33, output_tokens=44))
    result = _run(recorder)
    assert result["prompt_token_count"] == 33
    assert result["output_token_count"] == 44
    assert result["total_token_count"] == 77


def test_no_credentials_in_result() -> None:
    preview = _portal_preview()
    result = _run(RecordingHttpPost(_vertex_envelope(_faithful_response(preview))))
    serialized = json.dumps(result, default=str)
    assert FAKE_TOKEN not in serialized
    assert "Authorization" not in serialized
    assert "Bearer " not in serialized
    assert "ya29." not in serialized
    assert "AIza" not in serialized


def test_prompt_privacy_passed() -> None:
    preview = _portal_preview()
    result = _run(RecordingHttpPost(_vertex_envelope(_faithful_response(preview))))
    assert result["privacy_result"] == "passed"


def test_metrics_and_billing_pending() -> None:
    preview = _portal_preview()
    result = _run(RecordingHttpPost(_vertex_envelope(_faithful_response(preview))))
    for key in (
        "live_call_made", "provider_response_received", "provider_route", "provider_name",
        "model", "location", "total_token_count", "prompt_token_count", "output_token_count",
        "schema_validation_pass", "source_visible_body_preserved", "evidence_anchor_preserved",
        "candidate_facts_separated", "unknown_values_explicit", "uncertainty_flags_visible",
        "hallucinated_field_count", "review_required", "auto_accept", "active_written_count",
        "external_api_used", "privacy_result", "billing_check_pending",
        "posted_body_allowed_top_level_keys_only",
    ):
        assert key in result
    assert result["billing_check_pending"] is True
    assert result["provider_route"] == "vertex"
    assert result["model"] == "gemini-2.5-flash-lite"


def test_client_kind_fake_marks_no_real_network() -> None:
    preview = _portal_preview()
    result = _run(RecordingHttpPost(_vertex_envelope(_faithful_response(preview))))
    assert result["provider_client_kind"] == "fake_injected"
    assert result["external_api_used"] is False
    assert result["real_network_call_used"] is False
    assert result["single_live_call_only"] is True
