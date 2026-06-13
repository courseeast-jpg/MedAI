"""No-live tests for 15P-D remaining-families Vertex semantic comparison.

Never performs a real provider call. Uses an injectable fake http_post that
returns source-faithful synthetic responses per family (client_kind=
"fake_injected"). Verifies gate refusal, exactly-3 bounded calls, request shape,
stop-on-first-failure, hallucination/auto-accept rejection, and sanitization.
"""
from __future__ import annotations

import json

from app.ai_package_run_review_preview import build_run_review_package_previews
from execution.vertex_semantic_package_contract import build_fake_vertex_semantic_response
from scripts.run_medai_vertex_semantic_package_remaining_families_live_15p_d import (
    ALLOWED_TOP_LEVEL_KEYS,
    LIVE_ENV,
    MAX_LIVE_CALLS,
    MAX_OUTPUT_TOKENS,
    PORTAL_FAMILY,
    REMAINING_FAMILIES,
    run_remaining_families_comparison,
    select_remaining_previews,
)

FAKE_TOKEN = "fake-access-token-not-logged-15p-d"
GATE_ON = {LIVE_ENV: "YES"}
_PREVIEWS = {p.package_family: p for p in build_run_review_package_previews()}


class FamilyAwareFakeHttp:
    """Returns the source-faithful response for whichever family was posted."""

    def __init__(self, *, mutate=None):
        self.calls: list[tuple[str, dict, str]] = []
        self._mutate = mutate

    def __call__(self, url: str, payload: dict, token: str) -> dict:
        self.calls.append((url, payload, token))
        prompt = str(payload["contents"][0]["parts"][0]["text"])
        # Identify family by matching its synthetic source body in the prompt.
        preview = next(
            (p for f, p in _PREVIEWS.items() if p.source_visible_body and p.source_visible_body in prompt),
            None,
        )
        assert preview is not None, "fake http could not resolve family from prompt"
        body = build_fake_vertex_semantic_response(preview)
        if self._mutate:
            body = self._mutate(preview, body)
        return {
            "candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 60, "totalTokenCount": 110},
        }


def _run(http_post, *, environ=GATE_ON):
    return run_remaining_families_comparison(
        environ=environ,
        token_provider=lambda: FAKE_TOKEN,
        http_post=http_post,
        client_kind="fake_injected",
    )


def test_selects_three_remaining_families_excluding_portal() -> None:
    previews = select_remaining_previews()
    families = [p.package_family for p in previews]
    assert families == list(REMAINING_FAMILIES)
    assert PORTAL_FAMILY not in families
    assert len(families) == 3


def test_gate_refuses_without_env() -> None:
    fake = FamilyAwareFakeHttp()
    result = run_remaining_families_comparison(
        environ={}, token_provider=lambda: FAKE_TOKEN, http_post=fake, client_kind="fake_injected"
    )
    assert result["status"] == "BLOCKED_READY_FOR_LIVE_COMPARE"
    assert result["live_call_count"] == 0
    assert fake.calls == []


def test_does_not_reuse_other_gate_envs() -> None:
    fake = FamilyAwareFakeHttp()
    for wrong in ("MEDAI_VERTEX_LIVE_SMOKE_ALLOWED", "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED"):
        result = run_remaining_families_comparison(
            environ={wrong: "YES"}, token_provider=lambda: FAKE_TOKEN, http_post=fake, client_kind="fake_injected"
        )
        assert result["status"] == "BLOCKED_READY_FOR_LIVE_COMPARE"
    assert fake.calls == []


def test_three_families_pass_with_faithful_responses() -> None:
    fake = FamilyAwareFakeHttp()
    result = _run(fake)
    assert result["status"] == "PASS"
    assert result["package_families_attempted"] == 3
    assert result["package_families_passed"] == 3
    assert len(fake.calls) == 3  # exactly one per family
    assert result["schema_validation_pass_count"] == 3
    assert result["evidence_anchor_preserved_count"] == 3
    assert result["unknown_values_explicit_count"] == 3
    assert result["uncertainty_flags_visible_count"] == 3
    assert result["source_visible_body_preserved_count"] == 3
    assert result["candidate_facts_separated_count"] == 3
    assert result["posted_body_allowed_top_level_keys_only_count"] == 3
    assert result["hallucinated_field_count"] == 0
    assert result["review_required_all"] is True
    assert result["auto_accept_all_false"] is True
    assert result["active_written_count"] == 0
    # fake client is not a real network call
    assert result["external_api_used"] is False


def test_live_call_count_never_exceeds_three() -> None:
    fake = FamilyAwareFakeHttp()
    result = _run(fake)
    assert result["live_call_count"] <= MAX_LIVE_CALLS
    assert len(fake.calls) <= MAX_LIVE_CALLS


def test_every_posted_body_has_only_vertex_keys() -> None:
    fake = FamilyAwareFakeHttp()
    _run(fake)
    for _url, posted, token in fake.calls:
        assert set(posted.keys()) == {"contents", "generationConfig"}
        assert set(posted.keys()) <= ALLOWED_TOP_LEVEL_KEYS
        assert token == FAKE_TOKEN
        gc = posted["generationConfig"]
        assert gc["temperature"] == 0
        assert gc["maxOutputTokens"] == MAX_OUTPUT_TOKENS and gc["maxOutputTokens"] <= 512
        assert gc["responseMimeType"] == "application/json"
        for forbidden in ("provider_route", "provider_name", "model", "location", "metadata", "package_family"):
            assert forbidden not in posted


def test_stop_on_first_failure_hallucination() -> None:
    def mutate(preview, body):
        # Hallucinate only on the FIRST family (cytology) to trigger early stop.
        if preview.package_family == REMAINING_FAMILIES[0]:
            body["semantic_findings"].append(
                {
                    "label": "Fabricated",
                    "value": "x",
                    "source_section": "Nowhere",
                    "evidence_text": "not in source",
                    "uncertainty": "",
                    "unknown_value": False,
                    "source_faithful": True,
                }
            )
        return body

    fake = FamilyAwareFakeHttp(mutate=mutate)
    result = _run(fake)
    assert result["stopped_on_first_failure"] is True
    assert result["status"] == "FAIL_HALLUCINATED_FIELDS"
    assert result["package_families_attempted"] == 1
    assert len(fake.calls) == 1  # stopped after first failing call
    assert result["active_written_count"] == 0


def test_stop_on_auto_accept() -> None:
    def mutate(preview, body):
        if preview.package_family == REMAINING_FAMILIES[0]:
            body["auto_accept"] = True
        return body

    fake = FamilyAwareFakeHttp(mutate=mutate)
    result = _run(fake)
    assert result["status"] == "FAIL_AUTO_ACCEPT"
    assert len(fake.calls) == 1


def test_no_credentials_in_result() -> None:
    fake = FamilyAwareFakeHttp()
    result = _run(fake)
    serialized = json.dumps(result, default=str)
    assert FAKE_TOKEN not in serialized
    assert "Authorization" not in serialized
    assert "Bearer " not in serialized
    assert "ya29." not in serialized
    assert "AIza" not in serialized


def test_token_totals_aggregated() -> None:
    fake = FamilyAwareFakeHttp()
    result = _run(fake)
    assert result["prompt_token_count_all_calls"] == 150
    assert result["output_token_count_all_calls"] == 180
    assert result["total_token_count_all_calls"] == 330


def test_metrics_and_billing_pending() -> None:
    fake = FamilyAwareFakeHttp()
    result = _run(fake)
    for key in (
        "package_families_attempted", "package_families_passed", "live_call_count",
        "provider_response_received_count", "provider_route", "provider_name", "model", "location",
        "total_token_count_all_calls", "prompt_token_count_all_calls", "output_token_count_all_calls",
        "schema_validation_pass_count", "source_visible_body_preserved_count", "evidence_anchor_preserved_count",
        "candidate_facts_separated_count", "unknown_values_explicit_count", "uncertainty_flags_visible_count",
        "hallucinated_field_count", "review_required_all", "auto_accept_all_false", "active_written_count",
        "external_api_used", "privacy_result", "billing_check_pending",
        "posted_body_allowed_top_level_keys_only_count",
    ):
        assert key in result
    assert result["billing_check_pending"] is True
    assert result["provider_route"] == "vertex"
    assert result["model"] == "gemini-2.5-flash-lite"
    assert result["privacy_result"] == "passed"
