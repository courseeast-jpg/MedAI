"""No-live tests for the 15X Vertex synthetic calibration batch."""
from __future__ import annotations

import json

from execution.vertex_semantic_calibration_batch import (
    ALLOWED_TOP_LEVEL_KEYS,
    LIVE_ENV,
    MAX_LIVE_CALLS,
    MAX_OUTPUT_TOKENS,
    build_calibration_fixtures,
    build_fake_calibration_response,
    fixtures_public_dict,
    run_calibration_batch,
)

REQUIRED_CATEGORIES = {
    "portal_result_cards",
    "cytology_pathology_narrative",
    "urinalysis_table_like_lab",
    "mixed_narrative_numeric_result",
    "short_clinical_note_negation",
    "short_clinical_note_uncertainty",
    "medication_mention_no_ddi",
    "bilingual_cyrillic_snippet",
    "sparse_low_information_result",
    "multi_section_explicit_unknowns",
    "abnormal_numeric_with_units",
    "normal_numeric_with_units",
}
PRIVATE_MARKERS = ["DOB", "MRN", "Accession", "ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/"]


class FamilyAwareFakeHttp:
    def __init__(self, *, mutate=None):
        self.calls = []
        self._mutate = mutate

    def __call__(self, url, payload, token):
        self.calls.append((url, payload, token))
        prompt = payload["contents"][0]["parts"][0]["text"]
        fx = next(f for f in build_calibration_fixtures() if f.source_visible_body in prompt)
        body = build_fake_calibration_response(fx)
        if self._mutate:
            body = self._mutate(fx, body)
        return {
            "candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 80, "totalTokenCount": 180},
        }


def _run(http_post, *, environ=None):
    return run_calibration_batch(
        environ=environ if environ is not None else {LIVE_ENV: "YES"},
        token_provider=lambda: "fake-token-not-logged",
        http_post=http_post,
        client_kind="fake_injected",
    )


def test_refuses_without_gate() -> None:
    fake = FamilyAwareFakeHttp()
    agg = run_calibration_batch(environ={}, token_provider=lambda: "fake", http_post=fake, client_kind="fake_injected")
    assert agg["status"] == "BLOCKED_READY_FOR_CALIBRATION"
    assert fake.calls == []
    assert agg["live_call_made"] is False


def test_does_not_reuse_old_gates() -> None:
    fake = FamilyAwareFakeHttp()
    for wrong in (
        "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_REMAINING_FAMILIES_LIVE_ALLOWED",
    ):
        agg = run_calibration_batch(environ={wrong: "YES"}, token_provider=lambda: "fake", http_post=fake, client_kind="fake_injected")
        assert agg["status"] == "BLOCKED_READY_FOR_CALIBRATION"
    assert fake.calls == []


def test_fixture_count_in_range_and_categories_covered() -> None:
    fixtures = build_calibration_fixtures()
    assert 12 <= len(fixtures) <= MAX_LIVE_CALLS
    categories = {f.category for f in fixtures}
    assert REQUIRED_CATEGORIES.issubset(categories)


def test_no_private_identifiers_in_fixtures() -> None:
    blob = json.dumps([{
        "body": f.source_visible_body,
        "facts": f.candidate_facts,
        "anchors": f.evidence_anchors,
        "unknown": f.unknown_values,
        "uncertainty": f.uncertainty_flags,
    } for f in build_calibration_fixtures()], ensure_ascii=True)
    for marker in PRIVATE_MARKERS:
        assert marker not in blob


def test_all_fixtures_pass_with_faithful_responses() -> None:
    fake = FamilyAwareFakeHttp()
    agg = _run(fake)
    n = agg["fixture_count"]
    assert agg["status"] == "PASS"
    assert len(fake.calls) == n
    assert agg["schema_validation_pass_count"] == n
    assert agg["evidence_anchor_preserved_count"] == n
    assert agg["unknown_values_explicit_count"] == n
    assert agg["uncertainty_flags_visible_count"] == n
    assert agg["posted_body_allowed_top_level_keys_only_count"] == n
    assert agg["review_required_count"] == n
    assert agg["hallucinated_field_count"] == 0
    assert agg["auto_accept_true_count"] == 0
    assert agg["active_written_count"] == 0
    assert agg["failed_call_count"] == 0


def test_posted_body_only_contents_and_generation_config() -> None:
    fake = FamilyAwareFakeHttp()
    _run(fake)
    for _url, posted, token in fake.calls:
        assert set(posted.keys()) == {"contents", "generationConfig"}
        assert set(posted.keys()) <= ALLOWED_TOP_LEVEL_KEYS
        assert token == "fake-token-not-logged"
        gc = posted["generationConfig"]
        assert gc["temperature"] == 0
        assert gc["maxOutputTokens"] == MAX_OUTPUT_TOKENS and gc["maxOutputTokens"] <= 512
        assert gc["responseMimeType"] == "application/json"
        for forbidden in ("provider_route", "provider_name", "model", "location", "endpoint", "project_id", "package_id", "package_family", "metadata", "source_report_reference"):
            assert forbidden not in posted


def test_one_call_per_fixture_no_retry_and_call_bound() -> None:
    fake = FamilyAwareFakeHttp()
    agg = _run(fake)
    assert len(fake.calls) == agg["fixture_count"]
    assert agg["live_call_count"] <= MAX_LIVE_CALLS
    # no duplicate fixture posts (one call per fixture)
    prompts = [c[1]["contents"][0]["parts"][0]["text"] for c in fake.calls]
    assert len(prompts) == len(set(prompts))


def test_stop_on_first_failure_hallucination() -> None:
    first_category = build_calibration_fixtures()[0].fixture_id

    def mutate(fx, body):
        if fx.fixture_id == first_category:
            body["semantic_findings"].append({
                "label": "Fabricated", "value": "x", "source_section": "Nowhere",
                "evidence_text": "not in source", "uncertainty": "", "unknown_value": False, "source_faithful": True,
            })
        return body

    fake = FamilyAwareFakeHttp(mutate=mutate)
    agg = _run(fake)
    assert agg["stopped_early"] is True
    assert agg["status"] == "FAIL_HALLUCINATED_FIELDS"
    assert len(fake.calls) == 1
    assert agg["active_written_count"] == 0


def test_stop_on_schema_failure() -> None:
    first_id = build_calibration_fixtures()[0].fixture_id

    def mutate(fx, body):
        if fx.fixture_id == first_id:
            for finding in body["semantic_findings"]:
                finding.pop("evidence_text", None)
        return body

    fake = FamilyAwareFakeHttp(mutate=mutate)
    agg = _run(fake)
    assert agg["stopped_early"] is True
    assert agg["status"] in {"FAIL_SCHEMA_INVALID", "FAIL_HALLUCINATED_FIELDS"}


def test_stop_on_auto_accept() -> None:
    first_id = build_calibration_fixtures()[0].fixture_id

    def mutate(fx, body):
        if fx.fixture_id == first_id:
            body["auto_accept"] = True
        return body

    fake = FamilyAwareFakeHttp(mutate=mutate)
    agg = _run(fake)
    assert agg["status"] == "FAIL_AUTO_ACCEPT"
    assert len(fake.calls) == 1


def test_token_cost_ledger_fields_present() -> None:
    fake = FamilyAwareFakeHttp()
    agg = _run(fake)
    for r in agg["per_call_results"]:
        for field in (
            "fixture_id", "package_family", "provider_route", "model",
            "prompt_token_count", "output_token_count", "total_token_count",
            "estimated_input_cost_usd", "estimated_output_cost_usd", "estimated_total_cost_usd",
            "provider_response_received", "schema_validation_pass", "hallucinated_field_count",
            "privacy_result", "status",
        ):
            assert field in r
    for field in (
        "live_call_count", "fixture_count", "provider_response_received_count", "schema_validation_pass_count",
        "total_prompt_tokens", "total_output_tokens", "total_token_count_all_calls",
        "estimated_total_cost_usd_all_calls", "estimated_cost_ceiling_usd", "failed_call_count",
        "stopped_early", "stop_reason",
    ):
        assert field in agg
    assert agg["billing_check_pending"] is True


def test_no_credential_or_private_leak_in_output() -> None:
    fake = FamilyAwareFakeHttp()
    agg = _run(fake)
    blob = json.dumps(agg, default=str, ensure_ascii=True)
    for marker in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/", "fake-token-not-logged"):
        assert marker not in blob


def test_no_live_provider_call_during_no_live_tests() -> None:
    # All tests above use FamilyAwareFakeHttp; the module must not import provider SDKs eagerly.
    import sys
    for module in ("google.generativeai", "vertexai"):
        assert module not in sys.modules
