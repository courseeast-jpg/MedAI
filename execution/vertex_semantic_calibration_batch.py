"""Bounded synthetic/redacted Vertex semantic calibration batch (15X).

Runs at most 20 live Vertex calls (one per synthetic calibration fixture),
posting only {contents, generationConfig}, and records a per-call quality and
token/cost ledger. Synthetic/redacted fixtures only; JSON-only; temperature=0;
maxOutputTokens<=512. Stop on first failure. No retries. No active MKB writes;
review_required=true; auto_accept=false. No credential/token is ever recorded.

The fixtures duck-type the 15P-A RunReviewPackagePreview shape so the existing
``execution.vertex_semantic_package_contract`` prompt/validate helpers apply
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from execution.gemini_vertex_adapter import (
    VERTEX_LOCATION,
    VERTEX_MODEL,
    VERTEX_PROVIDER_NAME,
    VERTEX_PROVIDER_ROUTE,
    _default_http_post,
    acquire_google_cloud_access_token,
    build_vertex_config,
    build_vertex_generate_content_url,
    classify_vertex_provider_error,
    sanitize_error_message,
)
from execution.vertex_semantic_package_contract import (
    build_fake_vertex_semantic_response,
    build_vertex_semantic_request_payload,
    prompt_privacy_check,
    validate_vertex_semantic_response,
)

LIVE_ENV = "MEDAI_VERTEX_CALIBRATION_BATCH_SYNTHETIC_LIVE_ALLOWED"
MAX_LIVE_CALLS = 20
MAX_OUTPUT_TOKENS = 512
ALLOWED_TOP_LEVEL_KEYS = {"contents", "generationConfig"}

# Conservative LOCAL cost-estimation table for gemini-2.5-flash-lite (USD per
# 1,000 tokens). Documented constants only; no billing API / online pricing.
INPUT_USD_PER_1K = 0.0001   # ~= $0.10 per 1M input tokens
OUTPUT_USD_PER_1K = 0.0004  # ~= $0.40 per 1M output tokens (conservative)
COST_TABLE_NOTE = (
    "Conservative local estimate: input $0.0001/1K tokens, output $0.0004/1K tokens "
    "for gemini-2.5-flash-lite. Not from a billing API; billing_check_pending=true."
)


@dataclass(frozen=True)
class CalibrationFixture:
    fixture_id: str
    package_family: str
    package_family_label: str
    category: str
    source_visible_body: str
    candidate_facts: list[dict[str, Any]]
    evidence_anchors: list[dict[str, str]]
    unknown_values: list[str] = field(default_factory=list)
    uncertainty_flags: list[str] = field(default_factory=list)


def _fact(label: str, value: str, section: str, anchor: str, snippet: str, uncertainty: str = "source-visible candidate fact", unknown: bool = False) -> dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "source_section": section,
        "evidence_anchor_id": anchor,
        "evidence_snippet": snippet,
        "uncertainty": uncertainty,
        "unknown_value": unknown,
        "row_kind": "observation",
    }


def _anchor(anchor_id: str, section: str, snippet: str) -> dict[str, str]:
    return {"anchor_id": anchor_id, "source_section": section, "snippet": snippet}


def build_calibration_fixtures() -> list[CalibrationFixture]:
    fixtures: list[CalibrationFixture] = []

    # 1-2. portal result-card variants
    fixtures.append(CalibrationFixture(
        "cal_portal_v1", "portal_result_cards", "Portal result-card package", "portal_result_cards",
        "SYNTHETIC portal cards: Specific Gravity, pH, Glucose values shown.",
        [_fact("Specific Gravity", "1.015", "Portal Result Cards", "p1", "specific gravity card"),
         _fact("pH", "6.0", "Portal Result Cards", "p1", "ph card")],
        [_anchor("p1", "Portal Result Cards", "synthetic portal result cards")],
        uncertainty_flags=["Specific Gravity: source-visible candidate fact; operator must compare"],
    ))
    fixtures.append(CalibrationFixture(
        "cal_portal_v2", "portal_result_cards", "Portal result-card package", "portal_result_cards",
        "SYNTHETIC portal cards: Color and Appearance descriptive values shown.",
        [_fact("Urine Color", "Yellow", "Portal Result Cards", "p2", "color card"),
         _fact("Appearance", "Clear", "Portal Result Cards", "p2", "appearance card")],
        [_anchor("p2", "Portal Result Cards", "synthetic appearance cards")],
        uncertainty_flags=["Urine Color: source-visible candidate fact; operator must compare"],
    ))

    # 3-4. cytology/pathology narrative variants
    fixtures.append(CalibrationFixture(
        "cal_cyto_v1", "cytology_pathology_narrative", "Cytology/pathology narrative package", "cytology_pathology_narrative",
        "SYNTHETIC narrative: tests ordered section and descriptive impression present.",
        [_fact("Tests Ordered", "panel listed", "Tests Ordered", "c1", "tests ordered section present", "narrative-only; operator must compare")],
        [_anchor("c1", "Tests Ordered", "tests ordered section present")],
        uncertainty_flags=["Tests Ordered: narrative-only; operator must compare"],
    ))
    fixtures.append(CalibrationFixture(
        "cal_cyto_v2", "cytology_pathology_narrative", "Cytology/pathology narrative package", "cytology_pathology_narrative",
        "SYNTHETIC narrative: specimen description paragraph present, no numeric values.",
        [_fact("Specimen Description", "descriptive text present", "Specimen", "c2", "specimen description paragraph", "narrative-only; operator must compare")],
        [_anchor("c2", "Specimen", "specimen description paragraph")],
        uncertainty_flags=["Specimen Description: narrative-only; operator must compare"],
    ))

    # 5-6. urinalysis/table-like lab variants
    fixtures.append(CalibrationFixture(
        "cal_urine_v1", "urinalysis_table_like_lab", "Urinalysis/table-like lab package", "urinalysis_table_like_lab",
        "SYNTHETIC table: pH and protein rows shown with reference ranges.",
        [_fact("pH", "6.5", "Urinalysis", "u1", "ph row"),
         _fact("Protein", "Negative", "Urinalysis", "u1", "protein row")],
        [_anchor("u1", "Urinalysis", "synthetic urinalysis table")],
        uncertainty_flags=["pH: source-visible candidate fact; operator must compare"],
    ))
    fixtures.append(CalibrationFixture(
        "cal_urine_v2", "urinalysis_table_like_lab", "Urinalysis/table-like lab package", "urinalysis_table_like_lab",
        "SYNTHETIC table: glucose and ketones rows shown.",
        [_fact("Glucose", "Negative", "Urinalysis", "u2", "glucose row"),
         _fact("Ketones", "Negative", "Urinalysis", "u2", "ketones row")],
        [_anchor("u2", "Urinalysis", "synthetic urinalysis table v2")],
        uncertainty_flags=["Glucose: source-visible candidate fact; operator must compare"],
    ))

    # 7. mixed narrative + numeric variant
    fixtures.append(CalibrationFixture(
        "cal_mixed_v1", "mixed_narrative_numeric_result", "Mixed narrative + numeric result package", "mixed_narrative_numeric_result",
        "SYNTHETIC mixed: narrative impression plus a numeric value row present.",
        [_fact("Impression", "descriptive text present", "Impression", "m1", "narrative impression present", "narrative-only; operator must compare"),
         _fact("Value A", "12", "Results", "m1", "numeric value row")],
        [_anchor("m1", "Impression", "narrative plus numeric")],
        unknown_values=["Specimen date: unknown"],
        uncertainty_flags=["Impression: narrative-only; operator must compare"],
    ))

    # 8. short clinical note with negation
    fixtures.append(CalibrationFixture(
        "cal_negation", "short_clinical_note_negation", "Short clinical note (negation)", "short_clinical_note_negation",
        "SYNTHETIC note: 'no fever and no rash reported' (negation preserved).",
        [_fact("Fever", "no fever reported", "Note", "n1", "no fever reported", "negation; source-visible only")],
        [_anchor("n1", "Note", "no fever and no rash reported")],
        uncertainty_flags=["Fever: negation; source-visible only"],
    ))

    # 9. short clinical note with uncertainty
    fixtures.append(CalibrationFixture(
        "cal_uncertainty", "short_clinical_note_uncertainty", "Short clinical note (uncertainty)", "short_clinical_note_uncertainty",
        "SYNTHETIC note: 'possible mild finding, uncertain' (uncertainty preserved).",
        [_fact("Finding", "possible mild finding", "Note", "q1", "possible mild finding uncertain", "explicit uncertainty in source")],
        [_anchor("q1", "Note", "possible mild finding, uncertain")],
        uncertainty_flags=["Finding: explicit uncertainty in source"],
    ))

    # 10. medication mention without DDI decision
    fixtures.append(CalibrationFixture(
        "cal_medication", "medication_mention_no_ddi", "Medication mention (no interaction decision)", "medication_mention_no_ddi",
        "SYNTHETIC note: medication list mentions a generic agent; no interaction decision.",
        [_fact("Medication Mention", "generic agent listed", "Medications", "x1", "medication list mention", "source-visible mention; no interaction decision")],
        [_anchor("x1", "Medications", "medication list mention")],
        uncertainty_flags=["Medication Mention: source-visible mention; no interaction decision"],
    ))

    # 11. bilingual / Cyrillic-safe synthetic snippet
    fixtures.append(CalibrationFixture(
        "cal_cyrillic", "bilingual_cyrillic_snippet", "Bilingual/Cyrillic-safe synthetic snippet", "bilingual_cyrillic_snippet",
        "SYNTHETIC bilingual snippet: 'Анализ мочи pH 6.0' alongside English label.",
        [_fact("pH (bilingual)", "6.0", "Urinalysis", "b1", "Анализ мочи pH 6.0", "bilingual source-visible value")],
        [_anchor("b1", "Urinalysis", "Анализ мочи pH 6.0")],
        uncertainty_flags=["pH (bilingual): bilingual source-visible value"],
    ))

    # 12. sparse/low-information result
    fixtures.append(CalibrationFixture(
        "cal_sparse", "sparse_low_information_result", "Sparse low-information result", "sparse_low_information_result",
        "SYNTHETIC sparse result: single label present, value not provided.",
        [_fact("Result Label", "", "Results", "s1", "single sparse label", "value not provided in source", unknown=True)],
        [_anchor("s1", "Results", "single sparse label")],
        unknown_values=["Result value: unknown"],
        uncertainty_flags=["Result Label: value not provided in source"],
    ))

    # 13. multi-section report with explicit unknowns
    fixtures.append(CalibrationFixture(
        "cal_multi_unknown", "multi_section_explicit_unknowns", "Multi-section report with explicit unknowns", "multi_section_explicit_unknowns",
        "SYNTHETIC multi-section: section A has a value; section B value is unknown.",
        [_fact("Section A Value", "5", "Section A", "ms1", "section a value row"),
         _fact("Section B Value", "", "Section B", "ms2", "section b value unknown", "unknown in source", unknown=True)],
        [_anchor("ms1", "Section A", "section a value row"), _anchor("ms2", "Section B", "section b value unknown")],
        unknown_values=["Section B Value: unknown"],
        uncertainty_flags=["Section B Value: unknown in source"],
    ))

    # 14. abnormal numeric values with units
    fixtures.append(CalibrationFixture(
        "cal_abnormal_numeric", "abnormal_numeric_with_units", "Abnormal numeric values with units", "abnormal_numeric_with_units",
        "SYNTHETIC abnormal numeric: a value above reference shown with units.",
        [_fact("Marker X", "15 mg/dL", "Results", "an1", "marker x 15 mg/dL above range")],
        [_anchor("an1", "Results", "marker x 15 mg/dL above range")],
        uncertainty_flags=["Marker X: source-visible value above range; operator must compare"],
    ))

    # 15. normal numeric values with units
    fixtures.append(CalibrationFixture(
        "cal_normal_numeric", "normal_numeric_with_units", "Normal numeric values with units", "normal_numeric_with_units",
        "SYNTHETIC normal numeric: a value within reference shown with units.",
        [_fact("Marker Y", "5 mg/dL", "Results", "nn1", "marker y 5 mg/dL within range")],
        [_anchor("nn1", "Results", "marker y 5 mg/dL within range")],
        uncertainty_flags=["Marker Y: source-visible value within range; operator must compare"],
    ))

    return fixtures


def build_vertex_api_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {"contents": payload["contents"], "generationConfig": payload["generationConfig"]}


def _estimate_cost(prompt_tokens: int, output_tokens: int) -> dict[str, float]:
    input_cost = round((prompt_tokens / 1000.0) * INPUT_USD_PER_1K, 8)
    output_cost = round((output_tokens / 1000.0) * OUTPUT_USD_PER_1K, 8)
    return {
        "estimated_input_cost_usd": input_cost,
        "estimated_output_cost_usd": output_cost,
        "estimated_total_cost_usd": round(input_cost + output_cost, 8),
    }


def estimated_cost_ceiling_usd() -> float:
    per_call = (MAX_OUTPUT_TOKENS / 1000.0) * INPUT_USD_PER_1K + (MAX_OUTPUT_TOKENS / 1000.0) * OUTPUT_USD_PER_1K
    return round(per_call * MAX_LIVE_CALLS, 8)


def _count_hallucinated(response: Mapping[str, Any], fixture: CalibrationFixture) -> int:
    findings = response.get("semantic_findings")
    if not isinstance(findings, list):
        return 0
    allowed = {
        (str(f["label"]), str(f["source_section"]), str(f["evidence_snippet"]))
        for f in fixture.candidate_facts
    }
    hallucinated = 0
    for item in findings:
        if not isinstance(item, Mapping):
            hallucinated += 1
            continue
        identity = (str(item.get("label", "")), str(item.get("source_section", "")), str(item.get("evidence_text", "")))
        if identity not in allowed:
            hallucinated += 1
    return hallucinated


def is_calibration_live_allowed(environ: Mapping[str, str] | None = None) -> bool:
    import os

    env = os.environ if environ is None else environ
    return str(env.get(LIVE_ENV) or "").strip().upper() == "YES"


def compare_one_fixture(
    fixture: CalibrationFixture,
    *,
    token: str,
    http_post: Callable[[str, dict[str, Any], str], dict[str, Any]],
    config: Any,
    client_kind: str,
) -> dict[str, Any]:
    payload = build_vertex_semantic_request_payload(fixture)
    payload["generationConfig"]["maxOutputTokens"] = MAX_OUTPUT_TOKENS
    payload["generationConfig"]["temperature"] = 0
    vertex_api_body = build_vertex_api_body(payload)
    posted_keys = sorted(vertex_api_body.keys())
    prompt = str(payload["contents"][0]["parts"][0]["text"])
    privacy = prompt_privacy_check(prompt)
    is_real = client_kind == "real_rest"
    result: dict[str, Any] = {
        "fixture_id": fixture.fixture_id,
        "package_family": fixture.package_family,
        "category": fixture.category,
        "provider_route": VERTEX_PROVIDER_ROUTE,
        "model": VERTEX_MODEL,
        "posted_body_top_level_keys": posted_keys,
        "posted_body_allowed_top_level_keys_only": set(posted_keys) <= ALLOWED_TOP_LEVEL_KEYS,
        "provider_response_received": False,
        "schema_validation_pass": False,
        "source_visible_body_preserved": False,
        "evidence_anchor_preserved": False,
        "candidate_facts_separated": False,
        "unknown_values_explicit": False,
        "uncertainty_flags_visible": False,
        "hallucinated_field_count": 0,
        "prompt_token_count": 0,
        "output_token_count": 0,
        "total_token_count": 0,
        "estimated_input_cost_usd": 0.0,
        "estimated_output_cost_usd": 0.0,
        "estimated_total_cost_usd": 0.0,
        "review_required": True,
        "auto_accept": False,
        "active_written_count_delta": 0,
        "creates_active_mkb_record": False,
        "privacy_result": privacy["privacy_result"],
        "status": "PENDING",
        "block_reason": "",
    }
    if privacy["privacy_result"] != "passed":
        result["status"] = "FAIL_PRIVACY_GATE"
        result["block_reason"] = "calibration_prompt_privacy_gate_failed"
        return result
    if not result["posted_body_allowed_top_level_keys_only"]:
        result["status"] = "FAIL_INVALID_REQUEST_SHAPE"
        result["block_reason"] = "vertex_post_body_has_forbidden_top_level_keys"
        return result
    try:
        response = http_post(build_vertex_generate_content_url(config), vertex_api_body, token)
    except Exception as exc:  # noqa: BLE001 - sanitized below
        error = classify_vertex_provider_error(exc)
        result.update({"status": "FAIL_LIVE_CALL", "block_reason": "vertex_calibration_live_call_failed", **error})
        return result

    import json as _json

    parsed: dict[str, Any] = {}
    try:
        raw_text = str(response["candidates"][0]["content"]["parts"][0]["text"])
        loaded = _json.loads(raw_text)
        if isinstance(loaded, dict):
            parsed = loaded
    except Exception:
        parsed = {}
    usage = response.get("usageMetadata") if isinstance(response, Mapping) else {}
    usage = usage if isinstance(usage, Mapping) else {}
    result["provider_response_received"] = True
    result["prompt_token_count"] = int(usage.get("promptTokenCount") or 0)
    result["output_token_count"] = int(usage.get("candidatesTokenCount") or 0)
    result["total_token_count"] = int(usage.get("totalTokenCount") or 0)
    result.update(_estimate_cost(result["prompt_token_count"], result["output_token_count"]))
    del is_real

    if not parsed:
        result["status"] = "FAIL_RESPONSE_NOT_JSON"
        result["block_reason"] = "vertex_response_not_valid_json"
        return result

    schema_valid, errors = validate_vertex_semantic_response(parsed, fixture)
    hallucinated = _count_hallucinated(parsed, fixture)
    findings = parsed.get("semantic_findings") if isinstance(parsed.get("semantic_findings"), list) else []
    result["schema_validation_pass"] = bool(schema_valid)
    result["validation_errors"] = list(errors)
    result["hallucinated_field_count"] = hallucinated
    result["source_visible_body_preserved"] = parsed.get("package_family") == fixture.package_family
    result["evidence_anchor_preserved"] = bool(fixture.evidence_anchors) and all(
        str(f.get("evidence_text", "")).strip() for f in findings if isinstance(f, Mapping)
    )
    result["candidate_facts_separated"] = bool(fixture.candidate_facts) and bool(findings)
    result["unknown_values_explicit"] = all(isinstance(f, Mapping) and "unknown_value" in f for f in findings) and bool(findings)
    result["uncertainty_flags_visible"] = all(isinstance(f, Mapping) and "uncertainty" in f for f in findings) and bool(findings)
    result["review_required"] = parsed.get("review_required") is True
    result["auto_accept"] = parsed.get("auto_accept") is True

    passed = (
        schema_valid
        and hallucinated == 0
        and result["review_required"] is True
        and result["auto_accept"] is False
        and result["source_visible_body_preserved"]
        and result["evidence_anchor_preserved"]
        and result["unknown_values_explicit"]
        and result["uncertainty_flags_visible"]
    )
    if result["auto_accept"] is True:
        result["status"] = "FAIL_AUTO_ACCEPT"
        result["block_reason"] = "vertex_response_set_auto_accept"
    elif hallucinated > 0:
        result["status"] = "FAIL_HALLUCINATED_FIELDS"
        result["block_reason"] = "vertex_response_added_unsupported_fields"
    elif not schema_valid:
        result["status"] = "FAIL_SCHEMA_INVALID"
        result["block_reason"] = "vertex_response_schema_invalid"
    else:
        result["status"] = "PASS" if passed else "FAIL_CONTRACT_NOT_PRESERVED"
        result["block_reason"] = "" if passed else "vertex_response_contract_not_preserved"
    return result


def run_calibration_batch(
    *,
    environ: Mapping[str, str] | None = None,
    token_provider: Callable[[], str] | None = None,
    http_post: Callable[[str, dict[str, Any], str], dict[str, Any]] | None = None,
    client_kind: str = "real_rest",
) -> dict[str, Any]:
    import os

    env = os.environ if environ is None else environ
    fixtures = build_calibration_fixtures()
    is_real = client_kind == "real_rest"

    agg: dict[str, Any] = {
        "block": "MEDAI-VERTEX-SEMANTIC-CALIBRATION-BATCH-SYNTHETIC-LIVE-15X",
        "provider_route": VERTEX_PROVIDER_ROUTE,
        "provider_name": VERTEX_PROVIDER_NAME,
        "model": VERTEX_MODEL,
        "location": VERTEX_LOCATION,
        "provider_client_kind": client_kind,
        "fixture_count": len(fixtures),
        "max_live_calls": MAX_LIVE_CALLS,
        "live_call_count": 0,
        "provider_response_received_count": 0,
        "schema_validation_pass_count": 0,
        "source_visible_body_preserved_count": 0,
        "evidence_anchor_preserved_count": 0,
        "candidate_facts_separated_count": 0,
        "unknown_values_explicit_count": 0,
        "uncertainty_flags_visible_count": 0,
        "posted_body_allowed_top_level_keys_only_count": 0,
        "review_required_count": 0,
        "hallucinated_field_count": 0,
        "total_prompt_tokens": 0,
        "total_output_tokens": 0,
        "total_token_count_all_calls": 0,
        "estimated_total_cost_usd_all_calls": 0.0,
        "estimated_cost_ceiling_usd": estimated_cost_ceiling_usd(),
        "cost_table_note": COST_TABLE_NOTE,
        "failed_call_count": 0,
        "active_written_count": 0,
        "active_mkb_record_created_count": 0,
        "auto_accept_true_count": 0,
        "category_breakdown": {},
        "per_call_results": [],
        "stopped_early": False,
        "stop_reason": "",
        "status": "BLOCKED",
        "live_call_made": False,
        "external_api_used": False,
        "privacy_result": "passed",
        "billing_check_pending": True,
    }

    if not is_calibration_live_allowed(env):
        agg["status"] = "BLOCKED_READY_FOR_CALIBRATION"
        agg["stop_reason"] = f"{LIVE_ENV}_must_equal_YES"
        return agg
    if not (12 <= len(fixtures) <= MAX_LIVE_CALLS):
        agg["status"] = "BLOCKED_FIXTURE_COUNT_OUT_OF_RANGE"
        agg["stop_reason"] = f"fixture_count_{len(fixtures)}_not_in_[12,{MAX_LIVE_CALLS}]"
        return agg

    post = http_post or _default_http_post
    get_token = token_provider or acquire_google_cloud_access_token
    config = build_vertex_config(env)
    try:
        token = get_token()
        if not str(token or "").strip():
            raise RuntimeError("google_cloud_access_token_unavailable")
    except Exception as exc:  # noqa: BLE001
        error = classify_vertex_provider_error(exc)
        agg.update({"status": "FAIL_TOKEN_UNAVAILABLE", "stop_reason": "google_cloud_access_token_unavailable", **error})
        return agg

    agg["live_call_made"] = is_real
    agg["external_api_used"] = is_real
    for fixture in fixtures[:MAX_LIVE_CALLS]:
        if is_real:
            agg["live_call_count"] += 1
        result = compare_one_fixture(fixture, token=token, http_post=post, config=config, client_kind=client_kind)
        agg["per_call_results"].append(result)
        agg["category_breakdown"][fixture.category] = agg["category_breakdown"].get(fixture.category, 0) + 1
        agg["provider_response_received_count"] += int(bool(result["provider_response_received"]))
        agg["schema_validation_pass_count"] += int(bool(result["schema_validation_pass"]))
        agg["source_visible_body_preserved_count"] += int(bool(result["source_visible_body_preserved"]))
        agg["evidence_anchor_preserved_count"] += int(bool(result["evidence_anchor_preserved"]))
        agg["candidate_facts_separated_count"] += int(bool(result["candidate_facts_separated"]))
        agg["unknown_values_explicit_count"] += int(bool(result["unknown_values_explicit"]))
        agg["uncertainty_flags_visible_count"] += int(bool(result["uncertainty_flags_visible"]))
        agg["posted_body_allowed_top_level_keys_only_count"] += int(bool(result["posted_body_allowed_top_level_keys_only"]))
        agg["review_required_count"] += int(bool(result["review_required"]))
        agg["hallucinated_field_count"] += int(result["hallucinated_field_count"])
        agg["total_prompt_tokens"] += int(result["prompt_token_count"])
        agg["total_output_tokens"] += int(result["output_token_count"])
        agg["total_token_count_all_calls"] += int(result["total_token_count"])
        agg["estimated_total_cost_usd_all_calls"] = round(agg["estimated_total_cost_usd_all_calls"] + float(result["estimated_total_cost_usd"]), 8)
        if result["auto_accept"] is True:
            agg["auto_accept_true_count"] += 1
        if result["status"] != "PASS":
            agg["failed_call_count"] += 1
            agg["stopped_early"] = True
            agg["status"] = result["status"]
            agg["stop_reason"] = result["block_reason"] or result["status"]
            return agg

    agg["status"] = (
        "PASS"
        if (
            agg["schema_validation_pass_count"] == len(fixtures)
            and agg["hallucinated_field_count"] == 0
            and agg["auto_accept_true_count"] == 0
            and agg["active_written_count"] == 0
        )
        else "FAIL_CONTRACT_NOT_PRESERVED"
    )
    return agg


def build_fake_calibration_response(fixture: CalibrationFixture) -> dict[str, Any]:
    return build_fake_vertex_semantic_response(fixture)


def fixtures_public_dict() -> list[dict[str, Any]]:
    out = []
    for f in build_calibration_fixtures():
        out.append({
            "fixture_id": f.fixture_id,
            "package_family": f.package_family,
            "category": f.category,
            "source_visible_body": f.source_visible_body,
            "candidate_fact_count": len(f.candidate_facts),
            "evidence_anchor_count": len(f.evidence_anchors),
            "unknown_value_count": len(f.unknown_values),
            "uncertainty_flag_count": len(f.uncertainty_flags),
        })
    return out


def sanitize_text(text: str) -> str:
    return sanitize_error_message(text)


__all__ = [
    "LIVE_ENV",
    "MAX_LIVE_CALLS",
    "MAX_OUTPUT_TOKENS",
    "ALLOWED_TOP_LEVEL_KEYS",
    "INPUT_USD_PER_1K",
    "OUTPUT_USD_PER_1K",
    "COST_TABLE_NOTE",
    "CalibrationFixture",
    "build_calibration_fixtures",
    "build_fake_calibration_response",
    "build_vertex_api_body",
    "estimated_cost_ceiling_usd",
    "fixtures_public_dict",
    "is_calibration_live_allowed",
    "compare_one_fixture",
    "run_calibration_batch",
]
