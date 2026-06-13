#!/usr/bin/env python3
"""MEDAI-VERTEX-SEMANTIC-PACKAGE-COMPARISON-REMAINING-FAMILIES-LIVE-15P-D.

Tightly bounded live Vertex semantic-package comparison for the THREE remaining
synthetic package families not covered by 15P-C (portal result-card):

  1. cytology_pathology_narrative
  2. urinalysis_table_like_lab
  3. mixed_narrative_numeric_result

Exactly one live Vertex call per family (3 max), no retries, stop on first
failure. Posts only {contents, generationConfig}. Synthetic/redacted payloads
only; JSON-only; temperature=0; maxOutputTokens<=512. Never logs credentials,
tokens, auth headers, API keys, or ADC paths. No active MKB writes;
review_required=true; auto_accept=false.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ai_package_run_review_preview import (
    RunReviewPackagePreview,
    build_run_review_package_previews,
)
from execution.gemini_vertex_adapter import (
    VERTEX_ENDPOINT_HOST,
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
    build_vertex_semantic_request_payload,
    prompt_privacy_check,
    validate_vertex_semantic_response,
)

LIVE_ENV = "MEDAI_VERTEX_SEMANTIC_REMAINING_FAMILIES_LIVE_ALLOWED"
PORTAL_FAMILY = "portal_result_cards"
REMAINING_FAMILIES = (
    "cytology_pathology_narrative",
    "urinalysis_table_like_lab",
    "mixed_narrative_numeric_result",
)
MAX_LIVE_CALLS = 3
MAX_OUTPUT_TOKENS = 512
ALLOWED_TOP_LEVEL_KEYS = {"contents", "generationConfig"}

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_package_remaining_families_live_15p_d"
SUMMARY_JSON = REPORT_DIR / "summary.json"
RESULTS_JSON = REPORT_DIR / "live_remaining_families_results.json"
MATRIX_MD = REPORT_DIR / "live_remaining_families_matrix.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"


def is_live_allowed(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(LIVE_ENV) or "").strip().upper() == "YES"


def select_remaining_previews() -> list[RunReviewPackagePreview]:
    by_family = {p.package_family: p for p in build_run_review_package_previews()}
    previews: list[RunReviewPackagePreview] = []
    for family in REMAINING_FAMILIES:
        if family in by_family:
            previews.append(by_family[family])
    return previews


def build_vertex_api_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {"contents": payload["contents"], "generationConfig": payload["generationConfig"]}


def _sanitize_findings(findings: list[Any]) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for item in findings:
        if not isinstance(item, Mapping):
            continue
        safe.append(
            {
                "label": sanitize_error_message(str(item.get("label", ""))),
                "value": sanitize_error_message(str(item.get("value", ""))),
                "source_section": sanitize_error_message(str(item.get("source_section", ""))),
                "evidence_text": sanitize_error_message(str(item.get("evidence_text", ""))),
                "uncertainty": sanitize_error_message(str(item.get("uncertainty", ""))),
                "unknown_value": bool(item.get("unknown_value", False)),
                "source_faithful": bool(item.get("source_faithful", False)),
            }
        )
    return safe


def _count_hallucinated(response: Mapping[str, Any], preview: RunReviewPackagePreview) -> int:
    findings = response.get("semantic_findings")
    if not isinstance(findings, list):
        return 0
    allowed = {
        (str(fact["label"]), str(fact["source_section"]), str(fact["evidence_snippet"]))
        for fact in preview.candidate_facts
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


def compare_one_family(
    preview: RunReviewPackagePreview,
    *,
    token: str,
    http_post: Callable[[str, dict[str, Any], str], dict[str, Any]],
    config: Any,
    client_kind: str,
) -> dict[str, Any]:
    payload = build_vertex_semantic_request_payload(preview)
    payload["generationConfig"]["maxOutputTokens"] = MAX_OUTPUT_TOKENS
    payload["generationConfig"]["temperature"] = 0
    vertex_api_body = build_vertex_api_body(payload)
    posted_keys = sorted(vertex_api_body.keys())
    is_real = client_kind == "real_rest"
    result: dict[str, Any] = {
        "package_family": preview.package_family,
        "package_family_label": preview.package_family_label,
        "posted_body_top_level_keys": posted_keys,
        "posted_body_allowed_top_level_keys_only": set(posted_keys) <= ALLOWED_TOP_LEVEL_KEYS,
        "live_call_made": is_real,
        "external_api_used": is_real,
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
        "review_required": True,
        "auto_accept": False,
        "active_written_count": 0,
        "sanitized_findings": [],
        "validation_errors": [],
        "status": "PENDING",
        "block_reason": "",
    }
    if not result["posted_body_allowed_top_level_keys_only"]:
        result["status"] = "FAIL_INVALID_REQUEST_SHAPE"
        result["block_reason"] = "vertex_post_body_has_forbidden_top_level_keys"
        result["live_call_made"] = False
        result["external_api_used"] = False
        return result
    try:
        response = http_post(build_vertex_generate_content_url(config), vertex_api_body, token)
    except Exception as exc:  # noqa: BLE001 - sanitized below
        error = classify_vertex_provider_error(exc)
        result.update({"status": "FAIL_LIVE_CALL", "block_reason": "vertex_semantic_live_call_failed", **error})
        return result

    parsed: dict[str, Any] = {}
    try:
        raw_text = str(response["candidates"][0]["content"]["parts"][0]["text"])
        loaded = json.loads(raw_text)
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

    if not parsed:
        result["status"] = "FAIL_RESPONSE_NOT_JSON"
        result["block_reason"] = "vertex_response_not_valid_json"
        return result

    schema_valid, errors = validate_vertex_semantic_response(parsed, preview)
    hallucinated = _count_hallucinated(parsed, preview)
    findings = parsed.get("semantic_findings") if isinstance(parsed.get("semantic_findings"), list) else []
    result["schema_validation_pass"] = bool(schema_valid)
    result["validation_errors"] = list(errors)
    result["hallucinated_field_count"] = hallucinated
    result["sanitized_findings"] = _sanitize_findings(findings)
    result["source_visible_body_preserved"] = parsed.get("package_family") == preview.package_family
    result["evidence_anchor_preserved"] = bool(preview.evidence_anchors) and all(
        str(f.get("evidence_text", "")).strip() for f in findings if isinstance(f, Mapping)
    )
    result["candidate_facts_separated"] = bool(preview.candidate_facts) and bool(findings)
    result["unknown_values_explicit"] = all(
        isinstance(f, Mapping) and "unknown_value" in f for f in findings
    ) and bool(findings)
    result["uncertainty_flags_visible"] = all(
        isinstance(f, Mapping) and "uncertainty" in f for f in findings
    ) and bool(findings)
    result["review_required"] = parsed.get("review_required") is True
    result["auto_accept"] = parsed.get("auto_accept") is True

    passed = (
        schema_valid
        and hallucinated == 0
        and result["review_required"] is True
        and result["auto_accept"] is False
        and result["active_written_count"] == 0
        and result["evidence_anchor_preserved"]
        and result["unknown_values_explicit"]
        and result["uncertainty_flags_visible"]
        and result["source_visible_body_preserved"]
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


def run_remaining_families_comparison(
    *,
    environ: Mapping[str, str] | None = None,
    token_provider: Callable[[], str] | None = None,
    http_post: Callable[[str, dict[str, Any], str], dict[str, Any]] | None = None,
    client_kind: str = "real_rest",
) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    previews = select_remaining_previews()
    # Privacy precheck on every prompt before any call.
    privacy_results = []
    for preview in previews:
        payload = build_vertex_semantic_request_payload(preview)
        privacy_results.append(prompt_privacy_check(str(payload["contents"][0]["parts"][0]["text"])))
    privacy_ok = all(p["privacy_result"] == "passed" for p in privacy_results)

    aggregate: dict[str, Any] = {
        "block": "MEDAI-VERTEX-SEMANTIC-PACKAGE-COMPARISON-REMAINING-FAMILIES-LIVE-15P-D",
        "provider_route": VERTEX_PROVIDER_ROUTE,
        "provider_name": VERTEX_PROVIDER_NAME,
        "model": VERTEX_MODEL,
        "location": VERTEX_LOCATION,
        "endpoint_host": VERTEX_ENDPOINT_HOST,
        "remaining_families": list(REMAINING_FAMILIES),
        "provider_client_kind": client_kind,
        "max_live_calls": MAX_LIVE_CALLS,
        "package_families_attempted": 0,
        "package_families_passed": 0,
        "live_call_count": 0,
        "provider_response_received_count": 0,
        "schema_validation_pass_count": 0,
        "source_visible_body_preserved_count": 0,
        "evidence_anchor_preserved_count": 0,
        "candidate_facts_separated_count": 0,
        "unknown_values_explicit_count": 0,
        "uncertainty_flags_visible_count": 0,
        "posted_body_allowed_top_level_keys_only_count": 0,
        "hallucinated_field_count": 0,
        "prompt_token_count_all_calls": 0,
        "output_token_count_all_calls": 0,
        "total_token_count_all_calls": 0,
        "review_required_all": True,
        "auto_accept_all_false": True,
        "active_written_count": 0,
        "external_api_used": False,
        "privacy_result": "passed" if privacy_ok else "failed",
        "billing_check_pending": True,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "per_family_results": [],
        "status": "BLOCKED",
        "block_reason": "",
        "stopped_on_first_failure": False,
    }

    if not is_live_allowed(env):
        aggregate["status"] = "BLOCKED_READY_FOR_LIVE_COMPARE"
        aggregate["block_reason"] = f"{LIVE_ENV}_must_equal_YES"
        return aggregate
    if not previews:
        aggregate["status"] = "BLOCKED_NO_REMAINING_FAMILIES"
        aggregate["block_reason"] = "no_remaining_synthetic_package_families_found"
        return aggregate
    if not privacy_ok:
        aggregate["status"] = "BLOCKED_PRIVACY_GATE"
        aggregate["block_reason"] = "vertex_semantic_prompt_privacy_gate_failed"
        return aggregate

    is_real = client_kind == "real_rest"
    post = http_post or _default_http_post
    get_token = token_provider or acquire_google_cloud_access_token
    config = build_vertex_config(env)
    try:
        token = get_token()
        if not str(token or "").strip():
            raise RuntimeError("google_cloud_access_token_unavailable")
    except Exception as exc:  # noqa: BLE001
        error = classify_vertex_provider_error(exc)
        aggregate.update({"status": "FAIL_TOKEN_UNAVAILABLE", "block_reason": "google_cloud_access_token_unavailable", **error})
        return aggregate

    aggregate["external_api_used"] = is_real
    for preview in previews[:MAX_LIVE_CALLS]:
        aggregate["package_families_attempted"] += 1
        if is_real:
            aggregate["live_call_count"] += 1
        result = compare_one_family(
            preview, token=token, http_post=post, config=config, client_kind=client_kind
        )
        aggregate["per_family_results"].append(result)
        aggregate["provider_response_received_count"] += int(bool(result["provider_response_received"]))
        aggregate["schema_validation_pass_count"] += int(bool(result["schema_validation_pass"]))
        aggregate["source_visible_body_preserved_count"] += int(bool(result["source_visible_body_preserved"]))
        aggregate["evidence_anchor_preserved_count"] += int(bool(result["evidence_anchor_preserved"]))
        aggregate["candidate_facts_separated_count"] += int(bool(result["candidate_facts_separated"]))
        aggregate["unknown_values_explicit_count"] += int(bool(result["unknown_values_explicit"]))
        aggregate["uncertainty_flags_visible_count"] += int(bool(result["uncertainty_flags_visible"]))
        aggregate["posted_body_allowed_top_level_keys_only_count"] += int(bool(result["posted_body_allowed_top_level_keys_only"]))
        aggregate["hallucinated_field_count"] += int(result["hallucinated_field_count"])
        aggregate["prompt_token_count_all_calls"] += int(result["prompt_token_count"])
        aggregate["output_token_count_all_calls"] += int(result["output_token_count"])
        aggregate["total_token_count_all_calls"] += int(result["total_token_count"])
        if result["review_required"] is not True:
            aggregate["review_required_all"] = False
        if result["auto_accept"] is not False:
            aggregate["auto_accept_all_false"] = False
        if result["status"] == "PASS":
            aggregate["package_families_passed"] += 1
        else:
            # Stop immediately on first failure; do not attempt remaining families.
            aggregate["stopped_on_first_failure"] = True
            aggregate["status"] = result["status"]
            aggregate["block_reason"] = result["block_reason"]
            return aggregate

    aggregate["status"] = (
        "PASS"
        if aggregate["package_families_passed"] == len(previews) and aggregate["hallucinated_field_count"] == 0
        else "FAIL_CONTRACT_NOT_PRESERVED"
    )
    return aggregate


def _matrix_markdown(agg: dict[str, Any]) -> str:
    header = [
        "# 15P-D remaining-families Vertex semantic live comparison matrix",
        "",
        f"- Status: `{agg['status']}`",
        f"- Provider route: `{agg['provider_route']}` | Model: `{agg['model']}` | Location: `{agg['location']}`",
        f"- Families attempted/passed: `{agg['package_families_attempted']}` / `{agg['package_families_passed']}`",
        f"- Live calls: `{agg['live_call_count']}` (max `{agg['max_live_calls']}`) | "
        f"Total tokens: `{agg['total_token_count_all_calls']}`",
        "",
        "| Family | Status | schema | evidence | unknown | uncertainty | halluc | keys_only | tokens |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = [
        "| {family} | `{status}` | `{schema}` | `{ev}` | `{unk}` | `{unc}` | `{h}` | `{keys}` | `{tok}` |".format(
            family=r["package_family"],
            status=r["status"],
            schema=r["schema_validation_pass"],
            ev=r["evidence_anchor_preserved"],
            unk=r["unknown_values_explicit"],
            unc=r["uncertainty_flags_visible"],
            h=r["hallucinated_field_count"],
            keys=r["posted_body_allowed_top_level_keys_only"],
            tok=r["total_token_count"],
        )
        for r in agg["per_family_results"]
    ]
    footer = ["", "No credentials, tokens, auth headers, or ADC paths are recorded. Synthetic payloads only.", ""]
    return "\n".join(header + rows + footer)


def _implementation_markdown(agg: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-SEMANTIC-PACKAGE-COMPARISON-REMAINING-FAMILIES-LIVE-15P-D",
            "",
            f"- Status: `{agg['status']}`",
            f"- Families attempted: `{agg['package_families_attempted']}` | passed: `{agg['package_families_passed']}`",
            f"- Live call count: `{agg['live_call_count']}` (max {agg['max_live_calls']}; one per family; stop on first failure)",
            f"- Provider responses received: `{agg['provider_response_received_count']}`",
            f"- Provider route: `{agg['provider_route']}` | Model: `{agg['model']}`",
            f"- schema_validation_pass_count: `{agg['schema_validation_pass_count']}`",
            f"- source_visible_body_preserved_count: `{agg['source_visible_body_preserved_count']}`",
            f"- evidence_anchor_preserved_count: `{agg['evidence_anchor_preserved_count']}`",
            f"- candidate_facts_separated_count: `{agg['candidate_facts_separated_count']}`",
            f"- unknown_values_explicit_count: `{agg['unknown_values_explicit_count']}`",
            f"- uncertainty_flags_visible_count: `{agg['uncertainty_flags_visible_count']}`",
            f"- hallucinated_field_count: `{agg['hallucinated_field_count']}`",
            f"- posted_body_allowed_top_level_keys_only_count: `{agg['posted_body_allowed_top_level_keys_only_count']}`",
            f"- Token totals: prompt=`{agg['prompt_token_count_all_calls']}` "
            f"output=`{agg['output_token_count_all_calls']}` total=`{agg['total_token_count_all_calls']}`",
            f"- review_required_all: `{agg['review_required_all']}` | auto_accept_all_false: `{agg['auto_accept_all_false']}` "
            f"| active_written_count: `{agg['active_written_count']}`",
            f"- Privacy result: `{agg['privacy_result']}` | Billing check pending: `{agg['billing_check_pending']}`",
            "",
            "## Safety",
            "",
            "- One live Vertex call per remaining family (3 max); no retries; stop on first failure.",
            "- Each posted body contains only `contents` and `generationConfig` (no MedAI metadata).",
            "- JSON-only, temperature=0, maxOutputTokens<=512; no diagnosis/treatment requested.",
            "- No credentials/tokens/auth headers/ADC paths recorded; sanitized responses only.",
            "- No active MKB writes; output remains review-bound; no auto-accept.",
            "- 15P-C portal result-card live call NOT repeated; 15N-R4 smoke NOT rerun.",
            "",
        ]
    )


def write_reports(agg: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in agg.items() if k != "per_family_results"}
    summary["per_family_status"] = [
        {"package_family": r["package_family"], "status": r["status"]} for r in agg["per_family_results"]
    ]
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RESULTS_JSON.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    MATRIX_MD.write_text(_matrix_markdown(agg), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation_markdown(agg), encoding="utf-8")


def main() -> int:
    agg = run_remaining_families_comparison()
    write_reports(agg)
    print(
        json.dumps(
            {
                "status": agg["status"],
                "package_families_attempted": agg["package_families_attempted"],
                "package_families_passed": agg["package_families_passed"],
                "live_call_count": agg["live_call_count"],
                "provider_response_received_count": agg["provider_response_received_count"],
                "schema_validation_pass_count": agg["schema_validation_pass_count"],
                "hallucinated_field_count": agg["hallucinated_field_count"],
                "posted_body_allowed_top_level_keys_only_count": agg["posted_body_allowed_top_level_keys_only_count"],
                "total_token_count_all_calls": agg["total_token_count_all_calls"],
                "active_written_count": agg["active_written_count"],
                "privacy_result": agg["privacy_result"],
                "billing_check_pending": agg["billing_check_pending"],
            },
            indent=2,
        )
    )
    if agg["status"].startswith("BLOCKED"):
        return 0
    return 0 if agg["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
