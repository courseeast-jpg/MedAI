#!/usr/bin/env python3
"""MEDAI-VERTEX-SEMANTIC-PACKAGE-COMPARISON-LIVE-15P-C.

Corrected single-live-call Vertex semantic-package comparison. 15P-B failed with
HTTP 400 request_config_rejected because the posted body included MedAI metadata
top-level keys. 15P-C posts ONLY the Vertex-accepted keys:

    {"contents": [...], "generationConfig": {...}}

It runs EXACTLY ONE live Vertex call against one synthetic/redacted portal
result-card fixture, then validates whether the live JSON preserves the 15P-A
contract and 15O review guarantees.

Hard rules:
* Refuses unless MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED=YES.
* Exactly one live call, no retries, no loop, no background calls.
* Synthetic/redacted payload only; JSON-only; temperature=0; maxOutputTokens<=512.
* Posted body top-level keys are restricted to {contents, generationConfig}.
* Never logs credentials, tokens, auth headers, API keys, or ADC paths.
* No active MKB writes; review_required=true; auto_accept=false.
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

LIVE_COMPARE_ENV = "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED"
PREFERRED_PACKAGE_FAMILY = "portal_result_cards"
MAX_OUTPUT_TOKENS = 512
ALLOWED_TOP_LEVEL_KEYS = {"contents", "generationConfig"}

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_package_comparison_live_15p_c"
SUMMARY_JSON = REPORT_DIR / "summary.json"
RESULT_JSON = REPORT_DIR / "live_comparison_result.json"
MATRIX_MD = REPORT_DIR / "live_comparison_matrix.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"


def is_live_compare_allowed(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(LIVE_COMPARE_ENV) or "").strip().upper() == "YES"


def select_comparison_preview() -> RunReviewPackagePreview:
    """Pick the shortest synthetic portal result-card fixture (same as 15P-B)."""
    previews = build_run_review_package_previews()
    portal = [p for p in previews if p.package_family == PREFERRED_PACKAGE_FAMILY]
    candidates = portal or previews
    return min(candidates, key=lambda p: len(p.source_visible_body))


def build_vertex_api_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip all MedAI metadata; keep only Vertex generateContent keys."""
    return {
        "contents": payload["contents"],
        "generationConfig": payload["generationConfig"],
    }


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


def run_vertex_semantic_live_comparison(
    *,
    environ: Mapping[str, str] | None = None,
    token_provider: Callable[[], str] | None = None,
    http_post: Callable[[str, dict[str, Any], str], dict[str, Any]] | None = None,
    client_kind: str = "real_rest",
) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    preview = select_comparison_preview()
    payload = build_vertex_semantic_request_payload(preview)
    payload["generationConfig"]["maxOutputTokens"] = MAX_OUTPUT_TOKENS
    payload["generationConfig"]["temperature"] = 0
    prompt = str(payload["contents"][0]["parts"][0]["text"])
    privacy = prompt_privacy_check(prompt)
    vertex_api_body = build_vertex_api_body(payload)
    posted_keys = sorted(vertex_api_body.keys())
    posted_body_allowed_top_level_keys_only = set(posted_keys) <= ALLOWED_TOP_LEVEL_KEYS

    base: dict[str, Any] = {
        "block": "MEDAI-VERTEX-SEMANTIC-PACKAGE-COMPARISON-LIVE-15P-C",
        "provider_route": VERTEX_PROVIDER_ROUTE,
        "provider_name": VERTEX_PROVIDER_NAME,
        "model": VERTEX_MODEL,
        "location": VERTEX_LOCATION,
        "endpoint_host": VERTEX_ENDPOINT_HOST,
        "package_family": preview.package_family,
        "provider_client_kind": client_kind,
        "live_call_made": False,
        "provider_response_received": False,
        "external_api_used": False,
        "real_network_call_used": False,
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
        "posted_body_top_level_keys": posted_keys,
        "posted_body_allowed_top_level_keys_only": posted_body_allowed_top_level_keys_only,
        "sanitized_findings": [],
        "validation_errors": [],
        "review_required": True,
        "auto_accept": False,
        "active_written_count": 0,
        "privacy_result": privacy["privacy_result"],
        "billing_check_pending": True,
        "status": "BLOCKED",
        "block_reason": "",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "single_live_call_only": True,
        "prior_15p_b_failure_summary": "15P-B failed HTTP 400 request_config_rejected (MedAI metadata keys posted); fixed here by posting only contents+generationConfig.",
    }

    if not is_live_compare_allowed(env):
        base["status"] = "BLOCKED_READY_FOR_LIVE_COMPARE"
        base["block_reason"] = f"{LIVE_COMPARE_ENV}_must_equal_YES"
        return base
    if privacy["privacy_result"] != "passed":
        base["status"] = "BLOCKED_PRIVACY_GATE"
        base["block_reason"] = "vertex_semantic_prompt_privacy_gate_failed"
        return base
    if not posted_body_allowed_top_level_keys_only:
        base["status"] = "BLOCKED_INVALID_REQUEST_SHAPE"
        base["block_reason"] = "vertex_post_body_has_forbidden_top_level_keys"
        return base

    is_real = client_kind == "real_rest"
    post = http_post or _default_http_post
    get_token = token_provider or acquire_google_cloud_access_token
    try:
        token = get_token()
        if not str(token or "").strip():
            raise RuntimeError("google_cloud_access_token_unavailable")
        config = build_vertex_config(env)
        response = post(build_vertex_generate_content_url(config), vertex_api_body, token)
    except Exception as exc:  # noqa: BLE001 - sanitized below
        error = classify_vertex_provider_error(exc)
        base.update(
            {
                "live_call_made": is_real,
                "external_api_used": is_real,
                "real_network_call_used": is_real,
                "status": "FAIL_LIVE_CALL",
                "block_reason": "vertex_semantic_live_call_failed",
                **error,
            }
        )
        return base

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

    base["live_call_made"] = is_real
    base["external_api_used"] = is_real
    base["real_network_call_used"] = is_real
    base["provider_response_received"] = True
    base["prompt_token_count"] = int(usage.get("promptTokenCount") or 0)
    base["output_token_count"] = int(usage.get("candidatesTokenCount") or 0)
    base["total_token_count"] = int(usage.get("totalTokenCount") or 0)

    if not parsed:
        base["status"] = "FAIL_RESPONSE_NOT_JSON"
        base["block_reason"] = "vertex_response_not_valid_json"
        return base

    schema_valid, errors = validate_vertex_semantic_response(parsed, preview)
    hallucinated = _count_hallucinated(parsed, preview)
    findings = parsed.get("semantic_findings") if isinstance(parsed.get("semantic_findings"), list) else []

    base["schema_validation_pass"] = bool(schema_valid)
    base["validation_errors"] = list(errors)
    base["hallucinated_field_count"] = hallucinated
    base["sanitized_findings"] = _sanitize_findings(findings)
    base["source_visible_body_preserved"] = parsed.get("package_family") == preview.package_family
    base["evidence_anchor_preserved"] = bool(preview.evidence_anchors) and all(
        str(f.get("evidence_text", "")).strip() for f in findings if isinstance(f, Mapping)
    )
    base["candidate_facts_separated"] = bool(preview.candidate_facts) and bool(findings)
    base["unknown_values_explicit"] = all(
        isinstance(f, Mapping) and "unknown_value" in f for f in findings
    ) and bool(findings)
    base["uncertainty_flags_visible"] = all(
        isinstance(f, Mapping) and "uncertainty" in f for f in findings
    ) and bool(findings)
    base["review_required"] = parsed.get("review_required") is True
    base["auto_accept"] = parsed.get("auto_accept") is True

    passed = (
        schema_valid
        and hallucinated == 0
        and base["review_required"] is True
        and base["auto_accept"] is False
        and base["active_written_count"] == 0
        and base["evidence_anchor_preserved"]
        and base["unknown_values_explicit"]
        and base["uncertainty_flags_visible"]
        and base["source_visible_body_preserved"]
    )
    if base["auto_accept"] is True:
        base["status"] = "FAIL_AUTO_ACCEPT"
        base["block_reason"] = "vertex_response_set_auto_accept"
    elif hallucinated > 0:
        base["status"] = "FAIL_HALLUCINATED_FIELDS"
        base["block_reason"] = "vertex_response_added_unsupported_fields"
    elif not schema_valid:
        base["status"] = "FAIL_SCHEMA_INVALID"
        base["block_reason"] = "vertex_response_schema_invalid"
    else:
        base["status"] = "PASS" if passed else "FAIL_CONTRACT_NOT_PRESERVED"
        base["block_reason"] = "" if passed else "vertex_response_contract_not_preserved"
    return base


def _matrix_markdown(result: dict[str, Any]) -> str:
    rows = [
        ("live_call_made", result["live_call_made"]),
        ("provider_response_received", result["provider_response_received"]),
        ("posted_body_allowed_top_level_keys_only", result["posted_body_allowed_top_level_keys_only"]),
        ("schema_validation_pass", result["schema_validation_pass"]),
        ("source_visible_body_preserved", result["source_visible_body_preserved"]),
        ("evidence_anchor_preserved", result["evidence_anchor_preserved"]),
        ("candidate_facts_separated", result["candidate_facts_separated"]),
        ("unknown_values_explicit", result["unknown_values_explicit"]),
        ("uncertainty_flags_visible", result["uncertainty_flags_visible"]),
        ("hallucinated_field_count", result["hallucinated_field_count"]),
        ("review_required", result["review_required"]),
        ("auto_accept", result["auto_accept"]),
        ("active_written_count", result["active_written_count"]),
        ("external_api_used", result["external_api_used"]),
        ("total_token_count", result["total_token_count"]),
        ("privacy_result", result["privacy_result"]),
        ("billing_check_pending", result["billing_check_pending"]),
    ]
    return "\n".join(
        [
            "# 15P-C Vertex semantic live comparison matrix",
            "",
            f"- Status: `{result['status']}`",
            f"- Provider route: `{result['provider_route']}` | Model: `{result['model']}` | Location: `{result['location']}`",
            f"- Package family: `{result['package_family']}`",
            f"- Posted top-level keys: `{result['posted_body_top_level_keys']}`",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *[f"| {name} | `{value}` |" for name, value in rows],
            "",
            "No credentials, tokens, auth headers, or ADC paths are recorded. Synthetic payload only.",
            "",
        ]
    )


def _implementation_markdown(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-SEMANTIC-PACKAGE-COMPARISON-LIVE-15P-C",
            "",
            f"- Status: `{result['status']}`",
            f"- Live call made: `{result['live_call_made']}` (exactly one; no retries)",
            f"- Provider response received: `{result['provider_response_received']}`",
            f"- Posted body allowed-keys-only: `{result['posted_body_allowed_top_level_keys_only']}` "
            f"(keys: `{result['posted_body_top_level_keys']}`)",
            f"- Provider route: `{result['provider_route']}` | Model: `{result['model']}`",
            f"- Schema validation pass: `{result['schema_validation_pass']}`",
            f"- Hallucinated field count: `{result['hallucinated_field_count']}`",
            f"- Token counts: prompt=`{result['prompt_token_count']}` "
            f"output=`{result['output_token_count']}` total=`{result['total_token_count']}`",
            f"- Review required: `{result['review_required']}` | Auto-accept: `{result['auto_accept']}` "
            f"| Active written count: `{result['active_written_count']}`",
            f"- Privacy result: `{result['privacy_result']}` | Billing check pending: `{result['billing_check_pending']}`",
            "",
            "## 15P-B failure (sanitized, prior block)",
            "",
            f"- {result['prior_15p_b_failure_summary']}",
            "",
            "## Safety",
            "",
            "- Exactly one live Vertex call against one synthetic redacted portal result-card fixture.",
            "- Posted body contains only `contents` and `generationConfig` (no MedAI metadata).",
            "- JSON-only, temperature=0, maxOutputTokens<=512; no diagnosis/treatment requested.",
            "- No credentials/tokens/auth headers/ADC paths recorded; sanitized response only.",
            "- No active MKB writes; output remains review-bound; no auto-accept.",
            "",
        ]
    )


def write_reports(result: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items() if key != "sanitized_findings"}
    summary["sanitized_finding_count"] = len(result.get("sanitized_findings", []))
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RESULT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    MATRIX_MD.write_text(_matrix_markdown(result), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation_markdown(result), encoding="utf-8")


def main() -> int:
    result = run_vertex_semantic_live_comparison()
    write_reports(result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "live_call_made": result["live_call_made"],
                "provider_response_received": result["provider_response_received"],
                "provider_route": result["provider_route"],
                "model": result["model"],
                "posted_body_allowed_top_level_keys_only": result["posted_body_allowed_top_level_keys_only"],
                "schema_validation_pass": result["schema_validation_pass"],
                "hallucinated_field_count": result["hallucinated_field_count"],
                "total_token_count": result["total_token_count"],
                "review_required": result["review_required"],
                "auto_accept": result["auto_accept"],
                "active_written_count": result["active_written_count"],
                "privacy_result": result["privacy_result"],
                "billing_check_pending": result["billing_check_pending"],
            },
            indent=2,
        )
    )
    if result["status"].startswith("BLOCKED"):
        return 0
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
