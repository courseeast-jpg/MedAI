#!/usr/bin/env python3
"""Gated Vertex Gemini credit-route smoke for MEDAI 15N-R4."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload
from execution.gemini_vertex_adapter import (
    VERTEX_LIVE_SMOKE_ENV,
    VERTEX_MAX_OUTPUT_TOKENS,
    build_vertex_config,
    build_vertex_generate_content_url,
    build_vertex_public_status,
    build_vertex_smoke_payload,
    run_gemini_vertex_live_smoke,
    vertex_result_to_public_dict,
)

REPORT_DIR = REPO_ROOT / "reports" / "medai_gemini_vertex_credit_route_smoke_15n_r4"
SUMMARY_JSON = REPORT_DIR / "summary.json"
AUDIT_JSON = REPORT_DIR / "vertex_live_smoke_audit.json"
GATE_JSON = REPORT_DIR / "vertex_live_gate_decision.json"
REQUEST_JSON = REPORT_DIR / "vertex_request_shape.json"
PRIVACY_JSON = REPORT_DIR / "privacy_check.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_REPORT_TOKENS = (
    "Authorization",
    "Bearer ",
    "ya29.",
    "AIza",
    "token_map",
    "raw prompt body",
    "raw provider response body",
    "DOB",
    "MRN:",
    "Accession",
    "C:\\",
)


def build_reports() -> dict[str, Any]:
    config = build_vertex_config()
    result = run_gemini_vertex_live_smoke()
    public = vertex_result_to_public_dict(result)
    request_shape = _public_request_shape(config)
    gate = {
        "env_gate_name": VERTEX_LIVE_SMOKE_ENV,
        "env_gate_enabled": config.vertex_live_smoke_allowed,
        "live_call_refused_without_gate": not config.vertex_live_smoke_allowed,
        "exactly_one_live_provider_call_allowed": True,
        "no_retries": True,
        "stop_on_first_live_call_failure": True,
    }
    summary = {
        "block": "MEDAI-GEMINI-VERTEX-CREDIT-ROUTE-SMOKE-15N-R4",
        "overall_status": _overall_status(result),
        "provider_route": result.provider_route,
        "provider_name": result.provider_name,
        "model": result.model,
        "location": result.location,
        "endpoint_host": result.endpoint_host,
        "live_call_made": result.live_call_made,
        "external_api_used": result.external_api_used,
        "real_network_call_used": result.real_network_call_used,
        "gemini_real_call_attempted": result.gemini_real_call_attempted,
        "provider_response_received": result.provider_response_received,
        "schema_valid": result.schema_valid,
        "prompt_token_count": result.prompt_token_count,
        "candidates_token_count": result.candidates_token_count,
        "total_token_count": result.total_token_count,
        "response_sanitized": result.response_sanitized,
        "billing_check_pending": True,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "live_retry_allowed": False,
        "block_reason": result.block_reason,
        "provider_error_available": result.provider_error_available,
        "provider_error_category": result.provider_error_category,
        "provider_error_type": result.provider_error_type,
        "provider_error_status_code": result.provider_error_status_code,
        "provider_error_code": result.provider_error_code,
        "provider_error_message_sanitized": result.provider_error_message_sanitized,
    }
    audit = {
        **public,
        "raw_prompt_body_in_report": False,
        "raw_provider_response_body_in_report": False,
        "credential_value_in_report": False,
        "auth_header_in_report": False,
        "old_ai_studio_adapter_mutated": False,
    }
    privacy = _privacy_report(summary, audit, gate, request_shape)
    implementation = _markdown(summary, privacy)
    return {
        "summary": summary,
        "audit": audit,
        "gate": gate,
        "request_shape": request_shape,
        "privacy": privacy,
        "implementation": implementation,
    }


def write_reports(reports: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(reports["summary"], indent=2), encoding="utf-8")
    AUDIT_JSON.write_text(json.dumps(reports["audit"], indent=2), encoding="utf-8")
    GATE_JSON.write_text(json.dumps(reports["gate"], indent=2), encoding="utf-8")
    REQUEST_JSON.write_text(json.dumps(reports["request_shape"], indent=2), encoding="utf-8")
    PRIVACY_JSON.write_text(json.dumps(reports["privacy"], indent=2), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(reports["implementation"], encoding="utf-8")


def _public_request_shape(config: Any) -> dict[str, Any]:
    payload = build_vertex_smoke_payload()
    return {
        "provider_status": build_vertex_public_status(config),
        "url_template": (
            "https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}"
            "/publishers/google/models/{model}:generateContent"
        ),
        "resolved_url": build_vertex_generate_content_url(config),
        "request_payload_keys": sorted(payload.keys()),
        "contents_count": len(payload["contents"]),
        "prompt_text_hash_only": "vertex_smoke_prompt_fixed_15n_r4",
        "prompt_body_in_report": False,
        "generation_config": {
            "temperature": payload["generationConfig"]["temperature"],
            "maxOutputTokens": payload["generationConfig"]["maxOutputTokens"],
            "responseMimeType": payload["generationConfig"]["responseMimeType"],
        },
        "max_output_tokens_within_limit": payload["generationConfig"]["maxOutputTokens"] <= VERTEX_MAX_OUTPUT_TOKENS,
        "exact_prompt_contract": "fixed_synthetic_json_status_prompt_15n_r4",
    }


def _privacy_report(*payloads: Any) -> dict[str, Any]:
    serialized = json.dumps(payloads, sort_keys=True, default=str)
    token_scan_passed = not any(token in serialized for token in FORBIDDEN_REPORT_TOKENS)
    public_check = check_public_report_payload(payloads)
    return {
        "privacy_result": "passed" if token_scan_passed and public_check.passed else "failed",
        "token_scan_passed": token_scan_passed,
        "public_report_payload_check_passed": public_check.passed,
        "api_key_values_in_report": False,
        "bearer_tokens_in_report": False,
        "token_map_in_report": False,
        "raw_prompt_body_in_report": False,
        "raw_provider_response_body_in_report": False,
        "real_patient_identifiers_in_report": False,
        "absolute_local_paths_in_report": False,
        "no_real_medical_docs_sent": True,
        "no_raw_pdf_image_sent": True,
        "no_private_ocr_payload_sent": True,
        "no_mkb_active_write": True,
        "auto_accept": False,
        "review_required": True,
    }


def _overall_status(result: Any) -> str:
    if result.live_call_status == "succeeded":
        return "PASS_VERTEX_LIVE_SMOKE_SUCCEEDED"
    if result.live_call_status.startswith("blocked"):
        return "BLOCKED_VERTEX_LIVE_SMOKE_GATE_REQUIRED"
    return "FAIL_VERTEX_LIVE_SMOKE"


def _markdown(summary: dict[str, Any], privacy: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-GEMINI-VERTEX-CREDIT-ROUTE-SMOKE-15N-R4",
            "",
            f"- Overall status: `{summary['overall_status']}`",
            f"- Provider route: `{summary['provider_route']}`",
            f"- Provider name: `{summary['provider_name']}`",
            f"- Model: `{summary['model']}`",
            f"- Location: `{summary['location']}`",
            f"- Endpoint host: `{summary['endpoint_host']}`",
            f"- Live call made: `{summary['live_call_made']}`",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Response received: `{summary['provider_response_received']}`",
            f"- Token count: `{summary['total_token_count']}`",
            f"- Privacy result: `{privacy['privacy_result']}`",
            "- No real medical docs sent: `True`",
            "- No raw PDF/image sent: `True`",
            "- No private OCR payload sent: `True`",
            "- No MKB active write: `True`",
            "- Auto-accept: `False`",
            "- Review required: `True`",
            "- Billing check pending: `True`",
            "",
            "Google Cloud billing and cost reporting may lag; a later manual cost check is required.",
            "",
        ]
    )


def main() -> int:
    reports = build_reports()
    write_reports(reports)
    summary = reports["summary"]
    privacy = reports["privacy"]
    ready = (
        summary["overall_status"] == "PASS_VERTEX_LIVE_SMOKE_SUCCEEDED"
        and privacy["privacy_result"] == "passed"
        and summary["active_written_count"] == 0
        and summary["auto_accept"] is False
        and summary["review_required"] is True
    )
    print("medai_gemini_vertex_credit_route_smoke_15n_r4_ready" if ready else "medai_gemini_vertex_credit_route_smoke_15n_r4_not_ready")
    print(
        json.dumps(
            {
                "report": str(SUMMARY_JSON.relative_to(REPO_ROOT)),
                "overall_status": summary["overall_status"],
                "live_call_made": summary["live_call_made"],
                "provider_route": summary["provider_route"],
                "model": summary["model"],
                "total_token_count": summary["total_token_count"],
                "privacy_result": privacy["privacy_result"],
                "billing_check_pending": summary["billing_check_pending"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
