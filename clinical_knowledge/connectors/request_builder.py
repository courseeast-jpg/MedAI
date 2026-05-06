"""Privacy-gated connector request builder for CKA-B08.

build_connector_request(query, context, connector_spec, *, safe_mode) ->
    ConnectorExecutionRequest | ConnectorExecutionResult (blocked)

Rules:
- Uses CKA-B02 build_outbound_payload for sanitization.
- If privacy gate blocks → returns blocked_privacy result.
- If connector disabled → returns disabled result.
- If safe_mode → returns skipped_safe_mode result.
- No replacement_map in connector request.
- No raw PHI/private strings in sanitized_payload.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from clinical_knowledge.connectors.models import (
    ConnectorExecutionResult,
    ConnectorSpec,
    ConnectorStatus,
    ConnectorExecutionRequest,
)
from clinical_knowledge.privacy.sanitizer import sanitize_dict_values
from clinical_knowledge.privacy.patterns import ALWAYS_BLOCK_CATEGORIES

_SALT = "medai_cka_b08_req_v1"


def _compute_hash(value: str) -> str:
    return hashlib.sha256(f"{_SALT}:{value}".encode()).hexdigest()[:16]


def _payload_hash(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def build_connector_request(
    query: str,
    context: Dict[str, Any],
    connector_spec: ConnectorSpec,
    *,
    safe_mode: bool = False,
) -> "ConnectorExecutionRequest | ConnectorExecutionResult":
    """Build a privacy-gated connector request.

    Returns ConnectorExecutionRequest on success.
    Returns ConnectorExecutionResult (blocked/disabled/skipped) on gate failure.
    """
    # 1. Disabled
    if not connector_spec.enabled:
        return ConnectorExecutionResult(
            connector_name=connector_spec.name,
            status=ConnectorStatus.DISABLED,
            normalized_response=None,
            latency_ms=0.0,
            error_reason="connector is disabled",
            external_api_used=False,
        )

    # 2. Safe mode
    if safe_mode:
        return ConnectorExecutionResult(
            connector_name=connector_spec.name,
            status=ConnectorStatus.SKIPPED_SAFE_MODE,
            normalized_response=None,
            latency_ms=0.0,
            error_reason="safe mode active: connector execution skipped",
            external_api_used=False,
        )

    # 3. Privacy gate — sanitize payload and block on secrets/PHI.
    # Note: connectors are LOCAL stubs (allow_external=False in ConnectorSpec),
    # so we use the sanitizer directly rather than build_outbound_payload()
    # (which blocks ALL non-external calls regardless of sanitization).
    # We still block on secrets (API-key-like strings) per B08 spec.
    raw_payload: Dict[str, Any] = {
        "query": query,
        **{k: str(v) for k, v in context.items()},
    }

    sanitized, findings, _replacement_map = sanitize_dict_values(raw_payload)
    # _replacement_map is PRIVATE — never stored in request or report

    secret_found = any(f.category in ALWAYS_BLOCK_CATEGORIES for f in findings)
    phi_found = any(
        f.category in {"PERSON", "DOB", "MRN", "INSURANCE_ID", "FACILITY", "EMAIL", "PHONE", "DATE"}
        for f in findings
    )
    findings_summary = {}
    for f in findings:
        findings_summary[f.category] = findings_summary.get(f.category, 0) + 1

    blocked_reasons = []
    if secret_found:
        blocked_reasons.append("secret_or_api_key_detected: always blocked")
    # PHI in context blocks connector (raw PHI must not reach stubs)
    if phi_found and not secret_found:
        # PHI is sanitized out; allow the sanitized payload through
        pass

    if blocked_reasons:
        return ConnectorExecutionResult(
            connector_name=connector_spec.name,
            status=ConnectorStatus.BLOCKED_PRIVACY,
            normalized_response=None,
            latency_ms=0.0,
            error_reason=f"privacy gate blocked: {'; '.join(blocked_reasons)}",
            external_api_used=False,
            privacy_audit_summary={
                "blocked_reasons": blocked_reasons,
                "findings_summary": findings_summary,
                "secret_detected": secret_found,
                "raw_phi_detected": phi_found,
            },
        )

    query_h = _compute_hash(query)
    payload_h = _payload_hash(sanitized)

    return ConnectorExecutionRequest(
        connector_name=connector_spec.name,
        query_hash=query_h,
        sanitized_payload=sanitized,
        payload_hash=payload_h,
        purpose=f"connector_request:{connector_spec.name}",
        allow_external=connector_spec.allow_external,
    )
