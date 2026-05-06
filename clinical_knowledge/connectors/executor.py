"""Connector executor for CKA-B08.

execute_connectors(requests, registry, *, simulation_mode, safe_mode)
    -> list[ConnectorExecutionResult]

Rules:
- Deterministic sequential execution.
- Timeout simulation produces status=timeout without long sleep.
- Privacy-blocked requests must not execute stubs.
- Skipped-safe-mode requests must not execute stubs.
- Every result has external_api_used=False.
- Malformed raw responses normalize to malformed_response status.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from clinical_knowledge.connectors.models import (
    ConnectorExecutionRequest,
    ConnectorExecutionResult,
    ConnectorKind,
    ConnectorStatus,
    SimulationMode,
)
from clinical_knowledge.connectors.normalizer import normalize_connector_response
from clinical_knowledge.connectors.registry import ConnectorRegistry
from clinical_knowledge.connectors.stubs import (
    call_dxgpt_stub,
    call_generic_stub,
    call_patientnotes_ddi_stub,
    call_sage_epilepsy_stub,
)

_STUB_DISPATCH = {
    ConnectorKind.DXGPT_STUB: call_dxgpt_stub,
    ConnectorKind.SAGE_EPILEPSY_STUB: call_sage_epilepsy_stub,
    ConnectorKind.PATIENTNOTES_DDI_STUB: call_patientnotes_ddi_stub,
    ConnectorKind.GENERIC_STUB: call_generic_stub,
}


def _execute_one(
    request: ConnectorExecutionRequest,
    registry: ConnectorRegistry,
    simulation_mode: Optional[SimulationMode],
) -> ConnectorExecutionResult:
    """Execute a single connector request against its stub."""
    spec = registry.get(request.connector_name)
    if spec is None:
        return ConnectorExecutionResult(
            connector_name=request.connector_name,
            status=ConnectorStatus.ERROR,
            normalized_response=None,
            latency_ms=0.0,
            error_reason="connector not found in registry",
            external_api_used=False,
        )

    stub_fn = _STUB_DISPATCH.get(spec.kind)
    if stub_fn is None:
        return ConnectorExecutionResult(
            connector_name=request.connector_name,
            status=ConnectorStatus.ERROR,
            normalized_response=None,
            latency_ms=0.0,
            error_reason=f"no stub dispatch for kind={spec.kind}",
            external_api_used=False,
        )

    t0 = time.monotonic()
    raw = stub_fn(request.sanitized_payload, simulation_mode)
    latency_ms = (time.monotonic() - t0) * 1000.0

    # Handle stub errors
    stub_error = raw.get("_stub_error") if isinstance(raw, dict) else None

    if stub_error == "timeout":
        return ConnectorExecutionResult(
            connector_name=request.connector_name,
            status=ConnectorStatus.TIMEOUT,
            normalized_response=None,
            latency_ms=latency_ms,
            error_reason="connector timed out (simulated)",
            external_api_used=False,
        )
    if stub_error == "internal_error":
        return ConnectorExecutionResult(
            connector_name=request.connector_name,
            status=ConnectorStatus.ERROR,
            normalized_response=None,
            latency_ms=latency_ms,
            error_reason="connector internal error (simulated)",
            external_api_used=False,
        )
    if stub_error == "privacy_blocked":
        return ConnectorExecutionResult(
            connector_name=request.connector_name,
            status=ConnectorStatus.BLOCKED_PRIVACY,
            normalized_response=None,
            latency_ms=latency_ms,
            error_reason="privacy blocked (simulated)",
            external_api_used=False,
        )

    # Normalize response
    normalized = normalize_connector_response(raw, spec)
    if normalized is None:
        return ConnectorExecutionResult(
            connector_name=request.connector_name,
            status=ConnectorStatus.MALFORMED_RESPONSE,
            normalized_response=None,
            latency_ms=latency_ms,
            error_reason="response failed normalization (malformed or missing required fields)",
            external_api_used=False,
        )

    return ConnectorExecutionResult(
        connector_name=request.connector_name,
        status=ConnectorStatus.SUCCESS,
        normalized_response=normalized,
        latency_ms=latency_ms,
        error_reason=None,
        external_api_used=False,
    )


def execute_connectors(
    requests: List[Union[ConnectorExecutionRequest, ConnectorExecutionResult]],
    registry: ConnectorRegistry,
    *,
    simulation_mode: Optional[SimulationMode] = None,
    safe_mode: bool = False,
) -> List[ConnectorExecutionResult]:
    """Execute a list of connector requests.

    Items that are already ConnectorExecutionResult (blocked/disabled/skipped)
    are passed through unchanged. Only ConnectorExecutionRequest items are executed.
    """
    results: List[ConnectorExecutionResult] = []

    for item in requests:
        if isinstance(item, ConnectorExecutionResult):
            # Already resolved (blocked_privacy / disabled / skipped_safe_mode)
            results.append(item)
            continue

        # Should not execute if item is a request but safe_mode is active
        # (safe_mode should have been caught in request_builder, but double-check)
        if safe_mode:
            results.append(ConnectorExecutionResult(
                connector_name=item.connector_name,
                status=ConnectorStatus.SKIPPED_SAFE_MODE,
                normalized_response=None,
                latency_ms=0.0,
                error_reason="safe mode active: connector execution skipped",
                external_api_used=False,
            ))
            continue

        result = _execute_one(item, registry, simulation_mode)
        results.append(result)

    return results
