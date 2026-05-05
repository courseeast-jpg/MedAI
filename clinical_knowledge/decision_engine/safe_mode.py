"""Safe Mode evaluation for CKA-B03 Decision Engine."""
from __future__ import annotations

from typing import List

from clinical_knowledge.decision_engine.models import (
    ConnectorResponse,
    SafeModeState,
    ScoredResponse,
)

SAFE_MODE_PREFIX = "[SAFE MODE — MKB only, no external AI]"


def evaluate_safe_mode(
    connector_responses: List[ConnectorResponse],
    scored_responses: List[ScoredResponse],
    aggregate_confidence: float,
    threshold: float,
    manual_safe_mode: bool = False,
) -> SafeModeState:
    """Determine whether Safe Mode should be active.

    Triggers:
    1. All connector responses failed
    2. aggregate_confidence < threshold (default 0.4)
    3. manual_safe_mode=True
    """
    if manual_safe_mode:
        return SafeModeState(
            active=True,
            reason="Manual safe mode flag set by caller",
            prefix=SAFE_MODE_PREFIX,
            triggered_by_manual_flag=True,
        )

    all_failed = bool(connector_responses) and all(not r.success for r in connector_responses)
    if all_failed:
        return SafeModeState(
            active=True,
            reason="All connectors failed — falling back to MKB-only",
            prefix=SAFE_MODE_PREFIX,
            triggered_by_connector_failure=True,
        )

    if aggregate_confidence < threshold:
        return SafeModeState(
            active=True,
            reason=(
                f"Aggregate confidence {aggregate_confidence:.3f} below "
                f"threshold {threshold:.3f}"
            ),
            prefix=SAFE_MODE_PREFIX,
            triggered_by_low_confidence=True,
        )

    return SafeModeState(
        active=False,
        reason="",
        prefix=SAFE_MODE_PREFIX,
    )


def apply_safe_mode_prefix(response: str, state: SafeModeState) -> str:
    if state.active and not response.startswith(state.prefix):
        return f"{state.prefix} {response}"
    return response
