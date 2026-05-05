"""Layer 1 DDI evidence modifier for CKA-B05.

Modifies response scores based on DDI findings.
Does NOT block writes — scoring influence only.
"""
from __future__ import annotations

from clinical_knowledge.medication_safety.models import (
    DDICheckResult,
    DDISeverity,
    Layer1DDIScoreResult,
)

_PENALTIES = {
    DDISeverity.HIGH: 0.40,
    DDISeverity.MEDIUM: 0.20,
    DDISeverity.LOW: 0.05,
    DDISeverity.NONE: 0.0,
    DDISeverity.UNAVAILABLE: 0.0,  # cap instead of subtract
}

_UNAVAILABLE_CAP = 0.50


def apply_ddi_evidence_modifier(
    base_score: float,
    ddi_result: DDICheckResult,
) -> Layer1DDIScoreResult:
    """Apply DDI evidence penalty to base score. Layer 1: score only, no write gate."""
    severity = ddi_result.highest_severity

    if severity == DDISeverity.UNAVAILABLE:
        adjusted = min(base_score, _UNAVAILABLE_CAP)
        penalty = base_score - adjusted
        return Layer1DDIScoreResult(
            base_score=base_score,
            adjusted_score=round(adjusted, 4),
            penalty_applied=round(penalty, 4),
            highest_severity=severity,
            findings=ddi_result.findings,
            safe_public_summary={
                "severity": severity.value,
                "base_score": base_score,
                "adjusted_score": round(adjusted, 4),
                "capped_at": _UNAVAILABLE_CAP,
                "layer": 1,
                "blocks_write": False,
                "synthetic": True,
            },
        )

    penalty = _PENALTIES.get(severity, 0.0)
    adjusted = round(max(0.0, base_score - penalty), 4)

    return Layer1DDIScoreResult(
        base_score=base_score,
        adjusted_score=adjusted,
        penalty_applied=round(penalty, 4),
        highest_severity=severity,
        findings=ddi_result.findings,
        safe_public_summary={
            "severity": severity.value,
            "base_score": base_score,
            "adjusted_score": adjusted,
            "penalty": penalty,
            "layer": 1,
            "blocks_write": False,
            "synthetic": True,
        },
    )
