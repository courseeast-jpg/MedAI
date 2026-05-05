"""Local-only PatientNotes DDI stub for CKA-B05.

No network calls. No API keys. Synthetic interaction table only.
Synthetic medication names use clearly non-real identifiers.
"""
from __future__ import annotations

from typing import List

from clinical_knowledge.medication_safety.models import (
    DDICheckResult,
    DDICheckStatus,
    DDIFinding,
    DDISeverity,
)

# Synthetic interaction table — fake drug names only, no real medical assertions
_SYNTHETIC_INTERACTIONS: List[tuple] = [
    ("synth_med_alpha", "synth_med_beta", DDISeverity.HIGH,
     "Synthetic mechanism: hypothetical pathway conflict (synthetic only).",
     "Consult a qualified clinician before use."),
    ("synth_med_gamma", "synth_med_delta", DDISeverity.MEDIUM,
     "Synthetic mechanism: hypothetical absorption interference (synthetic only).",
     "Monitor synthetic parameters as directed by qualified clinician."),
    ("synth_med_epsilon", "synth_med_zeta", DDISeverity.LOW,
     "Synthetic mechanism: minor hypothetical effect (synthetic only).",
     "Awareness noted; consult a qualified clinician for guidance."),
]

_UNAVAILABLE_RESULT = DDICheckResult(
    checked=False,
    available=False,
    findings=[],
    highest_severity=DDISeverity.UNAVAILABLE,
    status=DDICheckStatus.UNAVAILABLE,
    safe_public_summary={"status": "unavailable", "synthetic": True},
)


def check_ddi_stub(
    candidate_medication: str,
    active_medications: List[str],
    *,
    mode: str = "normal",
) -> DDICheckResult:
    """Check DDI for candidate medication against active medications.

    No network calls. No API keys. Deterministic local-only.

    Modes: normal | unavailable | force_none | force_low | force_medium | force_high
    """
    if mode == "unavailable":
        return _UNAVAILABLE_RESULT

    if mode == "force_high":
        return _forced_result(
            candidate_medication, DDISeverity.HIGH, DDICheckStatus.HIGH_BLOCKED
        )
    if mode == "force_medium":
        return _forced_result(
            candidate_medication, DDISeverity.MEDIUM, DDICheckStatus.MEDIUM
        )
    if mode == "force_low":
        return _forced_result(
            candidate_medication, DDISeverity.LOW, DDICheckStatus.LOW
        )
    if mode == "force_none":
        return _clear_result()

    # Normal mode: check synthetic table
    findings = _check_table(candidate_medication, active_medications)
    if not findings:
        return _clear_result()

    highest = max(findings, key=lambda f: _severity_rank(f.severity))
    status = _severity_to_status(highest.severity)

    return DDICheckResult(
        checked=True,
        available=True,
        findings=findings,
        highest_severity=highest.severity,
        status=status,
        safe_public_summary={
            "checked": True,
            "highest_severity": highest.severity.value,
            "finding_count": len(findings),
            "synthetic": True,
        },
    )


def _check_table(candidate: str, active: List[str]) -> List[DDIFinding]:
    findings = []
    c_lower = candidate.lower().strip()
    for a_lower in [m.lower().strip() for m in active]:
        for drug_a, drug_b, severity, mechanism, note in _SYNTHETIC_INTERACTIONS:
            if (c_lower == drug_a and a_lower == drug_b) or (
                c_lower == drug_b and a_lower == drug_a
            ):
                findings.append(
                    DDIFinding(
                        drug_a=drug_a,
                        drug_b=drug_b,
                        severity=severity,
                        mechanism=mechanism,
                        management_note=note,
                        source="synthetic_ddi_table_v1",
                        synthetic=True,
                        safe_public_summary={
                            "severity": severity.value,
                            "synthetic": True,
                            "source": "synthetic_ddi_table_v1",
                        },
                    )
                )
    return findings


def _forced_result(
    candidate: str, severity: DDISeverity, status: DDICheckStatus
) -> DDICheckResult:
    finding = DDIFinding(
        drug_a=candidate,
        drug_b="synth_forced_partner",
        severity=severity,
        mechanism=f"Forced synthetic mechanism for {severity.value} (synthetic only).",
        management_note="Consult a qualified clinician.",
        source="synthetic_forced_mode",
        synthetic=True,
        safe_public_summary={"severity": severity.value, "synthetic": True},
    )
    return DDICheckResult(
        checked=True,
        available=True,
        findings=[finding],
        highest_severity=severity,
        status=status,
        safe_public_summary={
            "checked": True,
            "highest_severity": severity.value,
            "finding_count": 1,
            "synthetic": True,
            "forced_mode": True,
        },
    )


def _clear_result() -> DDICheckResult:
    return DDICheckResult(
        checked=True,
        available=True,
        findings=[],
        highest_severity=DDISeverity.NONE,
        status=DDICheckStatus.CLEAR,
        safe_public_summary={"checked": True, "highest_severity": "none", "synthetic": True},
    )


def _severity_rank(s: DDISeverity) -> int:
    return {
        DDISeverity.NONE: 0,
        DDISeverity.LOW: 1,
        DDISeverity.MEDIUM: 2,
        DDISeverity.HIGH: 3,
        DDISeverity.UNAVAILABLE: -1,
    }.get(s, 0)


def _severity_to_status(s: DDISeverity) -> DDICheckStatus:
    return {
        DDISeverity.NONE: DDICheckStatus.CLEAR,
        DDISeverity.LOW: DDICheckStatus.LOW,
        DDISeverity.MEDIUM: DDICheckStatus.MEDIUM,
        DDISeverity.HIGH: DDICheckStatus.HIGH_BLOCKED,
        DDISeverity.UNAVAILABLE: DDICheckStatus.UNAVAILABLE,
    }.get(s, DDICheckStatus.CLEAR)
