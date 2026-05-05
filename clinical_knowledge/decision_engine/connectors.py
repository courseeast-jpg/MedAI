"""Connector stubs for CKA-B03 Decision Engine.

ALL stubs are local-only — no network calls, no external APIs.
Returns deterministic synthetic responses for testing.
"""
from __future__ import annotations

from typing import Dict, Optional

from clinical_knowledge.decision_engine.models import ConnectorRequest, ConnectorResponse

# Connector IDs available in CKA-B03
CONNECTOR_IDS = ["dxgpt_stub", "sage_epilepsy_stub", "patientnotes_ddi_stub"]


def call_connector(request: ConnectorRequest) -> ConnectorResponse:
    """Dispatch to the appropriate stub. Raises ValueError for unknown IDs."""
    dispatch: Dict[str, _StubFn] = {
        "dxgpt_stub": _dxgpt_stub,
        "sage_epilepsy_stub": _sage_epilepsy_stub,
        "patientnotes_ddi_stub": _patientnotes_ddi_stub,
    }
    fn = dispatch.get(request.connector_id)
    if fn is None:
        return ConnectorResponse(
            connector_id=request.connector_id,
            success=False,
            content="",
            confidence=0.0,
            citations=[],
            error=f"Unknown connector: {request.connector_id}",
        )
    if not request.privacy_cleared:
        return ConnectorResponse(
            connector_id=request.connector_id,
            success=False,
            content="",
            confidence=0.0,
            citations=[],
            error="Privacy boundary not cleared — connector call blocked",
        )
    return fn(request)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

_StubFn = type(lambda r: None)


def _dxgpt_stub(request: ConnectorRequest) -> ConnectorResponse:
    """DxGPT differential-diagnosis stub — synthetic response only."""
    specialty = request.specialty
    task = request.task_type

    if task == "medication":
        content = (
            "[STUB dxgpt] Medication reference: consult current formulary for "
            "dosing and interaction guidance. This is a synthetic stub response."
        )
        confidence = 0.72
        citations = ["dxgpt_stub_ref_001"]
    elif task == "diagnosis":
        content = (
            f"[STUB dxgpt] Differential for {specialty} presentation: "
            "consider specialist referral. Synthetic stub — not clinical advice."
        )
        confidence = 0.65
        citations = ["dxgpt_stub_ref_002"]
    else:
        content = (
            f"[STUB dxgpt] General {specialty} knowledge summary. "
            "Synthetic stub response — no real data."
        )
        confidence = 0.55
        citations = []

    return ConnectorResponse(
        connector_id="dxgpt_stub",
        success=True,
        content=content,
        confidence=confidence,
        citations=citations,
    )


def _sage_epilepsy_stub(request: ConnectorRequest) -> ConnectorResponse:
    """SAGE Epilepsy knowledge stub — epilepsy-specific synthetic response."""
    specialty = request.specialty

    if specialty not in ("epilepsy", "neurology"):
        return ConnectorResponse(
            connector_id="sage_epilepsy_stub",
            success=False,
            content="",
            confidence=0.0,
            citations=[],
            error="sage_epilepsy_stub: specialty not in scope (epilepsy/neurology only)",
        )

    content = (
        "[STUB sage_epilepsy] Epilepsy management guidelines reference: "
        "antiepileptic drug selection depends on seizure type, patient profile, "
        "and comorbidities. Consult a neurologist for individualised care. "
        "Synthetic stub — not clinical advice."
    )
    return ConnectorResponse(
        connector_id="sage_epilepsy_stub",
        success=True,
        content=content,
        confidence=0.80,
        citations=["sage_epilepsy_stub_ref_001", "sage_epilepsy_stub_ref_002"],
    )


def _patientnotes_ddi_stub(request: ConnectorRequest) -> ConnectorResponse:
    """PatientNotes DDI check stub — Layer 1 placeholder only, no write gate."""
    if not request.privacy_cleared:
        return ConnectorResponse(
            connector_id="patientnotes_ddi_stub",
            success=False,
            content="",
            confidence=0.0,
            citations=[],
            error="DDI stub: privacy not cleared",
        )

    content = (
        "[STUB patientnotes_ddi] DDI Layer 1 check: no interactions detected "
        "in synthetic dataset. This is a placeholder — no real patient data accessed. "
        "Layer 1 only: score modifier, no write gate."
    )
    return ConnectorResponse(
        connector_id="patientnotes_ddi_stub",
        success=True,
        content=content,
        confidence=0.70,
        citations=["patientnotes_ddi_stub_ref_001"],
    )
