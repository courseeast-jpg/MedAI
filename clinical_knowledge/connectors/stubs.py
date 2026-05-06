"""Local deterministic connector stubs for CKA-B08.

Stubs:
1. dxgpt_stub       — synthetic diagnosis-support facts
2. sage_epilepsy_stub — synthetic epilepsy-support facts
3. patientnotes_ddi_stub — synthetic DDI facts (wraps B05 logic)
4. generic_stub     — generic synthetic facts

Rules:
- No network calls. No external APIs. No clinical advice.
- All source_kind="connector_stub".
- synthetic=True on all fact entries.
- Simulation modes: success / timeout / error / malformed_response /
  low_confidence / contradiction / privacy_blocked.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from clinical_knowledge.connectors.models import SimulationMode

# ---------------------------------------------------------------------------
# Shared synthetic fact builder helpers
# ---------------------------------------------------------------------------

def _base_response(
    connector_name: str,
    facts: list,
    citations: list,
    confidence: float,
) -> Dict[str, Any]:
    return {
        "connector_name": connector_name,
        "source_kind": "connector_stub",
        "facts": facts,
        "citations": citations,
        "confidence": confidence,
        "synthetic": True,
    }


def _synthetic_fact(
    fact_type: str,
    entity_text: str,
    structured: Dict[str, Any],
    specialty: str = "general",
    confidence: float = 0.85,
) -> Dict[str, Any]:
    return {
        "fact_type": fact_type,
        "entity_text": entity_text,
        "structured": structured,
        "specialty": specialty,
        "confidence": confidence,
        "synthetic": True,
    }


# ---------------------------------------------------------------------------
# dxgpt_stub
# ---------------------------------------------------------------------------

def call_dxgpt_stub(
    request_payload: Dict[str, Any],
    simulation_mode: Optional[SimulationMode] = None,
) -> Dict[str, Any]:
    """Deterministic dxgpt_stub response.

    Returns structured synthetic diagnosis-support facts.
    No medical advice. No dosing. No real patient data.
    """
    mode = simulation_mode or SimulationMode.SUCCESS

    if mode == SimulationMode.TIMEOUT:
        return {"_stub_error": "timeout", "connector_name": "dxgpt_stub"}
    if mode == SimulationMode.ERROR:
        return {"_stub_error": "internal_error", "connector_name": "dxgpt_stub"}
    if mode == SimulationMode.MALFORMED_RESPONSE:
        return {"raw_garbage": "not structured", "connector_name": "dxgpt_stub"}
    if mode == SimulationMode.PRIVACY_BLOCKED:
        return {"_stub_error": "privacy_blocked", "connector_name": "dxgpt_stub"}

    confidence = 0.55 if mode == SimulationMode.LOW_CONFIDENCE else 0.88

    facts = [
        _synthetic_fact(
            fact_type="diagnosis",
            entity_text="synthetic condition alpha",
            structured={"icd_concept": "SYNTHETIC-ICD-001", "synthetic_note": "test-only"},
            specialty="general",
            confidence=confidence,
        )
    ]

    if mode == SimulationMode.CONTRADICTION:
        facts.append(_synthetic_fact(
            fact_type="diagnosis",
            entity_text="synthetic condition alpha",
            structured={"icd_concept": "SYNTHETIC-ICD-CONFLICT-001", "synthetic_note": "test-only"},
            specialty="general",
            confidence=confidence,
        ))

    citations = [
        {"title": "Synthetic Reference A", "source": "synthetic_pub_test_only", "url": ""}
    ]

    return _base_response("dxgpt_stub", facts, citations, confidence)


# ---------------------------------------------------------------------------
# sage_epilepsy_stub
# ---------------------------------------------------------------------------

def call_sage_epilepsy_stub(
    request_payload: Dict[str, Any],
    simulation_mode: Optional[SimulationMode] = None,
) -> Dict[str, Any]:
    """Deterministic sage_epilepsy_stub response.

    Returns structured synthetic epilepsy-support facts.
    No medication instructions. No dosing advice.
    """
    mode = simulation_mode or SimulationMode.SUCCESS

    if mode == SimulationMode.TIMEOUT:
        return {"_stub_error": "timeout", "connector_name": "sage_epilepsy_stub"}
    if mode == SimulationMode.ERROR:
        return {"_stub_error": "internal_error", "connector_name": "sage_epilepsy_stub"}
    if mode == SimulationMode.MALFORMED_RESPONSE:
        return {"raw_garbage": "not structured", "connector_name": "sage_epilepsy_stub"}
    if mode == SimulationMode.PRIVACY_BLOCKED:
        return {"_stub_error": "privacy_blocked", "connector_name": "sage_epilepsy_stub"}

    confidence = 0.50 if mode == SimulationMode.LOW_CONFIDENCE else 0.82

    facts = [
        _synthetic_fact(
            fact_type="diagnosis",
            entity_text="synthetic condition alpha",
            structured={"icd_concept": "SYNTHETIC-ICD-001", "synthetic_note": "test-only"},
            specialty="general",  # matches dxgpt_stub for agreement grouping
            confidence=confidence,
        ),
        _synthetic_fact(
            fact_type="condition_category",
            entity_text="synthetic epilepsy support fact",
            structured={"category": "SYNTHETIC-NEURO-CATEGORY", "synthetic_note": "test-only"},
            specialty="neurology",
            confidence=confidence,
        ),
    ]

    if mode == SimulationMode.CONTRADICTION:
        facts[0] = _synthetic_fact(
            fact_type="diagnosis",
            entity_text="synthetic condition alpha",
            structured={"icd_concept": "SYNTHETIC-ICD-CONFLICT-002", "synthetic_note": "test-only"},
            specialty="general",  # must match to form a contradiction on same key
            confidence=confidence,
        )

    citations = [
        {"title": "Synthetic Epilepsy Reference", "source": "synthetic_neuro_test_only", "url": ""}
    ]

    return _base_response("sage_epilepsy_stub", facts, citations, confidence)


# ---------------------------------------------------------------------------
# patientnotes_ddi_stub
# ---------------------------------------------------------------------------

def call_patientnotes_ddi_stub(
    request_payload: Dict[str, Any],
    simulation_mode: Optional[SimulationMode] = None,
) -> Dict[str, Any]:
    """Deterministic patientnotes_ddi_stub response.

    Returns structured synthetic DDI facts only.
    No real PatientNotes API. No medication advice.
    """
    mode = simulation_mode or SimulationMode.SUCCESS

    if mode == SimulationMode.TIMEOUT:
        return {"_stub_error": "timeout", "connector_name": "patientnotes_ddi_stub"}
    if mode == SimulationMode.ERROR:
        return {"_stub_error": "internal_error", "connector_name": "patientnotes_ddi_stub"}
    if mode == SimulationMode.MALFORMED_RESPONSE:
        return {"raw_garbage": "not structured", "connector_name": "patientnotes_ddi_stub"}
    if mode == SimulationMode.PRIVACY_BLOCKED:
        return {"_stub_error": "privacy_blocked", "connector_name": "patientnotes_ddi_stub"}

    confidence = 0.52 if mode == SimulationMode.LOW_CONFIDENCE else 0.79

    facts = [
        _synthetic_fact(
            fact_type="ddi_check",
            entity_text="synthetic medication beta",
            structured={
                "ddi_status": "synthetic_interaction_detected",
                "severity": "synthetic_moderate",
                "pair": ["synthetic_med_A", "synthetic_med_B"],
                "synthetic_note": "test-only synthetic ddi check only",
            },
            specialty="pharmacology",
            confidence=confidence,
        )
    ]

    citations = [
        {"title": "Synthetic DDI Reference", "source": "synthetic_pharm_test_only", "url": ""}
    ]

    return _base_response("patientnotes_ddi_stub", facts, citations, confidence)


# ---------------------------------------------------------------------------
# generic_stub
# ---------------------------------------------------------------------------

def call_generic_stub(
    request_payload: Dict[str, Any],
    simulation_mode: Optional[SimulationMode] = None,
) -> Dict[str, Any]:
    """Deterministic generic_stub response."""
    mode = simulation_mode or SimulationMode.SUCCESS

    if mode == SimulationMode.TIMEOUT:
        return {"_stub_error": "timeout", "connector_name": "generic_stub"}
    if mode == SimulationMode.ERROR:
        return {"_stub_error": "internal_error", "connector_name": "generic_stub"}
    if mode == SimulationMode.MALFORMED_RESPONSE:
        return {"raw_garbage": "not structured", "connector_name": "generic_stub"}

    confidence = 0.60 if mode == SimulationMode.LOW_CONFIDENCE else 0.70

    facts = [
        _synthetic_fact(
            fact_type="generic",
            entity_text="synthetic generic fact",
            structured={"value": "SYNTHETIC-GENERIC-001", "synthetic_note": "test-only"},
            specialty="general",
            confidence=confidence,
        )
    ]

    return _base_response("generic_stub", facts, [], confidence)
