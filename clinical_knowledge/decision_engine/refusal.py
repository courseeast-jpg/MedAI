"""Refusal and escalation logic for CKA-B03 Decision Engine."""
from __future__ import annotations

from typing import Optional

from clinical_knowledge.decision_engine.models import QueryClassification

_REFUSAL_TEMPLATE = (
    "This system cannot provide {reason_detail}. "
    "Please consult a qualified clinician or pharmacist for personalised medical advice."
)

_CLINICAL_DIAGNOSIS_DISCLAIMER = (
    "This system does not provide clinical diagnoses. "
    "Information provided is for reference only. "
    "Please consult a qualified clinician."
)


def evaluate_refusal(classification: QueryClassification) -> tuple[bool, Optional[str]]:
    """Return (should_refuse, refusal_message).

    Refusal if:
    - classifier already set a refusal_reason (e.g. prescription dosing)
    """
    if classification.refusal_reason:
        msg = _REFUSAL_TEMPLATE.format(reason_detail="prescription dosing guidance")
        return True, msg
    return False, None


def diagnosis_disclaimer() -> str:
    return _CLINICAL_DIAGNOSIS_DISCLAIMER
