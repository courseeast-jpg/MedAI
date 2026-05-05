"""Deterministic query classifier for CKA-B03.

Keyword-based only — no LLM, no external APIs.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Tuple

from clinical_knowledge.decision_engine.models import (
    QueryClassification,
    QuerySpecialty,
    QueryTaskType,
)

_EPILEPSY_TERMS = {
    "epilepsy", "epileptic", "seizure", "seizures", "convulsion", "convulsions",
    "anticonvulsant", "anticonvulsants", "antiepileptic", "antiepileptics",
    "valproate", "levetiracetam", "lamotrigine", "carbamazepine", "phenytoin",
    "topiramate", "oxcarbazepine", "lacosamide", "perampanel", "brivaracetam",
    "clobazam", "clonazepam", "ethosuximide", "zonisamide", "vigabatrin",
    "felbamate", "gabapentin", "pregabalin",
}

_NEUROLOGY_TERMS = {
    "neurology", "neurological", "neuro", "brain", "spinal", "nerve",
    "stroke", "parkinson", "multiple sclerosis", "ms ", "dementia",
    "alzheimer", "neuropathy", "migraine", "headache", "tremor",
}

_MEDICATION_TERMS = {
    "drug", "medication", "medicine", "dose", "dosage", "dosing",
    "prescription", "prescribe", "mg", "tablet", "capsule", "injection",
    "interaction", "contraindication", "side effect", "adverse",
}

# Immediate refusal: prescription dosing requests
_PRESCRIPTION_DOSING_PATTERNS = [
    re.compile(r"\b(?:prescribe|prescription)\b.*\b(?:dose|dosage|dosing|mg)\b", re.I),
    re.compile(r"\b(?:dose|dosage|dosing)\b.*\b(?:prescribe|prescription)\b", re.I),
    re.compile(r"\bhow\s+much\s+(?:should|can|to)\s+(?:I\s+)?(?:take|give|prescribe)\b", re.I),
    re.compile(r"\bwhat\s+(?:dose|dosage)\s+(?:should|to|can)\b", re.I),
]

_DIAGNOSIS_TERMS = {
    "diagnose", "diagnosis", "differential", "condition", "disease",
    "disorder", "syndrome", "symptoms", "symptom",
}

_DOCUMENT_TERMS = {
    "report", "document", "file", "pdf", "upload", "scan", "image",
    "ocr", "extract", "parse",
}

_SUMMARY_TERMS = {
    "summarize", "summary", "overview", "brief", "explain", "what is",
    "describe", "tell me about",
}

# DDI-relevant: any medication/drug mention alongside a second drug or interaction keyword
_DDI_TRIGGER_TERMS = {"interaction", "contraindication", "combine", "combined", "co-administer"}


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _lower_set(query: str) -> set:
    return set(re.findall(r"[a-z]+", query.lower()))


def _detect_specialty(words: set, query_lower: str) -> Tuple[QuerySpecialty, float]:
    epilepsy_hits = sum(1 for t in _EPILEPSY_TERMS if t in query_lower)
    neuro_hits = sum(1 for t in _NEUROLOGY_TERMS if t in query_lower)

    if epilepsy_hits >= 1:
        conf = min(0.95, 0.6 + epilepsy_hits * 0.1)
        return QuerySpecialty.EPILEPSY, conf
    if neuro_hits >= 1:
        conf = min(0.85, 0.55 + neuro_hits * 0.1)
        return QuerySpecialty.NEUROLOGY, conf
    return QuerySpecialty.UNKNOWN, 0.40


def _detect_task_type(query_lower: str) -> QueryTaskType:
    if any(t in query_lower for t in _DOCUMENT_TERMS):
        return QueryTaskType.DOCUMENT
    if any(t in query_lower for t in _MEDICATION_TERMS):
        return QueryTaskType.MEDICATION
    if any(t in query_lower for t in _DIAGNOSIS_TERMS):
        return QueryTaskType.DIAGNOSIS
    if any(t in query_lower for t in _SUMMARY_TERMS):
        return QueryTaskType.SUMMARY
    return QueryTaskType.GENERAL


def _detect_medication_terms(query_lower: str) -> List[str]:
    found = []
    for term in sorted(_EPILEPSY_TERMS | _MEDICATION_TERMS):
        if term in query_lower:
            found.append(term)
    return found


def _check_prescription_dosing_refusal(query_lower: str) -> Optional[str]:
    for pat in _PRESCRIPTION_DOSING_PATTERNS:
        if pat.search(query_lower):
            return (
                "This system must not provide prescription dosing guidance. "
                "Please consult a qualified clinician or pharmacist."
            )
    return None


def _requires_ddi_check(query_lower: str, med_terms: List[str]) -> bool:
    if any(t in query_lower for t in _DDI_TRIGGER_TERMS):
        return True
    if len(med_terms) >= 2:
        return True
    return False


def classify_query(query: str) -> QueryClassification:
    """Classify a raw query deterministically. Never stores raw query text."""
    q_hash = _query_hash(query)
    q_lower = query.lower()
    words = _lower_set(query)

    refusal_reason = _check_prescription_dosing_refusal(q_lower)

    specialty, confidence = _detect_specialty(words, q_lower)
    task_type = _detect_task_type(q_lower)
    med_terms = _detect_medication_terms(q_lower)
    needs_ddi = _requires_ddi_check(q_lower, med_terms)

    clarification_required = (
        specialty == QuerySpecialty.UNKNOWN
        and task_type == QueryTaskType.GENERAL
        and not refusal_reason
    )

    return QueryClassification(
        raw_query_hash=q_hash,
        specialty=specialty,
        task_type=task_type,
        confidence=confidence,
        requires_ddi_check=needs_ddi,
        medication_terms_detected=med_terms,
        clarification_required=clarification_required,
        refusal_reason=refusal_reason,
    )
