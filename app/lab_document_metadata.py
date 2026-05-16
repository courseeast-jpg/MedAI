"""Metadata-only lab document labels for local Run & Review output.

This module does not parse lab values, interpret clinical meaning, adjust
confidence, or change validation outcomes. It only creates operator-facing
labels and review reason text from already available local metadata.
"""

from __future__ import annotations

import re
from typing import Any


LAB_RESULT_LABEL = "Lab result"
URINALYSIS_LABEL = "Urinalysis"
TREATMENT_PLAN_LABEL = "Treatment plan"
MEDICATION_PLAN_LABEL = "Medication plan"
UNKNOWN_DOCUMENT_LABEL = "Unknown"
TEXT_QUALITY_NOT_CHECKED = "Not checked"

_URINALYSIS_TERMS = (
    "urinalysis",
    "urine",
    "specific gravity",
    "leukocyte esterase",
    "nitrite",
    "ketones",
)
_URINALYSIS_SHORT_TERMS = ("ph", "protein", "glucose", "blood", "rbc", "wbc")
_LAB_TERMS = (
    "laboratory report",
    "laboratory result",
    "laboratory results",
    "lab results",
    "lab result",
    "test results",
    "test result",
    "patient result",
    "reference range",
    "reference interval",
    "result",
    "result value",
    "value",
    "flag",
    "units",
    "unit",
    "specimen",
    "specimen type",
    "collection date",
    "collected",
    "received",
    "reported",
    "accession",
    "ordering provider",
    "component",
    "analyte",
    "abnormal",
)
_LAB_PANEL_TERMS = (
    "cbc",
    "cmp",
    "comprehensive metabolic panel",
    "basic metabolic panel",
    "lipid panel",
)
_LAB_TEST_TERMS = (
    "hemoglobin",
    "hematocrit",
    "platelet",
    "glucose",
    "creatinine",
    "sodium",
    "potassium",
    "chloride",
    "calcium",
    "albumin",
    "bilirubin",
    "cholesterol",
    "triglycerides",
)
_LAB_SHORT_TERMS = ("wbc", "rbc", "hgb", "plt", "alt", "ast", "bun", "hdl", "ldl", "tsh")
_RU_LAB_TERMS = (
    "анализ",
    "результат",
    "результаты",
    "лабораторное исследование",
    "референсные значения",
    "норма",
    "единицы",
    "показатель",
    "значение",
    "биохимия",
    "общий анализ крови",
    "кровь",
    "гемоглобин",
    "лейкоциты",
    "эритроциты",
    "тромбоциты",
    "глюкоза",
    "креатинин",
    "мочевина",
    "билирубин",
    "холестерин",
)
_RU_LAB_SHORT_TERMS = ("оак", "алт", "аст")
_RU_URINALYSIS_TERMS = (
    "анализ мочи",
    "общий анализ мочи",
    "моча",
    "удельный вес",
    "белок",
    "глюкоза",
    "кетоны",
    "нитриты",
    "лейкоциты",
    "эритроциты",
    "эпителий",
    "бактерии",
)
_RU_URINALYSIS_SHORT_TERMS = ("оам", "ph")
_RU_TREATMENT_TERMS = (
    "схема лечения",
    "план лечения",
    "лечение",
    "курс",
    "дней",
)
_RU_MEDICATION_TERMS = (
    "назначение",
    "назначения",
    "препарат",
    "лекарство",
    "дозировка",
    "доза",
    "принимать",
    "утром",
    "вечером",
    "мг",
    "мл",
    "таблетка",
    "капсула",
    "инъекция",
)

_DOCUMENT_TYPE_MAP = {
    "lab_report": LAB_RESULT_LABEL,
    "lab result": LAB_RESULT_LABEL,
    "lab_result": LAB_RESULT_LABEL,
    "urinalysis": URINALYSIS_LABEL,
    "ua": URINALYSIS_LABEL,
    "treatment plan": TREATMENT_PLAN_LABEL,
    "treatment_plan": TREATMENT_PLAN_LABEL,
    "medication plan": MEDICATION_PLAN_LABEL,
    "medication_plan": MEDICATION_PLAN_LABEL,
    "generic": UNKNOWN_DOCUMENT_LABEL,
    "unknown": UNKNOWN_DOCUMENT_LABEL,
    "": UNKNOWN_DOCUMENT_LABEL,
}

_TEXT_QUALITY_MAP = {
    "readable_native": "Native text",
    "ocr_fallback_applied": "OCR fallback used",
    "unreadable_after_ocr": "Low quality",
    "empty": "No text found",
    "good": "Good",
    "usable_with_review": "Usable with review",
    "poor_ocr": "Low quality",
    "unknown": TEXT_QUALITY_NOT_CHECKED,
    "": TEXT_QUALITY_NOT_CHECKED,
}


def classify_lab_document_type(text: str | None) -> str:
    """Return a conservative display label from lexical lab cues only."""
    normalized = _normalize_text(text)
    if not normalized:
        return UNKNOWN_DOCUMENT_LABEL

    urinalysis_specific_hits = _count_term_hits(normalized, _URINALYSIS_TERMS)
    urinalysis_score = urinalysis_specific_hits
    urinalysis_score += _count_word_hits(normalized, _URINALYSIS_SHORT_TERMS)
    ru_urinalysis_specific_hits = _count_term_hits(normalized, _RU_URINALYSIS_TERMS)
    ru_urinalysis_score = ru_urinalysis_specific_hits
    ru_urinalysis_score += _count_word_hits(normalized, _RU_URINALYSIS_SHORT_TERMS)
    lab_score = _count_term_hits(normalized, _LAB_TERMS)
    lab_score += _count_term_hits(normalized, _LAB_PANEL_TERMS)
    lab_score += _count_term_hits(normalized, _LAB_TEST_TERMS)
    lab_score += _count_word_hits(normalized, _LAB_SHORT_TERMS)
    ru_lab_score = _count_term_hits(normalized, _RU_LAB_TERMS)
    ru_lab_score += _count_word_hits(normalized, _RU_LAB_SHORT_TERMS)
    treatment_score = _count_term_hits(normalized, _RU_TREATMENT_TERMS)
    medication_score = _count_term_hits(normalized, _RU_MEDICATION_TERMS)
    treatment_medication_score = treatment_score + medication_score
    lab_total_score = lab_score + ru_lab_score

    if "urinalysis" in normalized or (urinalysis_specific_hits >= 1 and urinalysis_score >= 2 and lab_score >= 1):
        return URINALYSIS_LABEL
    if (
        "общий анализ мочи" in normalized
        or "анализ мочи" in normalized
        or (ru_urinalysis_specific_hits >= 1 and ru_urinalysis_score >= 2 and ru_lab_score >= 1)
    ):
        return URINALYSIS_LABEL
    if medication_score >= 3 and medication_score >= lab_total_score and medication_score >= treatment_score:
        return MEDICATION_PLAN_LABEL
    if treatment_score >= 2 and treatment_medication_score > lab_total_score:
        return TREATMENT_PLAN_LABEL
    strong_lab_layout = (
        ("reference range" in normalized or "reference interval" in normalized)
        and ("result" in normalized or "value" in normalized)
    )
    strong_ru_lab_layout = (
        ("референсные значения" in normalized or "норма" in normalized)
        and ("результат" in normalized or "значение" in normalized or "показатель" in normalized)
    )
    lab_panel_present = _count_term_hits(normalized, _LAB_PANEL_TERMS) > 0
    lab_test_hits = _count_term_hits(normalized, _LAB_TEST_TERMS) + _count_word_hits(normalized, _LAB_SHORT_TERMS)
    if lab_score >= 3 or ru_lab_score >= 3 or strong_lab_layout or strong_ru_lab_layout or (lab_panel_present and lab_test_hits >= 2):
        return LAB_RESULT_LABEL
    return UNKNOWN_DOCUMENT_LABEL


def display_document_type(existing: Any = None, text: str | None = None) -> str:
    if existing:
        mapped = _DOCUMENT_TYPE_MAP.get(str(existing).strip().lower())
        if mapped:
            return mapped
        return str(existing)
    return classify_lab_document_type(text)


def normalize_text_quality_label(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        return _TEXT_QUALITY_MAP.get(text.lower(), text)
    return TEXT_QUALITY_NOT_CHECKED


def review_reason_for_result(
    *,
    document_type: str | None,
    validation_status: str | None,
    confidence: float | None,
    status: str | None = None,
) -> str:
    display_type = display_document_type(document_type)
    normalized_status = str(status or "").strip().lower()
    normalized_validation = str(validation_status or "").strip().lower()
    if normalized_status == "accepted" or normalized_validation == "accepted":
        return "Usable, but still check before relying on it."
    if normalized_status == "error":
        return "Processing failed; check the message and try again."
    if normalized_status == "empty" or normalized_validation == "empty":
        return "Needs review: MedAI could not read useful text."
    if display_type == URINALYSIS_LABEL and normalized_validation in {"needs_review", "rejected"}:
        return "Needs review: urinalysis-style document detected, but confidence is below the acceptance gate."
    if display_type == LAB_RESULT_LABEL and normalized_validation in {"needs_review", "rejected"}:
        return "Needs review: lab-style document detected, but confidence is below the acceptance gate."
    if display_type == TREATMENT_PLAN_LABEL and normalized_validation in {"needs_review", "rejected"}:
        return "Needs review: treatment-plan style document detected. Human review is required."
    if display_type == MEDICATION_PLAN_LABEL and normalized_validation in {"needs_review", "rejected"}:
        return "Needs review: medication-plan style document detected. Human review is required."
    if display_type == UNKNOWN_DOCUMENT_LABEL and normalized_validation in {"needs_review", "rejected"}:
        return "Needs review: MedAI could not confidently identify this document type."
    if confidence is not None and confidence < 0.65:
        return "Reason: Low confidence."
    return "Needs review: operator review required."


def reason_label_for_validation(validation_status: str | None, reason_codes: list[str] | None = None) -> str:
    codes = {str(code).strip().lower() for code in (reason_codes or []) if str(code).strip()}
    value = str(validation_status or "").strip().lower()
    if "confidence_below_reject_threshold" in codes or value == "rejected":
        return "Low confidence"
    if "confidence_below_accept_threshold" in codes or value == "needs_review":
        return "Review gate"
    if "empty_extraction" in codes or value == "empty":
        return "No text found"
    return ""


def _normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower())


def _count_term_hits(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if term in text)


def _count_word_hits(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", text, re.I))
