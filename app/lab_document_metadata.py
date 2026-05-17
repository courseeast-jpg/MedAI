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
_RU_LAB_TERMS_CYRILLIC = (
    "\u0430\u043d\u0430\u043b\u0438\u0437",
    "\u0438\u0441\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u043d\u0438\u0435",
    "\u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d\u043e\u0435 \u0438\u0441\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u043d\u0438\u0435",
    "\u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d\u044b\u0439 \u043e\u0442\u0447\u0435\u0442",
    "\u0431\u043b\u0430\u043d\u043a \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432",
    "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442",
    "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b",
    "\u0440\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u043d\u044b\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f",
    "\u0440\u0435\u0444 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f",
    "\u0440\u0435\u0444 \u0438\u043d\u0442\u0435\u0440\u0432\u0430\u043b",
    "\u043d\u043e\u0440\u043c\u0430",
    "\u0435\u0434\u0438\u043d\u0438\u0446\u044b",
    "\u0435\u0434 \u0438\u0437\u043c",
    "\u0435\u0434\u0438\u043d\u0438\u0446\u044b \u0438\u0437\u043c\u0435\u0440\u0435\u043d\u0438\u044f",
    "\u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c",
    "\u043d\u0430\u0438\u043c\u0435\u043d\u043e\u0432\u0430\u043d\u0438\u0435",
    "\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435",
    "\u0431\u0438\u043e\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b",
    "\u0441\u044b\u0432\u043e\u0440\u043e\u0442\u043a\u0430",
    "\u043f\u043b\u0430\u0437\u043c\u0430",
    "\u043a\u0440\u043e\u0432\u044c",
    "\u043e\u0431\u0449\u0438\u0439 \u0430\u043d\u0430\u043b\u0438\u0437 \u043a\u0440\u043e\u0432\u0438",
    "\u0431\u0438\u043e\u0445\u0438\u043c\u0438\u044f",
    "\u0433\u0435\u043c\u043e\u0433\u043b\u043e\u0431\u0438\u043d",
    "\u043b\u0435\u0439\u043a\u043e\u0446\u0438\u0442\u044b",
    "\u044d\u0440\u0438\u0442\u0440\u043e\u0446\u0438\u0442\u044b",
    "\u0442\u0440\u043e\u043c\u0431\u043e\u0446\u0438\u0442\u044b",
    "\u043d\u0435\u0439\u0442\u0440\u043e\u0444\u0438\u043b\u044b",
    "\u043b\u0438\u043c\u0444\u043e\u0446\u0438\u0442\u044b",
    "\u043c\u043e\u043d\u043e\u0446\u0438\u0442\u044b",
    "\u044d\u043e\u0437\u0438\u043d\u043e\u0444\u0438\u043b\u044b",
    "\u0433\u043b\u044e\u043a\u043e\u0437\u0430",
    "\u043a\u0440\u0435\u0430\u0442\u0438\u043d\u0438\u043d",
    "\u043c\u043e\u0447\u0435\u0432\u0438\u043d\u0430",
    "\u0431\u0438\u043b\u0438\u0440\u0443\u0431\u0438\u043d",
    "\u0445\u043e\u043b\u0435\u0441\u0442\u0435\u0440\u0438\u043d",
)
_RU_LAB_SHORT_TERMS_CYRILLIC = ("\u043e\u0430\u043a", "\u0430\u043b\u0442", "\u0430\u0441\u0442")
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
    ru_lab_score += _count_term_hits(normalized, _RU_LAB_TERMS_CYRILLIC)
    ru_lab_score += _count_word_hits(normalized, _RU_LAB_SHORT_TERMS_CYRILLIC)
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
    strong_ru_lab_layout = strong_ru_lab_layout or (
        (
            "\u0440\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u043d\u044b\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f" in normalized
            or "\u0440\u0435\u0444 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f" in normalized
            or "\u043d\u043e\u0440\u043c\u0430" in normalized
        )
        and (
            "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442" in normalized
            or "\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435" in normalized
            or "\u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c" in normalized
        )
    )
    lab_test_hits = _count_term_hits(normalized, _LAB_TEST_TERMS) + _count_word_hits(normalized, _LAB_SHORT_TERMS)
    if lab_score >= 3 or ru_lab_score >= 3 or strong_lab_layout or strong_ru_lab_layout or (lab_panel_present and lab_test_hits >= 2):
        return LAB_RESULT_LABEL
    return UNKNOWN_DOCUMENT_LABEL


def display_document_type(existing: Any = None, text: str | None = None) -> str:
    classified_from_text = classify_lab_document_type(text)
    if existing:
        existing_key = str(existing).strip().lower()
        mapped = _DOCUMENT_TYPE_MAP.get(existing_key)
        if mapped:
            if mapped == UNKNOWN_DOCUMENT_LABEL and classified_from_text != UNKNOWN_DOCUMENT_LABEL:
                return classified_from_text
            return mapped
        return str(existing)
    return classified_from_text


def safe_document_type_diagnostic(
    *,
    document_type_before: Any = None,
    text: str | None = None,
    extractor: Any = None,
    confidence: Any = None,
    ocr_quality: Any = None,
) -> dict[str, Any]:
    """Return privacy-safe text-path diagnostics without emitting source text."""
    normalized = _normalize_text(text)
    cyrillic_count = len(re.findall(r"[а-яё]", normalized, flags=re.I))
    total_chars = len(normalized.replace(" ", ""))
    cyrillic_ratio = (cyrillic_count / total_chars) if total_chars else 0.0
    categories = _russian_cue_categories(normalized)
    text_available = bool(normalized)
    document_type_after = display_document_type(document_type_before, text=text)
    likely_reason = "unknown"
    if not text_available:
        likely_reason = "no_text_available"
    elif cyrillic_count == 0:
        likely_reason = "no_cyrillic_text_detected"
    elif document_type_after == UNKNOWN_DOCUMENT_LABEL and len(categories) < 3:
        likely_reason = "too_few_cue_categories"
    elif document_type_after == UNKNOWN_DOCUMENT_LABEL:
        likely_reason = "cue_normalization_gap"
    elif (
        str(document_type_before or "").strip().lower() in {"unknown", "generic"}
        and document_type_after != UNKNOWN_DOCUMENT_LABEL
    ):
        likely_reason = "classifier_input_field_missing"

    return {
        "document_type_before": display_document_type(document_type_before),
        "document_type_after": document_type_after,
        "extractor": str(extractor) if extractor is not None else None,
        "confidence_bucket": _bucket_confidence(confidence),
        "ocr_text_quality": str(ocr_quality) if ocr_quality is not None else None,
        "text_available": text_available,
        "text_length_bucket": _bucket_text_length(len(normalized)),
        "cyrillic_detected": cyrillic_count > 0,
        "cyrillic_density_bucket": _bucket_cyrillic_density(cyrillic_ratio),
        "russian_lab_cue_categories_detected": categories,
        "cue_category_count": len(categories),
        "likely_reason_unknown": likely_reason,
    }


def safe_fallback_ocr_classification_diagnostic(text: str | None) -> dict[str, Any]:
    """Return fallback OCR classification diagnostics without exposing OCR text."""
    normalized = _normalize_text(text)
    cyrillic_count = len(re.findall(r"[\u0400-\u04ff]", normalized))
    matched_keys = _matched_russian_lab_cue_keys(normalized)
    candidate = classify_lab_document_type(text)
    if not normalized:
        block_reason = "no_fallback_text_available"
    elif cyrillic_count == 0:
        block_reason = "no_cyrillic_detected"
    elif candidate == UNKNOWN_DOCUMENT_LABEL:
        block_reason = "too_few_safe_lab_cue_keys" if len(matched_keys) < 3 else "classification_threshold_not_met"
    else:
        block_reason = "classified"
    return {
        "cyrillic_detected": cyrillic_count > 0,
        "cyrillic_char_count_bucket": _bucket_count(cyrillic_count),
        "matched_lab_cue_keys": matched_keys,
        "matched_document_type_candidate": candidate,
        "classification_block_reason": block_reason,
    }


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
    value = str(text or "").lower().replace("ё", "е").replace("С‘", "Рµ")
    value = re.sub(r"[\u00a0\t\r\n|;:,./\\()\[\]{}<>_+=-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _count_term_hits(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if term in text)


def _count_word_hits(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", text, re.I))


def _russian_cue_categories(text: str) -> list[str]:
    categories: list[str] = []
    category_terms = {
        "lab_header": ("лабораторное исследование", "анализ", "общий анализ крови", "биохимия"),
        "result_terms": ("результат", "результаты", "показатель", "значение"),
        "reference_terms": ("референсные значения", "норма"),
        "analyte_terms": (
            "гемоглобин",
            "лейкоциты",
            "эритроциты",
            "тромбоциты",
            "глюкоза",
            "креатинин",
            "мочевина",
            "билирубин",
            "холестерин",
        ),
        "units_terms": ("единицы", "мг", "мл", "ммоль", "г/л"),
        "blood_panel_terms": ("оак", "общий анализ крови", "кровь", "биохимия"),
        "urinalysis_terms": _RU_URINALYSIS_TERMS + _RU_URINALYSIS_SHORT_TERMS,
        "treatment_plan_terms": _RU_TREATMENT_TERMS,
        "medication_plan_terms": _RU_MEDICATION_TERMS,
    }
    for category, terms in category_terms.items():
        if _count_term_hits(text, terms) or _count_word_hits(text, terms):
            categories.append(category)
    return categories


def _matched_russian_lab_cue_keys(text: str) -> list[str]:
    cue_terms = {
        "generic_lab_form": (
            "\u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d\u043e\u0435 \u0438\u0441\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u043d\u0438\u0435",
            "\u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d\u044b\u0439 \u043e\u0442\u0447\u0435\u0442",
            "\u0431\u043b\u0430\u043d\u043a \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u043e\u0432",
            "\u0430\u043d\u0430\u043b\u0438\u0437",
            "\u0438\u0441\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u043d\u0438\u0435",
        ),
        "specimen_or_biomaterial": (
            "\u0431\u0438\u043e\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b",
            "\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b",
            "\u0441\u044b\u0432\u043e\u0440\u043e\u0442\u043a\u0430",
            "\u043f\u043b\u0430\u0437\u043c\u0430",
            "\u043a\u0440\u043e\u0432\u044c",
        ),
        "result_or_report": (
            "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442",
            "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b",
            "\u043e\u0442\u0447\u0435\u0442",
            "\u0437\u0430\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u0435",
        ),
        "table_header": (
            "\u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c",
            "\u043d\u0430\u0438\u043c\u0435\u043d\u043e\u0432\u0430\u043d\u0438\u0435",
            "\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435",
            "\u0435\u0434\u0438\u043d\u0438\u0446\u044b",
            "\u0435\u0434 \u0438\u0437\u043c",
            "\u043d\u043e\u0440\u043c\u0430",
            "\u0440\u0435\u0444 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f",
            "\u0440\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u043d\u044b\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f",
        ),
        "common_analyte": (
            "\u0433\u0435\u043c\u043e\u0433\u043b\u043e\u0431\u0438\u043d",
            "\u043b\u0435\u0439\u043a\u043e\u0446\u0438\u0442\u044b",
            "\u044d\u0440\u0438\u0442\u0440\u043e\u0446\u0438\u0442\u044b",
            "\u0442\u0440\u043e\u043c\u0431\u043e\u0446\u0438\u0442\u044b",
            "\u043d\u0435\u0439\u0442\u0440\u043e\u0444\u0438\u043b\u044b",
            "\u043b\u0438\u043c\u0444\u043e\u0446\u0438\u0442\u044b",
            "\u0433\u043b\u044e\u043a\u043e\u0437\u0430",
            "\u043a\u0440\u0435\u0430\u0442\u0438\u043d\u0438\u043d",
        ),
        "lab_panel_abbreviation": ("\u043e\u0430\u043a", "\u0430\u043b\u0442", "\u0430\u0441\u0442"),
    }
    return [key for key, terms in cue_terms.items() if _count_term_hits(text, terms) or _count_word_hits(text, terms)]


def _bucket_text_length(length: int) -> str:
    if length <= 0:
        return "none"
    if length < 80:
        return "tiny"
    if length < 500:
        return "short"
    if length < 2000:
        return "medium"
    return "long"


def _bucket_cyrillic_density(ratio: float) -> str:
    if ratio <= 0:
        return "none"
    if ratio < 0.2:
        return "low"
    if ratio < 0.6:
        return "medium"
    return "high"


def _bucket_confidence(value: Any) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if confidence < 0.5:
        return "low_under_0_50"
    if confidence < 0.65:
        return "moderate_0_50_to_0_64"
    if confidence < 0.85:
        return "review_band_0_65_to_0_84"
    return "high_0_85_or_above"


def _bucket_count(value: int) -> str:
    if value <= 0:
        return "none"
    if value < 20:
        return "low"
    if value < 120:
        return "medium"
    return "high"
