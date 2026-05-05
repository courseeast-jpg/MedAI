from __future__ import annotations

from document_classification import (
    classify_document,
    reconcile_cyrillic_nonlab_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_RU_PRESCRIPTION_TEXT = "\n".join([
    "Препараты:",
    "Свечи Диклофенак 100 мг по 1 свече на ночь",
    "Метронидазол 500 мг 2 раза в день после еды",
    "Левофлоксацин 500 мг 1 раз в день",
    "Рекомендации врача: принимать строго по назначению.",
])

_RU_LAB_TEXT = "\n".join([
    "Анализы крови",
    "Глюкоза 5.6 ммоль/л 3.3-6.1",
    "Гемоглобин 145 г/л 130-160",
    "WBC 6.2 x10E3/uL 3.4-10.8",
    "Holesterol 4.5 mmol/L",
])


def _classify_payload(text: str) -> dict:
    return classify_document(text).to_dict()


def _lab_payload(text: str, *, coverage_band: str = "none") -> dict:
    return {
        "document_classification": _classify_payload(text),
        "lab_coverage_band": coverage_band,
        "skipped_for_document_type": True,
    }


# ---------------------------------------------------------------------------
# Trigger: review_ocr_quality + good band + Cyrillic prescription -> review
# ---------------------------------------------------------------------------


def test_cyrillic_prescription_review_ocr_quality_reconciled_to_review() -> None:
    result = reconcile_cyrillic_nonlab_status(
        current_status="review_ocr_quality",
        is_ocr_low_quality=True,
        classification_reason_codes=[
            "table_structure_loss",
            "extraction_low_coverage",
            "classifier_legacy_ocr_flag",
            "legacy_normalized_low_coverage",
            "non_lab_document_skipped_lab_normalization",
            "document_type_prescription_not_lab",
            "language_aware_ocr_required",
        ],
        selected_text=_RU_PRESCRIPTION_TEXT,
        ocr_layout={"input_quality_band": "good"},
        lab_normalization=_lab_payload(_RU_PRESCRIPTION_TEXT, coverage_band="none"),
        entity_count=3,
    )

    assert result.triggered is True
    assert result.new_status == "review"
    assert result.cyrillic_non_lab_document_detected is True
    assert result.ocr_quality_recovered_non_lab is True
    assert "cyrillic_non_lab_document_review" in result.new_reason_codes
    assert "ocr_quality_recovered_non_lab" in result.new_reason_codes
    assert "prescription_or_medication_instruction_detected" in result.new_reason_codes
    # Original codes preserved
    assert "table_structure_loss" in result.new_reason_codes
    assert "non_lab_document_skipped_lab_normalization" in result.new_reason_codes


def test_reconciliation_never_produces_accepted() -> None:
    result = reconcile_cyrillic_nonlab_status(
        current_status="review_ocr_quality",
        is_ocr_low_quality=True,
        classification_reason_codes=[],
        selected_text=_RU_PRESCRIPTION_TEXT,
        ocr_layout={"input_quality_band": "good"},
        lab_normalization=_lab_payload(_RU_PRESCRIPTION_TEXT),
        entity_count=5,
    )

    assert result.new_status != "accepted"
    assert result.new_status in {"review", "review_ocr_quality"}


def test_reconciliation_for_usable_with_review_band() -> None:
    result = reconcile_cyrillic_nonlab_status(
        current_status="review_ocr_quality",
        is_ocr_low_quality=True,
        classification_reason_codes=[],
        selected_text=_RU_PRESCRIPTION_TEXT,
        ocr_layout={"input_quality_band": "usable_with_review"},
        lab_normalization=_lab_payload(_RU_PRESCRIPTION_TEXT),
        entity_count=2,
    )

    assert result.triggered is True
    assert result.new_status == "review"


# ---------------------------------------------------------------------------
# Negative gates
# ---------------------------------------------------------------------------


def test_no_reconciliation_when_not_review_ocr_quality() -> None:
    for current in ("review", "accepted", "error"):
        result = reconcile_cyrillic_nonlab_status(
            current_status=current,
            is_ocr_low_quality=False,
            classification_reason_codes=[],
            selected_text=_RU_PRESCRIPTION_TEXT,
            ocr_layout={"input_quality_band": "good"},
            lab_normalization=_lab_payload(_RU_PRESCRIPTION_TEXT),
            entity_count=3,
        )

        assert result.triggered is False
        assert result.new_status == current


def test_poor_ocr_remains_review_ocr_quality() -> None:
    for band in ("poor_ocr", "empty"):
        result = reconcile_cyrillic_nonlab_status(
            current_status="review_ocr_quality",
            is_ocr_low_quality=True,
            classification_reason_codes=[],
            selected_text=_RU_PRESCRIPTION_TEXT,
            ocr_layout={"input_quality_band": band},
            lab_normalization=_lab_payload(_RU_PRESCRIPTION_TEXT),
            entity_count=3,
        )

        assert result.triggered is False
        assert result.new_status == "review_ocr_quality"


def test_empty_extraction_does_not_block_reconciliation() -> None:
    # entity_count == 0 must not block reconciliation: empty-extraction
    # files still need to leave the review_ocr_quality bucket for non-lab
    # Cyrillic content. The spec's "no empty extraction leakage" requirement
    # only forbids accepted status, which reconciliation never produces.
    result = reconcile_cyrillic_nonlab_status(
        current_status="review_ocr_quality",
        is_ocr_low_quality=True,
        classification_reason_codes=[],
        selected_text=_RU_PRESCRIPTION_TEXT,
        ocr_layout={"input_quality_band": "good"},
        lab_normalization=_lab_payload(_RU_PRESCRIPTION_TEXT),
        entity_count=0,
    )

    assert result.triggered is True
    assert result.new_status == "review"
    assert result.new_status != "accepted"


def test_lab_report_documents_are_not_reconciled() -> None:
    # Lab reports must not be misclassified as prescriptions or get downgraded
    result = reconcile_cyrillic_nonlab_status(
        current_status="review_ocr_quality",
        is_ocr_low_quality=True,
        classification_reason_codes=[],
        selected_text=_RU_LAB_TEXT,
        ocr_layout={"input_quality_band": "good"},
        lab_normalization={
            "document_classification": _classify_payload(_RU_LAB_TEXT),
            "lab_coverage_band": "partial",
        },
        entity_count=3,
    )

    assert result.triggered is False


def test_microbiology_pcr_report_not_reconciled_by_phase45() -> None:
    # Microbiology PCR is handled by Phase 43 routing; Phase 45 leaves it
    # alone (its own code path applies).
    micro_text = "ПЦР микрофлора урогенитального тракта.\nАндрофлор результаты."
    result = reconcile_cyrillic_nonlab_status(
        current_status="review_ocr_quality",
        is_ocr_low_quality=True,
        classification_reason_codes=[],
        selected_text=micro_text,
        ocr_layout={"input_quality_band": "good"},
        lab_normalization=_lab_payload(micro_text),
        entity_count=3,
    )

    assert result.triggered is False


def test_low_cyrillic_ratio_blocks_reconciliation() -> None:
    # Mostly-English prescription-ish text would not pass the Cyrillic gate.
    text = "Take metformin 500mg twice daily after meals"
    result = reconcile_cyrillic_nonlab_status(
        current_status="review_ocr_quality",
        is_ocr_low_quality=True,
        classification_reason_codes=[],
        selected_text=text,
        ocr_layout={"input_quality_band": "good"},
        lab_normalization={
            "document_classification": _classify_payload(text),
            "lab_coverage_band": "none",
        },
        entity_count=2,
    )

    assert result.triggered is False


def test_strong_lab_coverage_blocks_reconciliation() -> None:
    # If lab parser actually parsed real rows, reconciliation must NOT fire
    # (because the document IS providing lab data).
    result = reconcile_cyrillic_nonlab_status(
        current_status="review_ocr_quality",
        is_ocr_low_quality=True,
        classification_reason_codes=[],
        selected_text=_RU_PRESCRIPTION_TEXT,
        ocr_layout={"input_quality_band": "good"},
        lab_normalization={
            "document_classification": _classify_payload(_RU_PRESCRIPTION_TEXT),
            "lab_coverage_band": "good",  # contradicts "non-lab" premise
        },
        entity_count=3,
    )

    assert result.triggered is False


# ---------------------------------------------------------------------------
# Unknown_medical with prescription signals
# ---------------------------------------------------------------------------


def test_unknown_medical_with_strong_rx_signals_reconciled() -> None:
    text = (
        "Свечи Диклофенак 100 мг "
        "по 1 свече на ночь "
        "после еды"
    )
    classification_payload = classify_document(text).to_dict()
    # Force-classify as unknown_medical to test that path
    classification_payload["document_type"] = "unknown_medical"
    classification_payload["metadata"]["rx_signal_total"] = 4

    result = reconcile_cyrillic_nonlab_status(
        current_status="review_ocr_quality",
        is_ocr_low_quality=True,
        classification_reason_codes=[],
        selected_text=text,
        ocr_layout={"input_quality_band": "good"},
        lab_normalization={
            "document_classification": classification_payload,
            "lab_coverage_band": "none",
        },
        entity_count=2,
    )

    assert result.triggered is True
    # prescription-specific code only when document_type == "prescription"
    assert "prescription_or_medication_instruction_detected" not in result.new_reason_codes


def test_unknown_medical_without_rx_signals_not_reconciled() -> None:
    text = "Просто текст на русском языке без явных медицинских деталей."
    classification_payload = classify_document(text).to_dict()
    classification_payload["document_type"] = "unknown_medical"
    classification_payload["metadata"]["rx_signal_total"] = 0

    result = reconcile_cyrillic_nonlab_status(
        current_status="review_ocr_quality",
        is_ocr_low_quality=True,
        classification_reason_codes=[],
        selected_text=text,
        ocr_layout={"input_quality_band": "good"},
        lab_normalization={
            "document_classification": classification_payload,
            "lab_coverage_band": "none",
        },
        entity_count=1,
    )

    assert result.triggered is False


# ---------------------------------------------------------------------------
# Token set extension regression
# ---------------------------------------------------------------------------


def test_extended_russian_prescription_tokens_match_phase45_terms() -> None:
    text = (
        "Назначения врача:\n"
        "Лекарство принимать после еды.\n"
        "Дозировка: 100 мг 2 раза в день.\n"
        "Рекомендации: на ночь, до еды.\n"
        "Аптека выдаст препарат по рецепту."
    )

    result = classify_document(text)

    assert result.document_type == "prescription"
    assert result.confidence >= 0.55
    assert result.metadata["prescription_token_count_ru"] >= 4
