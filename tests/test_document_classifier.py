from __future__ import annotations

from document_classification import classify_document


# ---------------------------------------------------------------------------
# English lab reports
# ---------------------------------------------------------------------------


def test_standard_english_lab_text_classified_as_lab_report() -> None:
    text = "\n".join([
        "Patient lab results",
        "WBC 6.2 x10E3/uL 3.4-10.8",
        "Glucose 103 mg/dL 65-99 H",
        "Hemoglobin 14.5 g/dL 12-16",
    ])

    result = classify_document(text)

    assert result.document_type == "lab_report"
    assert result.should_run_lab_normalization is True
    assert result.language_hint == "en"
    assert result.confidence >= 0.55


def test_urinalysis_classified_as_lab_report() -> None:
    text = "\n".join([
        "Urinalysis Test Results",
        "Specific Gravity 1.025 1.005-1.030",
        "Ketones Negative",
        "Blood Negative",
    ])

    result = classify_document(text)

    assert result.document_type == "lab_report"
    assert result.should_run_lab_normalization is True


# ---------------------------------------------------------------------------
# Prescription detection
# ---------------------------------------------------------------------------


def test_russian_prescription_classified_correctly() -> None:
    text = "\n".join([
        "Препараты:",
        "Свечи Диклофенак 100 мг",
        "Метронидазол по 500 мг 2 раза в день",
        "Левофлоксацин 500 мг",
    ])

    result = classify_document(text)

    assert result.document_type == "prescription"
    assert result.should_run_lab_normalization is False
    assert result.should_run_prescription_path is True
    assert result.review_reason == "document_type_prescription_not_lab"
    assert result.language_hint in {"ru", "mixed"}
    assert result.should_recommend_language_aware_ocr is True


def test_ocr_mangled_russian_prescription_detected() -> None:
    # Excerpt of mangled OCR text from Test Results 6.pdf
    text = "\n".join([
        "l1PEIIAPAT1I I L(ATA",
        "CseLJ:H ,IUIKJio¢eHaK 100Mr",
        "MeTpOHH,n:a3on no 500Mr 2",
        "<l>nyKOHa3011 no 150 Mr 1 p",
        "J1HHeKC cpOpT3 no lKancyne",
        "TaMcyJio3HH no 0.4Mr 1p",
    ])

    result = classify_document(text)

    assert result.document_type == "prescription"
    assert result.should_run_lab_normalization is False
    assert result.should_recommend_language_aware_ocr is True


# ---------------------------------------------------------------------------
# Microbiology / PCR detection
# ---------------------------------------------------------------------------


def test_english_pcr_text_classified_as_microbiology() -> None:
    text = "\n".join([
        "Androflor PCR Report",
        "Microbiology screen",
        "Candida spp. detected",
        "Ureaplasma not detected",
        "PCR analysis result",
    ])

    result = classify_document(text)

    assert result.document_type == "microbiology_pcr_report"
    # English microbiology — parser still runs (paired_microbiology_rows handles it)
    assert result.should_run_lab_normalization is True


def test_ocr_mangled_androflor_text_classified_as_microbiology() -> None:
    # Excerpt of mangled OCR text from Test Results 3.pdf
    text = "\n".join([
        "LllccneAoBaHHe MHKpocpnopbl yporeHHTailbHoro TpaKTa MY>KliHH",
        "MeTOAOM nl..tp B pe>KHMe peailbHOrO BpeMeHH",
        "AHApocpnop®,AHApOcpnop®CKpHH",
    ])

    result = classify_document(text)

    assert result.document_type == "microbiology_pcr_report"
    # Pseudo-Russian — parser is skipped to avoid forcing English row parsing
    assert result.should_run_lab_normalization is False
    assert result.should_recommend_language_aware_ocr is True
    assert result.review_reason == "microbiology_pcr_report_detected"


# ---------------------------------------------------------------------------
# Unknown / safety
# ---------------------------------------------------------------------------


def test_noisy_unknown_text_classified_as_unknown_safely() -> None:
    text = "asdf qwer zxcv 12345 noise no recognized terms here"

    result = classify_document(text)

    assert result.document_type in {"unknown_medical", "unknown_nonmedical"}
    # Default-safe: still allow lab normalization to attempt (which will fail
    # cleanly), preserving Phase 41 behavior
    assert result.should_run_lab_normalization is True
    assert result.confidence < 0.50


def test_classifier_does_not_auto_accept() -> None:
    # No matter what we classify, classifier itself never produces an "accepted"
    # status — that's the pipeline's job behind safety gates.
    for text in [
        "Glucose 103 mg/dL",
        "Свечи Диклофенак 100 мг",
        "Androflor PCR",
        "asdf qwer",
    ]:
        result = classify_document(text)
        # No accepted-status fields exist on the model
        assert not hasattr(result, "status")
        assert not hasattr(result, "accepted")


def test_low_confidence_warning_emitted() -> None:
    result = classify_document("noise blob random")

    assert "low_confidence_document_type" in result.warnings


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


def test_empty_text_classified_safely() -> None:
    result = classify_document("")

    assert result.document_type == "unknown_medical"
    assert result.should_run_lab_normalization is True  # preserve existing behavior
    assert result.should_recommend_language_aware_ocr is False
    assert result.confidence < 0.50


def test_metadata_includes_token_counts() -> None:
    result = classify_document("Glucose 103 mg/dL Reference range 65-99")

    assert result.metadata["lab_token_count_en"] >= 2
    assert "cyrillic_ratio" in result.metadata
    assert "latin_ratio" in result.metadata
