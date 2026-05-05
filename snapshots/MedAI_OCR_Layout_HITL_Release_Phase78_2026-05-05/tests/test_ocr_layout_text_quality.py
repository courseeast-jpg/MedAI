from __future__ import annotations

from ocr_layout.text_quality import assess_text_quality


def test_good_medical_text_scores_as_good_or_usable() -> None:
    text = (
        "Patient urinalysis result negative. Glucose negative. Protein negative. "
        "RBC normal. WBC normal. Nitrite negative. Culture no growth. "
    ) * 8

    quality = assess_text_quality(text)

    assert quality.score >= 0.72
    assert quality.band == "good"
    assert quality.metrics["script"] == "en"


def test_empty_text_is_empty_band() -> None:
    quality = assess_text_quality(" \n\t ")

    assert quality.score == 0.0
    assert quality.band == "empty"
    assert "empty_or_near_empty_text" in quality.warnings


def test_noisy_ocr_scores_as_poor() -> None:
    text = "|||| ____ \ufffd \ufffd 11111111 ~~~~~ BIO MED L4B ??? " * 3

    quality = assess_text_quality(text)

    assert quality.band == "poor_ocr"
    assert "low_alphabetic_ratio" in quality.warnings
    assert "repeated_symbol_noise" in quality.warnings


def test_mixed_script_detection() -> None:
    text = "Patient анализ urine мочи glucose глюкоза normal норма " * 4

    quality = assess_text_quality(text)

    assert quality.metrics["script"] == "mixed"
