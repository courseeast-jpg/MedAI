from __future__ import annotations

from pathlib import Path
from unittest import mock

from document_classification import classify_document
from ocr_layout import cyrillic_ocr, ocr_candidates, ocr_capabilities
from ocr_layout.ocr_capabilities import OcrCapabilityReport


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


def test_capability_report_when_tesseract_missing(monkeypatch) -> None:
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    with mock.patch("ocr_layout.ocr_capabilities.shutil.which", return_value=None), \
         mock.patch("ocr_layout.ocr_capabilities.os.path.isfile", return_value=False):
        report = ocr_capabilities.get_ocr_capability_report()

    assert report.tesseract_available is False
    assert report.tesseract_path is None
    assert report.russian_available is False
    assert "tesseract_binary_not_found" in report.warnings
    assert "cyrillic_ocr_unavailable" in report.warnings


def test_capability_report_when_rus_traineddata_missing(monkeypatch) -> None:
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    with mock.patch(
        "ocr_layout.ocr_capabilities.detect_tesseract_available",
        return_value="/usr/bin/tesseract",
    ), mock.patch(
        "ocr_layout.ocr_capabilities.detect_tesseract_languages",
        return_value=(["eng", "fra"], []),
    ):
        report = ocr_capabilities.get_ocr_capability_report()

    assert report.tesseract_available is True
    assert report.russian_available is False
    assert "cyrillic_ocr_unavailable" in report.warnings
    assert "eng" in report.available_languages


def test_capability_report_when_rus_present(monkeypatch) -> None:
    with mock.patch(
        "ocr_layout.ocr_capabilities.detect_tesseract_available",
        return_value="/usr/bin/tesseract",
    ), mock.patch(
        "ocr_layout.ocr_capabilities.detect_tesseract_languages",
        return_value=(["eng", "rus", "ukr"], []),
    ):
        report = ocr_capabilities.get_ocr_capability_report()

    assert report.tesseract_available is True
    assert report.russian_available is True
    assert "cyrillic_ocr_unavailable" not in report.warnings


def test_has_russian_ocr_support_helper() -> None:
    assert ocr_capabilities.has_russian_ocr_support(["eng", "rus", "fra"]) is True
    assert ocr_capabilities.has_russian_ocr_support(["eng", "fra"]) is False
    assert ocr_capabilities.has_russian_ocr_support([]) is False


# ---------------------------------------------------------------------------
# Cyrillic OCR candidate
# ---------------------------------------------------------------------------


def _unavailable_capability() -> OcrCapabilityReport:
    return OcrCapabilityReport(
        tesseract_available=False,
        tesseract_path=None,
        available_languages=[],
        russian_available=False,
        english_available=False,
        warnings=["tesseract_binary_not_found", "cyrillic_ocr_unavailable"],
        metadata={},
    )


def _no_rus_capability() -> OcrCapabilityReport:
    return OcrCapabilityReport(
        tesseract_available=True,
        tesseract_path="/usr/bin/tesseract",
        available_languages=["eng", "fra"],
        russian_available=False,
        english_available=True,
        warnings=["cyrillic_ocr_unavailable"],
        metadata={},
    )


def test_cyrillic_ocr_unavailable_returns_warning_not_failure(tmp_path: Path) -> None:
    # Even though source doesn't exist as a real PDF, the unavailability check
    # must short-circuit cleanly without raising.
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"not a real pdf")

    result = cyrillic_ocr.try_cyrillic_ocr_candidate(
        fake_pdf, capability=_unavailable_capability()
    )

    assert result["text"] == ""
    assert "cyrillic_ocr_unavailable" in result["warnings"]
    assert "tesseract_binary_not_found" in result["warnings"]
    assert result["metadata"]["ocr_attempted"] is False


def test_cyrillic_ocr_no_rus_traineddata_returns_warning(tmp_path: Path) -> None:
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"not a real pdf")

    result = cyrillic_ocr.try_cyrillic_ocr_candidate(
        fake_pdf, capability=_no_rus_capability()
    )

    assert result["text"] == ""
    assert "cyrillic_ocr_unavailable" in result["warnings"]
    assert "rus_traineddata_missing" in result["warnings"]
    assert result["metadata"]["ocr_attempted"] is False


def test_cyrillic_ocr_missing_source_returns_warning(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.pdf"
    cap = OcrCapabilityReport(
        tesseract_available=True,
        tesseract_path="/usr/bin/tesseract",
        available_languages=["eng", "rus"],
        russian_available=True,
        english_available=True,
        warnings=[],
        metadata={},
    )

    result = cyrillic_ocr.try_cyrillic_ocr_candidate(missing, capability=cap)

    assert result["text"] == ""
    assert "cyrillic_ocr_source_not_found" in result["warnings"]


def test_cyrillic_ocr_unsupported_extension_returns_warning(tmp_path: Path) -> None:
    txt = tmp_path / "notes.txt"
    txt.write_text("hello", encoding="utf-8")
    cap = OcrCapabilityReport(
        tesseract_available=True,
        tesseract_path="/usr/bin/tesseract",
        available_languages=["eng", "rus"],
        russian_available=True,
        english_available=True,
        warnings=[],
        metadata={},
    )

    result = cyrillic_ocr.try_cyrillic_ocr_candidate(txt, capability=cap)

    assert result["text"] == ""
    assert "cyrillic_ocr_unsupported_extension" in result["warnings"]


# ---------------------------------------------------------------------------
# Trigger heuristic
# ---------------------------------------------------------------------------


def _candidate_with(text: str, *, cyrillic_ratio: float = 0.0) -> ocr_candidates.OcrCandidate:
    return ocr_candidates.OcrCandidate(
        engine_name="existing_pdf_pipeline",
        text=text,
        quality_score=0.75,
        quality_band="good",
        language="en",
        warnings=[],
        metadata={"quality_metrics": {"cyrillic_ratio": cyrillic_ratio}},
    )


def test_should_attempt_cyrillic_when_high_cyrillic_ratio() -> None:
    candidate = _candidate_with("Глюкоза 5.6 ммоль/л", cyrillic_ratio=0.85)

    assert ocr_candidates._should_attempt_cyrillic_ocr(candidate) is True


def test_should_attempt_cyrillic_for_pseudo_russian_prescription() -> None:
    candidate = _candidate_with(
        "CseLJ:H ,IUIKJio¢eHaK 100Mr\nMeTpOHH,n:a3on no 500Mr 2",
        cyrillic_ratio=0.0,
    )

    assert ocr_candidates._should_attempt_cyrillic_ocr(candidate) is True


def test_should_not_attempt_cyrillic_for_clean_english_lab() -> None:
    candidate = _candidate_with(
        "Glucose 103 mg/dL 65-99 H\nWBC 6.2 x10E3/uL 3.4-10.8",
        cyrillic_ratio=0.0,
    )

    assert ocr_candidates._should_attempt_cyrillic_ocr(candidate) is False


def test_should_not_attempt_cyrillic_for_empty_text() -> None:
    candidate = _candidate_with("", cyrillic_ratio=0.0)

    assert ocr_candidates._should_attempt_cyrillic_ocr(candidate) is False


# ---------------------------------------------------------------------------
# Integration: collect_candidates emits placeholder when Cyrillic OCR
# unavailable (mocked) for a Russian source
# ---------------------------------------------------------------------------


def test_collect_candidates_emits_unavailable_placeholder_when_capability_missing(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"not a real pdf")
    pseudo_ru_text = (
        "CseLJ:H ,IUIKJio¢eHaK 100Mr\n"
        "MeTpOHH,n:a3on no 500Mr 2\n"
        "<l>nyKOHa3011 no 150 Mr 1 p"
    )
    cyrillic_candidate = ocr_candidates.OcrCandidate(
        engine_name="existing_pdf_pipeline",
        text=pseudo_ru_text,
        quality_score=0.80,
        quality_band="good",
        language="en",
        warnings=[],
        metadata={"quality_metrics": {"cyrillic_ratio": 0.05}},
    )
    pymupdf_candidate = ocr_candidates.OcrCandidate(
        engine_name="pymupdf_native_text",
        text=pseudo_ru_text,
        quality_score=0.70,
        quality_band="good",
        language="en",
        warnings=[],
        metadata={"quality_metrics": {"cyrillic_ratio": 0.05}},
    )

    with mock.patch(
        "ocr_layout.ocr_candidates.get_ocr_capability_report",
        return_value=_unavailable_capability(),
    ):
        result = ocr_candidates._maybe_cyrillic_ocr_candidate(
            pdf_path,
            [cyrillic_candidate, pymupdf_candidate],
        )

    assert result is not None
    assert result.engine_name == "tesseract_rus_unavailable"
    assert result.text == ""
    assert result.metadata.get("cyrillic_ocr_unavailable") is True


def test_collect_candidates_skips_cyrillic_attempt_for_clean_english(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "labs.pdf"
    pdf_path.write_bytes(b"not a real pdf")
    english_lab_text = (
        "Glucose 103 mg/dL 65-99 H\n"
        "WBC 6.2 x10E3/uL 3.4-10.8\n"
        "Hemoglobin 14.2 g/dL 12-16"
    )
    candidates = [
        ocr_candidates.OcrCandidate(
            engine_name="existing_pdf_pipeline",
            text=english_lab_text,
            quality_score=0.85,
            quality_band="good",
            language="en",
            warnings=[],
            metadata={"quality_metrics": {"cyrillic_ratio": 0.0}},
        )
    ]

    result = ocr_candidates._maybe_cyrillic_ocr_candidate(pdf_path, candidates)

    assert result is None


# ---------------------------------------------------------------------------
# Safety: classification behavior preserved
# ---------------------------------------------------------------------------


def test_phase43_classification_unchanged_by_phase44() -> None:
    # Sanity: the document classifier still labels the same fixtures the same
    # way after Phase 44 imports.
    russian_rx = "Свечи Диклофенак 100 мг по 1 свече на ночь"
    english_lab = "Glucose 103 mg/dL\nWBC 6.2 x10E3/uL\nReference range 65-99"

    rx_result = classify_document(russian_rx)
    lab_result = classify_document(english_lab)

    assert rx_result.document_type == "prescription"
    assert lab_result.document_type == "lab_report"


def test_cyrillic_ocr_does_not_auto_accept() -> None:
    # The candidate dataclass has no "accepted" field — only the pipeline
    # produces accepted/review status, behind safety gates.
    cap = OcrCapabilityReport(
        tesseract_available=True,
        tesseract_path="/usr/bin/tesseract",
        available_languages=["eng", "rus"],
        russian_available=True,
        english_available=True,
        warnings=[],
        metadata={},
    )
    sample = ocr_candidates.OcrCandidate(
        engine_name="tesseract_rus_eng",
        text="Глюкоза 5.6 ммоль/л",
        quality_score=0.95,
        quality_band="good",
        language="ru",
        warnings=[],
        metadata={},
    )

    assert not hasattr(sample, "status")
    assert not hasattr(sample, "accepted")
    assert cap.russian_available is True


def test_poor_cyrillic_ocr_remains_review_candidate(tmp_path: Path) -> None:
    # If Cyrillic OCR returns garbage (empty/very short), the candidate quality
    # band must reflect that — score will be poor, band 'empty' or 'poor_ocr'.
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"not a real pdf")
    candidate = ocr_candidates.build_candidate(
        engine_name="tesseract_rus_eng",
        text="",
        source_path=pdf_path,
        metadata={},
    )

    assert candidate.quality_band in {"empty", "poor_ocr"}
    assert candidate.quality_score < 0.50
