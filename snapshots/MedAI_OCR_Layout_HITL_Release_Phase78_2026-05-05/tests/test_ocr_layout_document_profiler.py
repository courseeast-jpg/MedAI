from __future__ import annotations

from pathlib import Path

from ocr_layout.document_profiler import profile_document


def test_profiles_text_file_language_and_density(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    text = "Patient urine culture result negative with normal glucose and protein. " * 8

    profile = profile_document(source, text)

    assert profile.input_type == "text_file"
    assert profile.language == "en"
    assert profile.low_text_density is False
    assert profile.empty_or_near_empty is False


def test_profiles_scanned_pdf_from_ocr_audit(tmp_path: Path) -> None:
    source = tmp_path / "scan.pdf"
    text = "URINE RESULT NEGATIVE RBC WBC NITRITE " * 5

    profile = profile_document(
        source,
        text,
        extraction_metadata={
            "ocr_fallback_used": True,
            "page_audits": [{"page": 1, "native_text_length": 0, "ocr_fallback_used": True}],
        },
    )

    assert profile.input_type == "scanned_pdf"
    assert profile.page_count == 1
    assert profile.low_text_density is False


def test_profiles_mixed_pdf_and_table_layout(tmp_path: Path) -> None:
    source = tmp_path / "mixed.pdf"
    text = "Analyte | Result | Flag\nRBC | 4 | H\nWBC | 2 | N\n" * 8

    profile = profile_document(
        source,
        text,
        extraction_metadata={
            "page_audits": [
                {"page": 1, "native_text_length": 300, "ocr_fallback_used": False},
                {"page": 2, "native_text_length": 20, "ocr_fallback_used": True},
            ],
        },
    )

    assert profile.input_type == "mixed_pdf"
    assert profile.table_layout_heavy is True
    assert "layout_or_table_heavy" in profile.warnings


def test_profiles_cyrillic_language(tmp_path: Path) -> None:
    source = tmp_path / "ru.txt"
    text = "Пациент анализ мочи результат отрицательный белок глюкоза норма " * 5

    profile = profile_document(source, text)

    assert profile.language == "ru"
