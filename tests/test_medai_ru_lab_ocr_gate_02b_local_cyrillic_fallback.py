from __future__ import annotations

from pathlib import Path

from app.lab_document_metadata import LAB_RESULT_LABEL, classify_lab_document_type
from ingestion.cyrillic_ocr_gate import (
    build_cyrillic_ocr_shadow_marker,
    run_local_cyrillic_ocr_fallback,
)


NUMERIC_TEXT = "\n".join(
    [
        "100 200 300 400 500 600 700 800 900",
        "1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8",
        "10 20 30 40 50 60 70 80 90",
    ]
    * 40
)

CYRILLIC_VISIBLE_TEXT = (
    "Лабораторное исследование "
    "Результаты анализ крови "
    "Показатель Значение Единицы Референсные значения "
    "Гемоглобин значение норма "
    "Лейкоциты значение норма"
) * 6


def _recommended_marker() -> dict:
    return build_cyrillic_ocr_shadow_marker(NUMERIC_TEXT, current_ocr_skipped=True)


def test_fallback_runs_only_when_cyrillic_ocr_recommended_is_true(tmp_path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["eng", "rus"],
        ocr_runner=lambda _path, _language: CYRILLIC_VISIBLE_TEXT,
        document_type_classifier=classify_lab_document_type,
    )

    assert metadata["ocr_gate_fallback_executed"] is True
    assert metadata["ocr_gate_fallback_engine"] == "tesseract_local"
    assert metadata["ocr_gate_fallback_language"] == "rus+eng"


def test_fallback_does_not_run_when_cyrillic_is_already_visible(tmp_path) -> None:
    marker = build_cyrillic_ocr_shadow_marker(CYRILLIC_VISIBLE_TEXT, current_ocr_skipped=True)
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        marker,
        language_lister=lambda: ["eng", "rus"],
        ocr_runner=lambda _path, _language: CYRILLIC_VISIBLE_TEXT,
        document_type_classifier=classify_lab_document_type,
    )

    assert metadata["ocr_gate_fallback_executed"] is False
    assert metadata["ocr_gate_fallback_error_bucket"] == "gate_not_recommended"


def test_fallback_does_not_run_for_no_text_cases_outside_gate(tmp_path) -> None:
    marker = build_cyrillic_ocr_shadow_marker("", current_ocr_skipped=True)
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        marker,
        language_lister=lambda: ["eng", "rus"],
        ocr_runner=lambda _path, _language: CYRILLIC_VISIBLE_TEXT,
        document_type_classifier=classify_lab_document_type,
    )

    assert metadata["ocr_gate_fallback_executed"] is False
    assert metadata["ocr_gate_fallback_error_bucket"] == "gate_not_recommended"


def test_fallback_uses_local_tesseract_metadata_path_only(tmp_path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["eng", "rus"],
        ocr_runner=lambda _path, language: CYRILLIC_VISIBLE_TEXT if language == "rus+eng" else "",
        document_type_classifier=classify_lab_document_type,
    )

    assert metadata["ocr_gate_fallback_engine"] == "tesseract_local"
    assert "cloud" not in str(metadata).lower()
    assert "external_api" not in str(metadata).lower()


def test_fallback_result_can_update_document_type_to_lab_result(tmp_path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["eng", "rus"],
        ocr_runner=lambda _path, _language: CYRILLIC_VISIBLE_TEXT,
        document_type_classifier=classify_lab_document_type,
    )

    assert metadata["ocr_gate_fallback_document_type"] == LAB_RESULT_LABEL


def test_fallback_does_not_change_confidence_or_thresholds(tmp_path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["eng", "rus"],
        ocr_runner=lambda _path, _language: CYRILLIC_VISIBLE_TEXT,
        document_type_classifier=classify_lab_document_type,
    )

    assert "confidence" not in metadata
    assert "threshold" not in metadata


def test_fallback_never_causes_auto_acceptance(tmp_path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["eng", "rus"],
        ocr_runner=lambda _path, _language: CYRILLIC_VISIBLE_TEXT,
        document_type_classifier=classify_lab_document_type,
    )

    assert metadata["ocr_gate_fallback_review_only"] is True
    assert metadata["ocr_gate_fallback_auto_accept_allowed"] is False
    assert "accepted" not in str(metadata).lower()


def test_raw_ocr_text_is_not_in_public_result_structures(tmp_path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["eng", "rus"],
        ocr_runner=lambda _path, _language: CYRILLIC_VISIBLE_TEXT,
        document_type_classifier=classify_lab_document_type,
    )

    assert "raw_text" not in metadata
    assert "ocr_text" not in metadata
    assert CYRILLIC_VISIBLE_TEXT not in str(metadata)


def test_ocr_failure_is_safe_and_keeps_review_metadata(tmp_path) -> None:
    def failing_runner(_path: Path, _language: str) -> str:
        raise RuntimeError("synthetic failure")

    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["eng", "rus"],
        ocr_runner=failing_runner,
        document_type_classifier=classify_lab_document_type,
    )

    assert metadata["ocr_gate_fallback_executed"] is False
    assert metadata["ocr_gate_fallback_attempted"] is True
    assert metadata["ocr_gate_fallback_error_bucket"] == "local_ocr_failed"
    assert metadata["ocr_gate_fallback_review_only"] is True


def test_russian_language_unavailable_is_safe(tmp_path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["eng"],
        ocr_runner=lambda _path, _language: CYRILLIC_VISIBLE_TEXT,
        document_type_classifier=classify_lab_document_type,
    )

    assert metadata["ocr_gate_fallback_executed"] is False
    assert metadata["ocr_gate_fallback_error_bucket"] == "russian_language_unavailable"


def test_external_api_remains_disabled_not_used(tmp_path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus"],
        ocr_runner=lambda _path, _language: CYRILLIC_VISIBLE_TEXT,
        document_type_classifier=classify_lab_document_type,
    )

    assert metadata["ocr_gate_fallback_engine"] == "tesseract_local"
    assert "external_api" not in str(metadata).lower()
    assert "cloud" not in str(metadata).lower()
