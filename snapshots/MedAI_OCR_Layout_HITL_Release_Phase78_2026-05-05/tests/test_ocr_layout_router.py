from __future__ import annotations

from pathlib import Path

from ocr_layout.ocr_candidates import build_candidate
from ocr_layout.ocr_router import route_ocr_input


def test_routes_clean_text_to_existing_pipeline(tmp_path: Path) -> None:
    text = (
        "Patient urinalysis result negative. Glucose negative. Protein negative. "
        "RBC normal. WBC normal. Nitrite negative. Culture no growth. "
    ) * 8
    candidate = build_candidate(engine_name="plain_text", text=text, source_path=tmp_path / "clean.txt")

    decision = route_ocr_input([candidate])

    assert decision.route_decision == "digital_clean_text"
    assert decision.selected_engine == "plain_text"
    assert decision.input_quality_band == "good"


def test_routes_scanned_or_low_text_for_ocr_pdf_candidate(tmp_path: Path) -> None:
    text = "URINE RESULT NEGATIVE RBC WBC NITRITE " * 8
    candidate = build_candidate(
        engine_name="existing_pdf_pipeline",
        text=text,
        source_path=tmp_path / "scan.pdf",
        metadata={
            "ocr_fallback_used": True,
            "page_audits": [{"page": 1, "native_text_length": 0, "ocr_fallback_used": True}],
        },
    )

    decision = route_ocr_input([candidate])

    assert decision.route_decision == "scanned_or_low_text"


def test_routes_poor_ocr_to_review_path(tmp_path: Path) -> None:
    poor = build_candidate(
        engine_name="existing_pdf_pipeline",
        text="|||| ____ \ufffd \ufffd 11111111 ~~~~~ BIO MED L4B ??? " * 3,
        source_path=tmp_path / "bad.pdf",
        metadata={"ocr_fallback_used": True},
    )

    decision = route_ocr_input([poor])

    assert decision.route_decision == "poor_ocr"
    assert decision.input_quality_band == "poor_ocr"


def test_selects_best_quality_candidate(tmp_path: Path) -> None:
    poor = build_candidate(
        engine_name="pymupdf_native_text",
        text="|||| ____ ???",
        source_path=tmp_path / "mixed.pdf",
    )
    good = build_candidate(
        engine_name="existing_pdf_pipeline",
        text="Patient urinalysis result negative glucose negative protein negative RBC normal WBC normal. " * 8,
        source_path=tmp_path / "mixed.pdf",
    )

    decision = route_ocr_input([poor, good])

    assert decision.selected_engine == "existing_pdf_pipeline"
    assert decision.selected_text == good.text
