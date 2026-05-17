from __future__ import annotations

from pathlib import Path

from app.operator_safety import (
    PHASE52_SAFETY_WARNING,
    detailed_operator_guidance,
    operator_guidance_catalog,
    status_label,
)


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def app_source() -> str:
    return APP_MAIN.read_text(encoding="utf-8")


def test_current_run_top_labels_are_operator_friendly() -> None:
    source = app_source()

    for text in (
        "Local session",
        "System ready",
        "Medical connector active",
        "No run started yet. Upload or select documents to begin.",
        "Build / audit details",
    ):
        assert text in source
    assert PHASE52_SAFETY_WARNING == (
        "Review required. Not for diagnosis. MedAI does not diagnose, recommend treatment, "
        "interpret medications, or accept extracted values on its own."
    )


def test_safety_chips_are_plain_language() -> None:
    source = app_source()

    for text in (
        "Local safe mode",
        "Human review",
        "Local only",
        "Cloud APIs off",
        "Privacy check on",
    ):
        assert text in source


def test_current_run_document_processing_labels_are_plain_language() -> None:
    source = app_source()

    for text in (
        "Add documents, then start a run.",
        "Supported files: PDF or TXT. Files stay local.",
        "Document category",
        "Choose files",
        "Remove queued files",
        "Start run",
        "Documents waiting",
        "Files ready",
        "Waiting to start",
        "Needs review",
        "OCR / scan review",
        "No text found",
        "No documents added yet. Choose files to begin.",
        "Bad scans and empty results go to review.",
        "Clear last report",
    ):
        assert text in source


def test_result_guide_uses_agreed_plain_language() -> None:
    source = app_source()
    catalog = operator_guidance_catalog()

    assert "Result guide" in source
    assert detailed_operator_guidance("accepted") == "Usable, but still check before relying on it."
    assert detailed_operator_guidance("review") == "MedAI is unsure; compare with the source file."
    assert detailed_operator_guidance("review_ocr_quality") == (
        "File quality is too low; re-scan or upload a clearer copy."
    )
    assert detailed_operator_guidance("empty") == "MedAI could not read useful text."
    assert detailed_operator_guidance("error") == "Processing failed; check the message and try again."
    assert catalog["Safety rule"] == "Empty or unclear files cannot be accepted automatically."


def test_status_labels_use_operator_terms() -> None:
    assert status_label("review") == "Needs review"
    assert status_label("review_ocr_quality") == "OCR / scan review"
    assert status_label("empty") == "No text found"


def test_technical_values_remain_in_audit_expander() -> None:
    source = app_source()
    audit_index = source.index("Build / audit details")

    for text in ("Snapshot:", "Commit:", "Run ID:", "Timestamp:", "Internal connector:"):
        assert source.index(text) > audit_index


def test_heading_anchor_css_is_present() -> None:
    source = app_source()

    assert '[data-testid="stHeaderActionElements"]' in source
    assert "visibility: hidden !important" in source


def test_ui_polish_does_not_change_backend_thresholds() -> None:
    from app import config

    assert config.EXTRACTION_ACCEPT_THRESHOLD == 0.65
    assert config.EXTRACTION_REVIEW_THRESHOLD == 0.50
    assert config.MEDAI_LOCAL_ONLY is True
    assert config.MEDAI_ALLOW_EXTERNAL_API is False
