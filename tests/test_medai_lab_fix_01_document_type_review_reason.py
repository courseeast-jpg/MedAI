from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import app.test_launcher as launcher
from app.lab_document_metadata import (
    LAB_RESULT_LABEL,
    UNKNOWN_DOCUMENT_LABEL,
    URINALYSIS_LABEL,
    classify_lab_document_type,
    normalize_text_quality_label,
    review_reason_for_result,
)
from app.main import item_status, operator_review_reason_for_item, visible_reason_codes


def test_synthetic_urinalysis_text_classified_as_urinalysis() -> None:
    text = """
    Urinalysis
    Specific gravity 1.020
    pH 6.0
    Protein negative
    Leukocyte esterase trace
    RBC 0-2
    WBC 0-5
    """

    assert classify_lab_document_type(text) == URINALYSIS_LABEL


def test_synthetic_lab_result_text_classified_as_lab_result() -> None:
    text = """
    Laboratory report
    CBC result flag unit reference range
    WBC 5.2 unit reference range
    HGB 14.1 unit reference range
    """

    assert classify_lab_document_type(text) == LAB_RESULT_LABEL


def test_weak_irrelevant_text_remains_unknown() -> None:
    assert classify_lab_document_type("Appointment reminder. Bring forms.") == UNKNOWN_DOCUMENT_LABEL


def test_lab_detection_does_not_change_auto_acceptance_or_thresholds(tmp_path, monkeypatch) -> None:
    source = tmp_path / "synthetic.pdf"
    source.write_bytes(b"synthetic")
    review_dir = tmp_path / "review"
    archive_dir = tmp_path / "archive"
    monkeypatch.setattr(launcher, "TEST_REVIEW_DIR", review_dir)
    monkeypatch.setattr(launcher, "TEST_ARCHIVE_DIR", archive_dir)

    class FakePipeline:
        def process_pdf(self, source_path: Path, *, specialty: str, session_id: str):
            return SimpleNamespace(
                outcome="queued_for_review",
                validation_status="rejected",
                validation_errors=[{"code": "confidence_below_reject_threshold"}],
                extractor_result={
                    "raw_text": "Urinalysis result specific gravity pH protein glucose RBC WBC reference range",
                    "selected_extractor": "spacy",
                    "confidence": 0.45,
                    "text_quality_status": "readable_native",
                },
                audit={},
            )

    result = launcher._process_one_file(FakePipeline(), source, specialty="general", run_id="run_1")

    assert result.status == "review"
    assert result.validation_status == "rejected"
    assert result.document_type == URINALYSIS_LABEL
    assert result.ocr_quality_band == "Native text"
    assert result.operator_reason_label == "Low confidence"
    assert result.operator_review_reason == (
        "Needs review: lab-style document detected, but confidence is below the acceptance gate."
    )
    assert not archive_dir.exists()
    assert (review_dir / "synthetic.pdf").exists()


def test_low_confidence_lab_result_remains_needs_review() -> None:
    item = {
        "status": "review",
        "validation_status": "rejected",
        "document_type": LAB_RESULT_LABEL,
        "confidence": 0.45,
    }

    assert item_status(item) == "review"
    assert operator_review_reason_for_item(item) == (
        "Needs review: lab-style document detected, but confidence is below the acceptance gate."
    )


def test_unknown_document_review_reason_is_clear() -> None:
    assert review_reason_for_result(
        document_type=UNKNOWN_DOCUMENT_LABEL,
        validation_status="needs_review",
        confidence=0.63,
        status="review",
    ) == "Needs review: MedAI could not confidently identify this document type."


def test_ui_status_mapping_hides_unexplained_rejected_chip() -> None:
    item = {
        "status": "review",
        "validation_status": "rejected",
        "document_type": LAB_RESULT_LABEL,
        "confidence": 0.45,
    }

    assert item_status(item) == "review"
    assert visible_reason_codes(["rejected"]) == []
    assert operator_review_reason_for_item(item).startswith("Needs review:")


def test_external_api_and_clinical_interpretation_are_not_added() -> None:
    reason = review_reason_for_result(
        document_type=URINALYSIS_LABEL,
        validation_status="rejected",
        confidence=0.45,
        status="review",
    )

    assert "diagnosis" not in reason.lower()
    assert "normal" not in reason.lower()
    assert "abnormal" not in reason.lower()
    assert normalize_text_quality_label(None) == "Not checked"


def test_accepted_status_guidance_is_not_changed_to_review() -> None:
    assert review_reason_for_result(
        document_type=LAB_RESULT_LABEL,
        validation_status="accepted",
        confidence=0.9,
        status="accepted",
    ) == "Usable, but still check before relying on it."
