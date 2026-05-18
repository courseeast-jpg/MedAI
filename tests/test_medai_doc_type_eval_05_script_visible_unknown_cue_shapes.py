from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.document_type_registry import LAB_RESULT_LABEL, UNKNOWN_DOCUMENT_LABEL
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts import run_medai_doc_type_batch_eval as eval01


RAW_FILENAME = "private_shape_case.pdf"
RAW_TEXT = "raw private extracted document text must not appear"


def _unknown_result(raw_text: str, *, external_api_used: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        outcome="queued_for_review",
        validation_status="needs_review",
        audit={"document_type": UNKNOWN_DOCUMENT_LABEL, "ocr_quality_band": "readable"},
        extractor_result={
            "raw_text": raw_text,
            "language_text_visibility": "unknown",
            "external_api_used": external_api_used,
            "auto_accept_allowed": False,
            "document_family_classification_diagnostic": {
                "candidate_family": UNKNOWN_DOCUMENT_LABEL,
                "matched_family_cue_keys": [],
                "ambiguous_candidates": [],
                "classification_block_reason": "too_few_safe_family_cue_keys",
                "conflict_resolution_reason": "none",
            },
        },
    )


def _record(tmp_path: Path, raw_text: str) -> dict:
    source = tmp_path / "safe.txt"
    source.write_text("placeholder", encoding="utf-8")
    return eval01.build_safe_file_record(source, safe_id="file_001", result=_unknown_result(raw_text))


def test_latin_script_visible_unknown_with_table_shape_gets_table_bucket(tmp_path: Path) -> None:
    record = _record(
        tmp_path,
        "Name       12.4\nValue      14.1\nTotal      26.5",
    )

    assert record["predicted_document_type"] == UNKNOWN_DOCUMENT_LABEL
    assert record["dominant_script"] == "latin"
    assert record["table_like_structure_detected"] == "yes"
    assert record["auto_accept_allowed"] is False


def test_latin_mri_shape_is_possible_imaging_without_family_label(tmp_path: Path) -> None:
    record = _record(
        tmp_path,
        "MRI report\nDescription section\nConclusion section\nT1 T2 FLAIR DWI",
    )

    assert record["predicted_document_type"] == UNKNOWN_DOCUMENT_LABEL
    assert record["imaging_modality_shape_detected"] == "yes"
    assert record["cue_audit_result"] == "possible_imaging_shape_without_language_cues"


def test_latin_lab_table_shape_can_be_promoted_by_family03_structure_cues(tmp_path: Path) -> None:
    record = _record(
        tmp_path,
        "Component Result Unit Reference\nGlucose 92 mg/dL 70-99\nCreatinine 0.9 mg/dL 0.6-1.2",
    )

    assert record["predicted_document_type"] == LAB_RESULT_LABEL
    assert "lab_table_column_structure" in record["matched_safe_cue_keys"]
    assert record["cue_audit_result"] == "not_applicable"
    assert record["review_status"] == "review"


def test_header_noise_shape_gets_header_noise_bucket(tmp_path: Path) -> None:
    record = _record(tmp_path, "Page header copy scan footer")

    assert record["predicted_document_type"] == UNKNOWN_DOCUMENT_LABEL
    assert record["cue_audit_result"] == "likely_nonmedical_or_header_noise"


def test_script_visible_unknown_cue_audit_summary_counts_safe_shapes(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, "MRI report Description Conclusion T1 T2"),
        _record(tmp_path, "Component Result Unit Reference\nGlucose 92 mg/dL 70-99"),
        _record(tmp_path, "Page header copy scan footer"),
    ]
    report = eval01.build_report(records)
    audit = report["script_visible_unknown_cue_audit"]

    assert audit["script_visible_unknown_total"] == 2
    assert audit["imaging_like_shape"] == 1
    assert audit["likely_header_noise"] == 1
    assert "possible_lab_shape_without_language_cues" not in audit["cue_audit_result_counts"]


def test_eval05_reports_are_privacy_clean_and_do_not_emit_raw_text(tmp_path: Path) -> None:
    source = tmp_path / RAW_FILENAME
    source.write_text("placeholder", encoding="utf-8")
    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_unknown_result(RAW_TEXT, external_api_used=False),
    )
    report = eval01.build_report([record])
    json_path, md_path = eval01.write_reports(report, tmp_path / "reports")
    rendered = (
        json.dumps(json.loads(json_path.read_text(encoding="utf-8")), ensure_ascii=False)
        + md_path.read_text(encoding="utf-8")
    )

    assert RAW_FILENAME not in rendered
    assert RAW_TEXT not in rendered
    assert report["external_api_used_count"] == 0
    assert report["auto_accept_allowed_count"] == 0
    assert record["review_status"] == "review"
    assert record["predicted_document_type"] == UNKNOWN_DOCUMENT_LABEL
    assert check_public_report_payload(json.loads(json_path.read_text(encoding="utf-8"))).passed is True
    assert check_public_report_payload({"report": md_path.read_text(encoding="utf-8")}).passed is True
