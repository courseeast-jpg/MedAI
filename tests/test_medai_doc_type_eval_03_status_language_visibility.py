from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.document_type_registry import LAB_RESULT_LABEL, UNKNOWN_DOCUMENT_LABEL
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts import run_medai_doc_type_batch_eval as eval01


RAW_FILENAME = "private_language_case.pdf"
RAW_TEXT = "private raw text must not appear"


def _result(
    *,
    document_type: str = UNKNOWN_DOCUMENT_LABEL,
    cue_keys: list[str] | None = None,
    raw_text_marker: object = RAW_TEXT,
    visibility: str | None = "unknown",
    validation_status: str = "needs_review",
    outcome: str = "queued_for_review",
    auto_accept_allowed: bool = False,
) -> SimpleNamespace:
    extractor_result = {
        "document_family_classification_diagnostic": {
            "candidate_family": document_type,
            "matched_family_cue_keys": cue_keys or [],
            "ambiguous_candidates": [],
            "classification_block_reason": "classified"
            if document_type != UNKNOWN_DOCUMENT_LABEL
            else "too_few_safe_family_cue_keys",
            "conflict_resolution_reason": "none",
        },
        "auto_accept_allowed": auto_accept_allowed,
    }
    if raw_text_marker is not None:
        extractor_result["raw_text"] = raw_text_marker
    if visibility is not None:
        extractor_result["language_text_visibility"] = visibility
        extractor_result["ocr_gate_reason"] = "detector_returned_unknown"
    return SimpleNamespace(
        outcome=outcome,
        validation_status=validation_status,
        extractor_result=extractor_result,
        audit={"document_type": document_type, "ocr_quality_band": "unknown"},
    )


def test_unknown_runtime_accepted_is_normalized_to_review_without_auto_accept(tmp_path: Path) -> None:
    source = tmp_path / "safe.txt"
    source.write_text("placeholder", encoding="utf-8")

    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_result(validation_status="accepted"),
    )
    report = eval01.build_report([record])

    assert record["predicted_document_type"] == UNKNOWN_DOCUMENT_LABEL
    assert record["raw_review_status"] == "accepted"
    assert record["review_status"] == "review"
    assert record["accepted_status_source"] == "runtime_validation_status"
    assert record["status_mapping_action"] == "normalized_unknown_runtime_accepted_to_review"
    assert "invalid_status_mapping_normalized" in record["status_consistency_flags"]
    assert report["accepted_count"] == 0
    assert report["auto_accept_allowed_count"] == 0
    assert report["status_consistency"]["unknown_accepted_anomaly_file_ids"] == ["file_001"]


def test_known_accepted_document_type_is_unaffected(tmp_path: Path) -> None:
    source = tmp_path / "safe.txt"
    source.write_text("placeholder", encoding="utf-8")

    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_result(
            document_type=LAB_RESULT_LABEL,
            cue_keys=["table_header"],
            validation_status="accepted",
        ),
    )
    report = eval01.build_report([record])

    assert record["predicted_document_type"] == LAB_RESULT_LABEL
    assert record["raw_review_status"] == "accepted"
    assert record["review_status"] == "accepted"
    assert record["status_mapping_action"] == "unchanged"
    assert report["accepted_count"] == 1
    assert report["auto_accept_allowed_count"] == 0


def test_language_visibility_audit_detects_detector_not_called(tmp_path: Path) -> None:
    source = tmp_path / "safe.txt"
    source.write_text("placeholder", encoding="utf-8")

    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_result(raw_text_marker="visible alphabetic text", visibility=None),
    )
    report = eval01.build_report([record])

    assert record["text_source_present"] == "yes"
    assert record["text_extraction_attempted"] == "yes"
    assert record["language_detector_attempted"] == "no"
    assert record["language_detector_input_bucket"] == "tiny"
    assert record["script_detection_attempted"] == "yes"
    assert record["script_detection_result"] == "latin"
    assert record["visibility_unknown_reason"] == "detector_not_called"
    assert report["language_visibility_audit"]["unknown_visibility_detector_not_called"] == 1


def test_language_visibility_audit_detects_numeric_only_text(tmp_path: Path) -> None:
    source = tmp_path / "safe.txt"
    source.write_text("placeholder", encoding="utf-8")

    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_result(raw_text_marker="12345 67890", visibility="unknown"),
    )
    report = eval01.build_report([record])

    assert record["script_detection_result"] == "numeric_only"
    assert record["visibility_unknown_reason"] == "numeric_or_symbol_only_text"
    assert report["language_visibility_audit"]["unknown_visibility_numeric_or_symbol_only_text"] == 1


def test_language_visibility_audit_detects_missing_metadata(tmp_path: Path) -> None:
    source = tmp_path / "safe.txt"
    source.write_text("placeholder", encoding="utf-8")

    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_result(raw_text_marker=None, visibility=None),
    )
    report = eval01.build_report([record])

    assert record["text_source_present"] == "unknown"
    assert record["visibility_unknown_reason"] == "text_not_passed_to_visibility_detector"
    assert report["language_visibility_audit"]["unknown_visibility_text_not_passed_to_detector"] == 1


def test_eval03_report_fields_are_privacy_clean(tmp_path: Path) -> None:
    source = tmp_path / RAW_FILENAME
    source.write_text("placeholder", encoding="utf-8")
    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_result(raw_text_marker=RAW_TEXT, visibility="unknown"),
    )
    report = eval01.build_report([record])
    json_path, md_path = eval01.write_reports(report, tmp_path / "reports")

    rendered = (
        json.dumps(json.loads(json_path.read_text(encoding="utf-8")), ensure_ascii=False)
        + md_path.read_text(encoding="utf-8")
    )
    assert RAW_FILENAME not in rendered
    assert RAW_TEXT not in rendered
    assert "language_visibility_audit" in rendered
    assert "visibility_unknown_reason" in rendered

    empty_report = eval01.build_report([])
    empty_json, empty_md = eval01.write_reports(empty_report, tmp_path / "empty_reports")
    assert check_public_report_payload(json.loads(empty_json.read_text(encoding="utf-8"))).passed is True
    assert check_public_report_payload({"report": empty_md.read_text(encoding="utf-8")}).passed is True
