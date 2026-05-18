from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.document_type_registry import (
    IMAGING_REPORT_LABEL,
    LAB_RESULT_LABEL,
    TREATMENT_PLAN_LABEL,
    UNKNOWN_DOCUMENT_LABEL,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts import run_medai_doc_type_batch_eval as eval01


RAW_FILENAME = "private_patient_scan.pdf"
RAW_PATH_TEXT = "C:\\private\\patient\\private_patient_scan.pdf"
RAW_OCR_TEXT = "raw extracted document text should not appear"


def _fake_result(
    *,
    document_type: str,
    cue_keys: list[str],
    ambiguous: list[str] | None = None,
    outcome: str = "queued_for_review",
    validation_status: str = "needs_review",
    fallback: bool = False,
    language_text_visibility: str | None = None,
    auto_accept_allowed: bool = False,
    external_api_used: bool = False,
) -> SimpleNamespace:
    visibility = language_text_visibility or ("incomplete" if fallback else "visible")
    return SimpleNamespace(
        outcome=outcome,
        validation_status=validation_status,
        extractor_result={
            "selected_extractor": "rules_based",
            "confidence": 0.45,
            "raw_text": RAW_OCR_TEXT,
            "document_family_classification_diagnostic": {
                "candidate_family": document_type,
                "matched_family_cue_keys": cue_keys,
                "matched_language_cue_groups": ["synthetic"],
                "ambiguous_candidates": ambiguous or [],
                "classification_block_reason": "classified" if document_type != UNKNOWN_DOCUMENT_LABEL else "too_few_safe_family_cue_keys",
                "conflict_resolution_reason": "none",
                "review_only": True,
                "auto_accept_allowed": auto_accept_allowed,
            },
            "auto_accept_allowed": auto_accept_allowed,
            "ocr_gate_fallback_executed": fallback,
            "language_text_visibility": visibility,
            "ocr_gate_reason": "numeric_table_text_without_cyrillic" if fallback else "cyrillic_visible",
            "ocr_gate_fallback_cyrillic_detected": fallback,
            "ocr_gate_fallback_text_visibility": "recovered" if fallback else None,
            "external_api_used": external_api_used,
        },
        audit={
            "document_type": document_type,
            "ocr_quality_band": "readable",
            "external_api_used": external_api_used,
        },
    )


def test_build_safe_file_record_anonymizes_filename_path_and_raw_text(tmp_path: Path) -> None:
    source = tmp_path / RAW_FILENAME
    source.write_text("synthetic placeholder", encoding="utf-8")

    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_fake_result(document_type=IMAGING_REPORT_LABEL, cue_keys=["imaging_modality"]),
    )
    rendered = json.dumps(record, ensure_ascii=False)

    assert record["file_id"] == "file_001"
    assert record["extension"] == ".pdf"
    assert record["predicted_document_type"] == IMAGING_REPORT_LABEL
    assert RAW_FILENAME not in rendered
    assert RAW_PATH_TEXT not in rendered
    assert RAW_OCR_TEXT not in rendered


def test_report_summarizes_lab_treatment_imaging_unknown_and_ambiguity(tmp_path: Path) -> None:
    paths = []
    for name in ("a.txt", "b.txt", "c.txt", "d.txt"):
        path = tmp_path / name
        path.write_text("synthetic placeholder", encoding="utf-8")
        paths.append(path)

    records = [
        eval01.build_safe_file_record(
            paths[0],
            safe_id="file_001",
            result=_fake_result(document_type=LAB_RESULT_LABEL, cue_keys=["table_header"]),
        ),
        eval01.build_safe_file_record(
            paths[1],
            safe_id="file_002",
            result=_fake_result(document_type=TREATMENT_PLAN_LABEL, cue_keys=["administration_schedule_pattern"]),
        ),
        eval01.build_safe_file_record(
            paths[2],
            safe_id="file_003",
            result=_fake_result(document_type=IMAGING_REPORT_LABEL, cue_keys=["imaging_modality"], fallback=True),
        ),
        eval01.build_safe_file_record(
            paths[3],
            safe_id="file_004",
            result=_fake_result(
                document_type=UNKNOWN_DOCUMENT_LABEL,
                cue_keys=[],
                ambiguous=[IMAGING_REPORT_LABEL, TREATMENT_PLAN_LABEL],
            ),
        ),
    ]
    report = eval01.build_report(records)

    assert report["total_files_evaluated"] == 4
    assert report["count_by_predicted_document_type"][LAB_RESULT_LABEL] == 1
    assert report["count_by_predicted_document_type"][TREATMENT_PLAN_LABEL] == 1
    assert report["count_by_predicted_document_type"][IMAGING_REPORT_LABEL] == 1
    assert report["count_by_predicted_document_type"][UNKNOWN_DOCUMENT_LABEL] == 1
    assert report["ambiguous_family_conflict_count"] == 1
    assert report["ocr_fallback_used_count"] == 1
    assert report["external_api_used_count"] == 0
    assert report["auto_accept_allowed_count"] == 0
    assert report["accepted_count"] == 0
    assert report["recommended_next_actions"]["conflict-resolution update needed"] == 1
    assert report["unknown_diagnostics"]["unknown_total"] == 1
    assert report["unknown_diagnostics"]["unknown_failure_bucket_counts"]["ambiguous_below_threshold"] == 1


def test_unknown_documents_are_review_cases_not_runtime_failures(tmp_path: Path) -> None:
    source = tmp_path / "unknown.txt"
    source.write_text("synthetic placeholder", encoding="utf-8")
    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_fake_result(document_type=UNKNOWN_DOCUMENT_LABEL, cue_keys=[]),
    )

    assert record["predicted_document_type"] == UNKNOWN_DOCUMENT_LABEL
    assert record["review_status"] == "review"
    assert record["classification_block_reason"] == "too_few_safe_family_cue_keys"
    assert record["unknown_failure_bucket"] == "no_safe_document_family_cues"


def test_unknown_accepted_is_flagged_and_source_is_separate_from_auto_accept(tmp_path: Path) -> None:
    source = tmp_path / "unknown_accepted.txt"
    source.write_text("synthetic placeholder", encoding="utf-8")
    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_fake_result(
            document_type=UNKNOWN_DOCUMENT_LABEL,
            cue_keys=[],
            outcome="written",
            validation_status="needs_review",
            auto_accept_allowed=False,
        ),
    )
    report = eval01.build_report([record])

    assert record["raw_review_status"] == "accepted"
    assert record["review_status"] == "review"
    assert record["accepted_status_source"] == "runtime_outcome"
    assert record["status_mapping_action"] == "normalized_unknown_runtime_accepted_to_review"
    assert record["unknown_failure_bucket"] == "status_mapping_anomaly"
    assert "invalid_status_mapping_normalized" in record["status_consistency_flags"]
    assert report["accepted_count"] == 0
    assert report["auto_accept_allowed_count"] == 0
    assert report["accepted_status_source_counts"]["runtime_outcome"] == 1
    assert report["status_consistency"]["unknown_accepted_anomaly_count"] == 1
    assert report["recommended_next_actions"]["status mapping fix"] == 1


def test_unknown_diagnostic_buckets_cover_fallback_text_and_generic_cues(tmp_path: Path) -> None:
    paths = []
    for name in ("fallback.txt", "ocr_needed.txt", "generic.txt", "sparse.txt"):
        path = tmp_path / name
        path.write_text("synthetic placeholder", encoding="utf-8")
        paths.append(path)
    fallback_unknown = eval01.build_safe_file_record(
        paths[0],
        safe_id="file_001",
        result=_fake_result(document_type=UNKNOWN_DOCUMENT_LABEL, cue_keys=[], fallback=True),
    )
    ocr_not_triggered = eval01.build_safe_file_record(
        paths[1],
        safe_id="file_002",
        result=_fake_result(document_type=UNKNOWN_DOCUMENT_LABEL, cue_keys=[]),
    )
    ocr_not_triggered["cyrillic_ocr_recommended"] = True
    ocr_not_triggered["unknown_failure_bucket"] = eval01.unknown_failure_bucket(ocr_not_triggered)
    generic_cues = eval01.build_safe_file_record(
        paths[2],
        safe_id="file_003",
        result=_fake_result(document_type=UNKNOWN_DOCUMENT_LABEL, cue_keys=["result_or_report"]),
    )
    low_visibility = eval01.build_safe_file_record(
        paths[3],
        safe_id="file_004",
        result=_fake_result(
            document_type=UNKNOWN_DOCUMENT_LABEL,
            cue_keys=[],
            language_text_visibility="incomplete",
        ),
    )
    report = eval01.build_report([fallback_unknown, ocr_not_triggered, generic_cues, low_visibility])
    buckets = report["unknown_diagnostics"]["unknown_failure_bucket_counts"]

    assert buckets["fallback_ran_but_no_family_match"] == 1
    assert buckets["OCR_not_triggered"] == 1
    assert buckets["generic_cues_only"] == 1
    assert buckets["insufficient_text_visibility"] == 1
    assert report["unknown_diagnostics"]["unknown_with_fallback_true"] == 1
    assert report["unknown_diagnostics"]["unknown_with_fallback_false"] == 3
    assert report["unknown_diagnostics"]["unknown_with_any_family_cue_keys"] == 1
    assert report["unknown_diagnostics"]["unknown_with_no_family_cue_keys"] == 3
    assert len(report["unknown_diagnostics"]["priority_unknown_samples"]) == 4


def test_external_api_and_auto_accept_policy_failures_are_visible(tmp_path: Path) -> None:
    source = tmp_path / "policy.txt"
    source.write_text("synthetic placeholder", encoding="utf-8")
    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_fake_result(
            document_type=LAB_RESULT_LABEL,
            cue_keys=["table_header"],
            external_api_used=True,
            auto_accept_allowed=True,
        ),
    )
    report = eval01.build_report([record])

    assert report["evaluation_status"] == "failed"
    assert report["external_api_used_count"] == 1
    assert report["auto_accept_allowed_count"] == 1
    assert report["recommended_next_actions"]["external API violation"] == 1
    assert report["recommended_next_actions"]["auto-accept violation"] == 1


def test_write_reports_are_privacy_clean_and_filename_free(tmp_path: Path) -> None:
    source = tmp_path / RAW_FILENAME
    source.write_text("synthetic placeholder", encoding="utf-8")
    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_fake_result(document_type=IMAGING_REPORT_LABEL, cue_keys=["imaging_modality"]),
    )
    report = eval01.build_report([record])
    json_path, md_path = eval01.write_reports(report, tmp_path / "reports")

    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown_payload = {"report": md_path.read_text(encoding="utf-8")}
    rendered = json.dumps(json_payload, ensure_ascii=False) + markdown_payload["report"]

    assert RAW_FILENAME not in rendered
    assert RAW_PATH_TEXT not in rendered
    assert RAW_OCR_TEXT not in rendered
    assert "unknown_failure_bucket" in rendered
    assert "accepted_status_source" in rendered
    empty_report = eval01.build_report([])
    empty_json, empty_md = eval01.write_reports(empty_report, tmp_path / "empty_reports")
    assert check_public_report_payload(json.loads(empty_json.read_text(encoding="utf-8"))).passed is True
    assert check_public_report_payload({"report": empty_md.read_text(encoding="utf-8")}).passed is True


def test_run_batch_evaluation_uses_file_ids_and_writes_reports(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "private_one.txt").write_text("synthetic placeholder", encoding="utf-8")
    (input_dir / "private_two.txt").write_text("synthetic placeholder", encoding="utf-8")
    output_dir = tmp_path / "out"

    def processor(_path: Path, safe_id: str) -> SimpleNamespace:
        label = LAB_RESULT_LABEL if safe_id == "file_001" else IMAGING_REPORT_LABEL
        return _fake_result(document_type=label, cue_keys=["table_header"])

    report = eval01.run_batch_evaluation(
        eval01.EvaluationOptions(input_dir=input_dir, output_dir=output_dir),
        processor=processor,
    )

    assert report["total_files_evaluated"] == 2
    assert [item["file_id"] for item in report["anonymous_per_file_table"]] == ["file_001", "file_002"]
    assert (output_dir / eval01.REPORT_JSON_NAME).exists()
    assert (output_dir / eval01.REPORT_MD_NAME).exists()
