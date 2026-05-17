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
) -> SimpleNamespace:
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
                "auto_accept_allowed": False,
            },
            "ocr_gate_fallback_executed": fallback,
            "language_text_visibility": "incomplete" if fallback else "visible",
            "ocr_gate_reason": "numeric_table_text_without_cyrillic" if fallback else "cyrillic_visible",
            "ocr_gate_fallback_cyrillic_detected": fallback,
            "ocr_gate_fallback_text_visibility": "recovered" if fallback else None,
            "external_api_used": False,
        },
        audit={
            "document_type": document_type,
            "ocr_quality_band": "readable",
            "external_api_used": False,
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
    assert report["recommended_next_actions"]["conflict-resolution update needed"] == 1


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
    assert check_public_report_payload(json_payload).passed is True
    assert check_public_report_payload(markdown_payload).passed is True


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
