from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.document_type_registry import UNKNOWN_DOCUMENT_LABEL
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts import run_medai_doc_type_batch_eval as eval01


RAW_FILENAME = "private_scan.pdf"
RAW_PATH = "C:\\private\\patient\\private_scan.pdf"
RAW_TEXT = "raw OCR text must not be reported"


def _unknown_result(
    *,
    raw_text: str = "",
    visibility: str = "unknown",
    fallback: bool = False,
    validation_status: str = "needs_review",
    external_api_used: bool = False,
    auto_accept_allowed: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        outcome="queued_for_review",
        validation_status=validation_status,
        extractor_result={
            "raw_text": raw_text,
            "language_text_visibility": visibility,
            "ocr_gate_reason": "not_recommended",
            "ocr_gate_fallback_executed": fallback,
            "ocr_gate_fallback_cyrillic_detected": False,
            "ocr_gate_fallback_text_visibility": None,
            "external_api_used": external_api_used,
            "auto_accept_allowed": auto_accept_allowed,
            "document_family_classification_diagnostic": {
                "candidate_family": UNKNOWN_DOCUMENT_LABEL,
                "matched_family_cue_keys": [],
                "ambiguous_candidates": [],
                "classification_block_reason": "too_few_safe_family_cue_keys",
                "conflict_resolution_reason": "none",
            },
        },
        audit={"document_type": UNKNOWN_DOCUMENT_LABEL, "ocr_quality_band": "unknown"},
    )


def test_fallback_false_unknown_gets_safe_ocr_routing_diagnostic_bucket(
    tmp_path: Path,
) -> None:
    source = tmp_path / "safe.txt"
    source.write_text("placeholder", encoding="utf-8")

    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_unknown_result(raw_text="", visibility="unknown", fallback=False),
    )

    assert record["predicted_document_type"] == UNKNOWN_DOCUMENT_LABEL
    assert record["ocr_gate_fallback_executed"] is False
    assert record["native_text_length_bucket"] == "none"
    assert record["language_visibility_status"] == "unknown"
    assert record["unknown_ocr_routing_bucket"] == "routing_not_eligible"


def test_image_like_pdf_not_routed_to_ocr_is_flagged(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "safe.pdf"
    source.write_bytes(b"%PDF-1.4 synthetic")
    monkeypatch.setattr(eval01, "pdf_text_layer_detected", lambda _path: "no")

    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_unknown_result(raw_text="", visibility="unknown", fallback=False),
    )
    report = eval01.build_report([record])

    assert record["image_like_pdf"] == "yes"
    assert record["ocr_fallback_eligible"] == "yes"
    assert record["ocr_fallback_not_triggered_reason"] == "image_like_pdf_but_not_routed_to_ocr"
    assert record["unknown_ocr_routing_bucket"] == "image_like_pdf_but_not_routed_to_ocr"
    assert report["unknown_ocr_routing_diagnostics"]["unknown_image_like_pdfs_not_routed_to_ocr"] == 1
    assert report["unknown_ocr_routing_diagnostics"]["unknown_files_eligible_for_fallback_but_not_triggered"] == 1


def test_text_layer_too_short_unknown_is_flagged(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "safe.pdf"
    source.write_bytes(b"%PDF-1.4 synthetic")
    monkeypatch.setattr(eval01, "pdf_text_layer_detected", lambda _path: "yes")

    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_unknown_result(raw_text="tiny", visibility="visible", fallback=False),
    )
    report = eval01.build_report([record])

    assert record["pdf_text_layer_detected"] == "yes"
    assert record["native_text_length_bucket"] == "tiny"
    assert record["ocr_fallback_not_triggered_reason"] == "text_layer_present_but_too_short"
    assert record["unknown_ocr_routing_bucket"] == "text_layer_present_but_too_short"
    assert report["unknown_ocr_routing_diagnostics"]["unknown_text_layer_pdfs_with_too_little_text"] == 1


def test_unknown_accepted_runtime_validation_status_is_flagged_without_fixing(tmp_path: Path) -> None:
    source = tmp_path / "safe.txt"
    source.write_text("placeholder", encoding="utf-8")

    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_unknown_result(validation_status="accepted"),
    )
    report = eval01.build_report([record])

    assert record["accepted_status_source"] == "runtime_validation_status"
    assert record["raw_review_status"] == "accepted"
    assert record["review_status"] == "review"
    assert record["status_mapping_action"] == "normalized_unknown_runtime_accepted_to_review"
    assert record["unknown_failure_bucket"] == "status_mapping_anomaly"
    assert report["status_consistency"]["unknown_accepted_anomaly_file_ids"] == ["file_001"]


def test_eval02_report_payload_remains_privacy_clean(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / RAW_FILENAME
    source.write_bytes(b"%PDF-1.4 synthetic")
    monkeypatch.setattr(eval01, "pdf_text_layer_detected", lambda _path: "no")
    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_unknown_result(raw_text=RAW_TEXT, visibility="unknown", fallback=False),
    )
    report = eval01.build_report([record])
    json_path, md_path = eval01.write_reports(report, tmp_path / "reports")

    rendered = (
        json.dumps(json.loads(json_path.read_text(encoding="utf-8")), ensure_ascii=False)
        + md_path.read_text(encoding="utf-8")
    )
    assert RAW_FILENAME not in rendered
    assert RAW_PATH not in rendered
    assert RAW_TEXT not in rendered
    assert report["external_api_used_count"] == 0
    assert report["auto_accept_allowed_count"] == 0

    empty_report = eval01.build_report([])
    empty_json, empty_md = eval01.write_reports(empty_report, tmp_path / "empty_reports")
    assert check_public_report_payload(json.loads(empty_json.read_text(encoding="utf-8"))).passed is True
    assert check_public_report_payload({"report": empty_md.read_text(encoding="utf-8")}).passed is True
