from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.document_type_registry import LAB_RESULT_LABEL, UNKNOWN_DOCUMENT_LABEL
from scripts import run_medai_doc_type_batch_eval as eval01


def _result(
    *,
    document_type: str = UNKNOWN_DOCUMENT_LABEL,
    cue_keys: list[str] | None = None,
    outcome: str = "queued_for_review",
    validation_status: str = "needs_review",
    fallback: bool = False,
    visibility: str = "visible",
    external_api_used: bool = False,
    auto_accept_allowed: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        outcome=outcome,
        validation_status=validation_status,
        extractor_result={
            "raw_text": "private raw text must stay out",
            "language_text_visibility": visibility,
            "ocr_gate_reason": "cyrillic_visible",
            "ocr_gate_fallback_executed": fallback,
            "ocr_gate_fallback_cyrillic_detected": fallback,
            "ocr_gate_fallback_text_visibility": "recovered" if fallback else None,
            "external_api_used": external_api_used,
            "auto_accept_allowed": auto_accept_allowed,
            "document_family_classification_diagnostic": {
                "candidate_family": document_type,
                "matched_family_cue_keys": cue_keys or [],
                "ambiguous_candidates": [],
                "classification_block_reason": "classified"
                if document_type != UNKNOWN_DOCUMENT_LABEL
                else "too_few_safe_family_cue_keys",
                "conflict_resolution_reason": "none",
            },
        },
        audit={"document_type": document_type, "ocr_quality_band": "readable"},
    )


def test_fix_flags_unknown_accepted_as_policy_anomaly(tmp_path: Path) -> None:
    path = tmp_path / "safe.txt"
    path.write_text("placeholder", encoding="utf-8")
    record = eval01.build_safe_file_record(
        path,
        safe_id="file_001",
        result=_result(outcome="written", validation_status="needs_review"),
    )
    report = eval01.build_report([record])

    assert record["predicted_document_type"] == UNKNOWN_DOCUMENT_LABEL
    assert record["review_status"] == "accepted"
    assert record["accepted_status_source"] == "runtime_outcome"
    assert record["unknown_failure_bucket"] == "status_mapping_anomaly"
    assert report["status_consistency"]["unknown_accepted_anomaly_file_ids"] == ["file_001"]


def test_fix_reports_unknown_aggregate_diagnostics(tmp_path: Path) -> None:
    path_a = tmp_path / "a.txt"
    path_b = tmp_path / "b.txt"
    path_a.write_text("placeholder", encoding="utf-8")
    path_b.write_text("placeholder", encoding="utf-8")
    fallback = eval01.build_safe_file_record(path_a, safe_id="file_001", result=_result(fallback=True))
    generic = eval01.build_safe_file_record(
        path_b,
        safe_id="file_002",
        result=_result(cue_keys=["result_or_report"]),
    )
    report = eval01.build_report([fallback, generic])

    assert report["unknown_diagnostics"]["unknown_total"] == 2
    assert report["unknown_diagnostics"]["unknown_with_fallback_true"] == 1
    assert report["unknown_diagnostics"]["unknown_with_any_family_cue_keys"] == 1
    assert len(report["unknown_diagnostics"]["priority_unknown_samples"]) == 2


def test_fix_separates_accepted_from_auto_accept_allowed(tmp_path: Path) -> None:
    path = tmp_path / "accepted.txt"
    path.write_text("placeholder", encoding="utf-8")
    accepted = eval01.build_safe_file_record(
        path,
        safe_id="file_001",
        result=_result(document_type=LAB_RESULT_LABEL, cue_keys=["table_header"], validation_status="accepted"),
    )
    report = eval01.build_report([accepted])

    assert report["accepted_count"] == 1
    assert report["accepted_status_source_counts"]["runtime_validation_status"] == 1
    assert report["auto_accept_allowed_count"] == 0


def test_fix_policy_failure_counts_do_not_hide_external_or_auto_accept(tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_text("placeholder", encoding="utf-8")
    record = eval01.build_safe_file_record(
        path,
        safe_id="file_001",
        result=_result(
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
