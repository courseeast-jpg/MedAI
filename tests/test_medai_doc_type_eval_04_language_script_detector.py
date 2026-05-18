from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.document_type_registry import UNKNOWN_DOCUMENT_LABEL
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts import run_medai_doc_type_batch_eval as eval01


RAW_FILENAME = "private_detector_case.pdf"
RAW_TEXT = "raw extracted text must not appear"


def _unknown_result(
    *,
    raw_text: str = "",
    visibility: str = "unknown",
    language_confidence: float | None = None,
    auto_accept_allowed: bool = False,
    external_api_used: bool = False,
) -> SimpleNamespace:
    extractor_result = {
        "raw_text": raw_text,
        "language_text_visibility": visibility,
        "ocr_gate_reason": "detector_returned_unknown",
        "auto_accept_allowed": auto_accept_allowed,
        "external_api_used": external_api_used,
        "document_family_classification_diagnostic": {
            "candidate_family": UNKNOWN_DOCUMENT_LABEL,
            "matched_family_cue_keys": [],
            "ambiguous_candidates": [],
            "classification_block_reason": "too_few_safe_family_cue_keys",
            "conflict_resolution_reason": "none",
        },
    }
    if language_confidence is not None:
        extractor_result["language_support"] = {
            "language_confidence": language_confidence,
            "detected_language": "unknown",
        }
    return SimpleNamespace(
        outcome="queued_for_review",
        validation_status="needs_review",
        extractor_result=extractor_result,
        audit={"document_type": UNKNOWN_DOCUMENT_LABEL, "ocr_quality_band": "unknown"},
    )


def _record(tmp_path: Path, result: SimpleNamespace) -> dict:
    source = tmp_path / "safe.txt"
    source.write_text("placeholder", encoding="utf-8")
    return eval01.build_safe_file_record(source, safe_id="file_001", result=result)


def test_numeric_heavy_text_gets_detector_numeric_bucket(tmp_path: Path) -> None:
    record = _record(tmp_path, _unknown_result(raw_text="12345 67890 24680 13579"))

    assert record["script_detection_result"] == "numeric_only"
    assert record["language_script_detector_unknown_bucket"] == "detector_input_numeric_heavy"
    assert record["language_script_visibility"] == "not_applicable"


def test_cyrillic_text_with_unknown_language_gets_script_visibility(tmp_path: Path) -> None:
    record = _record(tmp_path, _unknown_result(raw_text="пример кириллического текста"))
    report = eval01.build_report([record])

    assert record["script_detection_result"] == "cyrillic"
    assert record["language_visibility_status"] == "cyrillic_visible_language_unknown"
    assert record["language_script_visibility"] == "cyrillic_visible_language_unknown"
    assert record["language_script_detector_unknown_bucket"] == "script_detectable_language_unknown"
    assert report["language_script_detector_unknown_diagnostics"]["cyrillic_present_but_language_unknown"] == 1


def test_latin_text_with_unknown_language_gets_script_visibility(tmp_path: Path) -> None:
    record = _record(tmp_path, _unknown_result(raw_text="visible latin alphabetic text"))
    report = eval01.build_report([record])

    assert record["script_detection_result"] == "latin"
    assert record["language_visibility_status"] == "latin_visible_language_unknown"
    assert record["language_script_visibility"] == "latin_visible_language_unknown"
    assert record["language_script_detector_unknown_bucket"] == "script_detectable_language_unknown"
    assert report["language_script_detector_unknown_diagnostics"]["latin_present_but_language_unknown"] == 1


def test_mixed_script_gets_mixed_bucket(tmp_path: Path) -> None:
    record = _record(tmp_path, _unknown_result(raw_text="latin пример mixed"))
    report = eval01.build_report([record])

    assert record["script_detection_result"] == "mixed"
    assert record["language_script_detector_unknown_bucket"] == "detector_input_mixed_script"
    assert record["language_script_visibility"] == "mixed_script_visible_language_unknown"
    assert report["language_script_detector_unknown_diagnostics"]["mixed_script"] == 1


def test_empty_detector_input_gets_empty_bucket(tmp_path: Path) -> None:
    record = _record(tmp_path, _unknown_result(raw_text=""))

    assert record["language_detector_input_bucket"] == "empty"
    assert record["language_script_detector_unknown_bucket"] == "detector_input_empty"
    assert record["visibility_unknown_reason"] == "no_text_available"


def test_low_detector_confidence_gets_low_confidence_bucket(tmp_path: Path) -> None:
    record = _record(
        tmp_path,
        _unknown_result(raw_text="??? abc ??? def ???", language_confidence=0.2),
    )
    # Symbol-heavy inputs outrank low confidence because they explain the detector weakness more directly.
    assert record["language_script_detector_unknown_bucket"] in {
        "detector_input_symbol_heavy",
        "detector_confidence_below_threshold",
    }


def test_eval04_public_report_fields_are_privacy_clean(tmp_path: Path) -> None:
    source = tmp_path / RAW_FILENAME
    source.write_text("placeholder", encoding="utf-8")
    record = eval01.build_safe_file_record(
        source,
        safe_id="file_001",
        result=_unknown_result(raw_text=RAW_TEXT, external_api_used=False, auto_accept_allowed=False),
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
    assert "language_script_detector_unknown_diagnostics" in rendered

    empty_report = eval01.build_report([])
    empty_json, empty_md = eval01.write_reports(empty_report, tmp_path / "empty_reports")
    assert check_public_report_payload(json.loads(empty_json.read_text(encoding="utf-8"))).passed is True
    assert check_public_report_payload({"report": empty_md.read_text(encoding="utf-8")}).passed is True
