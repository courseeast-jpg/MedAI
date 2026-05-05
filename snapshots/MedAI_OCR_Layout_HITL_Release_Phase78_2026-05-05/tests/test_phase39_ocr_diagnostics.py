from __future__ import annotations

import scripts.run_phase39_ocr_diagnostics as phase39


def test_phase39_report_tracks_mismatch_and_keeps_taxonomy() -> None:
    batch = {
        "results": [
            {
                "filename": "good-but-review.pdf",
                "status": "review_ocr_quality",
                "downstream_classifier_status": "review_ocr_quality",
                "downstream_classifier_reason": "classifier_legacy_ocr_flag,legacy_normalized_low_coverage",
                "classification_reason_codes": ["classifier_legacy_ocr_flag", "legacy_normalized_low_coverage"],
                "entity_count": 6,
                "confidence": 0.7,
                "empty_extraction_flag": False,
                "table_layout_warning": True,
                "normalization_applied": True,
                "ocr_status_mismatch": True,
                "mismatch_type": "good_input_but_downstream_ocr_review",
                "input_quality_band": "good",
                "input_quality_score": 0.82,
                "route_decision": "digital_clean_text",
            },
            {
                "filename": "accepted.pdf",
                "status": "accepted",
                "downstream_classifier_status": "accepted",
                "downstream_classifier_reason": "accepted_clean_input",
                "classification_reason_codes": ["accepted_clean_input"],
                "entity_count": 4,
                "confidence": 0.8,
                "input_quality_band": "good",
                "input_quality_score": 0.9,
                "route_decision": "digital_clean_text",
            },
        ],
    }

    report = phase39.build_phase39_report(phase38_batch=batch, phase38_summary={})

    assert report["phase39_diagnostics"]["ocr_status_mismatch_count"] == 1
    assert report["phase39_diagnostics"]["status_taxonomy_changed"] is False
    assert report["phase39_diagnostics"]["safety_regression"] is False
    assert report["mismatch_files"] == ["good-but-review.pdf"]
    assert "review_extraction_quality" in report["phase40_recommendation"]


def test_phase39_detects_poor_ocr_acceptance_as_safety_regression() -> None:
    batch = {
        "results": [
            {
                "filename": "bad.pdf",
                "status": "accepted",
                "downstream_classifier_status": "accepted",
                "input_quality_band": "poor_ocr",
                "input_quality_score": 0.1,
                "route_decision": "poor_ocr",
                "entity_count": 3,
                "confidence": 0.9,
            }
        ],
    }

    report = phase39.build_phase39_report(phase38_batch=batch, phase38_summary={})

    assert report["phase39_diagnostics"]["safety_regression"] is True
