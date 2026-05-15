from __future__ import annotations

from types import SimpleNamespace

from app.operator_safety import build_result_summary, operator_guidance, privacy_mode_labels


def test_phase49_operator_guidance_for_accepted():
    assert operator_guidance("accepted") == "Usable, but still check before relying on it."


def test_phase49_operator_guidance_for_review():
    assert operator_guidance("review") == "MedAI is unsure; compare with the source file."


def test_phase49_operator_guidance_for_review_ocr_quality():
    assert operator_guidance("review_ocr_quality") == "File quality is too low; re-scan or upload a clearer copy."


def test_phase49_operator_guidance_for_empty():
    assert operator_guidance("empty") == "MedAI could not read useful text."


def test_phase49_privacy_mode_labels_render_expected_text():
    local = privacy_mode_labels(local_only=True, allow_external_api=False, require_pii_scrub=True)
    external = privacy_mode_labels(local_only=False, allow_external_api=True, require_pii_scrub=True)

    assert local.local_only == "ON"
    assert local.external_apis == "DISABLED"
    assert local.pii_scrub_required == "YES"
    assert "Local-only mode active" in local.warning
    assert external.local_only == "OFF"
    assert external.external_apis == "ENABLED"
    assert "External APIs are enabled" in external.warning


def test_phase49_result_summary_includes_operator_privacy_fields():
    result = SimpleNamespace(
        outcome="queued_for_review",
        validation_status="needs_review",
        validation_errors=[{"code": "low_confidence"}],
        audit={
            "input_quality_band": "good",
            "selected_engine": "existing_text",
            "reason_codes": ["extraction_low_confidence"],
            "lab_table_detected": True,
            "parsed_lab_row_count": 4,
            "lab_coverage_band": "partial",
            "document_type": "lab_report",
        },
        extractor_result={
            "privacy_gate": {
                "mode": "external_allowed_redacted",
                "allowed": True,
                "provider": "gemini",
                "payload_redacted": True,
            }
        },
    )

    summary = build_result_summary(result)

    assert summary["final_status"] == "review"
    assert summary["operator_next_action"] == "MedAI is unsure; compare with the source file."
    assert summary["privacy_gate_status"] == "external_allowed_redacted"
    assert summary["external_api_used"] is True
    assert summary["payload_redacted"] is True
