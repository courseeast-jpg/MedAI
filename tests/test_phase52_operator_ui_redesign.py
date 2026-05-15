from __future__ import annotations

from pathlib import Path

from app import config
from app.main import PHASE52_OPERATOR_TABS
from app.operator_safety import (
    PHASE52_SAFETY_WARNING,
    PRIVACY_INVARIANT_GUIDANCE,
    detailed_operator_guidance,
    operator_guidance,
    operator_guidance_catalog,
    privacy_mode_labels,
    status_badge,
    status_label,
)


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def app_source() -> str:
    return APP_MAIN.read_text(encoding="utf-8")


def test_phase52_safety_header_text_is_present():
    source = app_source()

    assert "MedAI v2 - OCR / Layout HITL" in source
    assert "Local session" in source
    assert "No run started yet. Upload or select documents to begin." in source
    assert "Build / audit details" in source


def test_phase52_safe_local_mode_and_warning_are_present():
    source = app_source()

    assert "Local safe mode" in source
    assert "Human review" in source
    assert "Cloud APIs off" in source
    assert PHASE52_SAFETY_WARNING in source or "PHASE52_SAFETY_WARNING" in source


def test_phase52_tabs_include_required_operator_tabs():
    assert "Current Run" in PHASE52_OPERATOR_TABS
    assert "Blind Audit" in PHASE52_OPERATOR_TABS
    assert "Report Archive" in PHASE52_OPERATOR_TABS
    assert "Review Package" in PHASE52_OPERATOR_TABS


def test_phase52_raw_json_hidden_by_default():
    source = app_source()

    assert 'st.expander("Show raw run record", expanded=False)' in source
    assert "st.json(item)" in source


def test_phase52_status_label_mapping_and_badges():
    assert status_label("accepted") == "Accepted"
    assert status_label("review") == "Needs review"
    assert status_label("review_ocr_quality") == "OCR / scan review"
    assert status_label("empty") == "No text found"
    assert status_label("error") == "Error"
    assert status_badge("accepted")["color"] == "green"
    assert status_badge("review")["color"] == "amber"
    assert status_badge("review_ocr_quality")["color"] == "orange"
    assert status_badge("empty")["color"] == "gray"
    assert status_badge("error")["color"] == "red"


def test_phase52_operator_guidance_text_is_present():
    catalog = operator_guidance_catalog()

    assert "Usable, but still check" in detailed_operator_guidance("accepted")
    assert "MedAI is unsure" in detailed_operator_guidance("review")
    assert "File quality is too low" in detailed_operator_guidance("review_ocr_quality")
    assert "could not read useful text" in detailed_operator_guidance("empty")
    assert "Processing failed" in detailed_operator_guidance("error")
    assert operator_guidance("review") == "MedAI is unsure; compare with the source file."
    assert catalog["Safety rule"] == "Empty or unclear files cannot be accepted automatically."


def test_phase52_blind_audit_tab_references_real_validation_input():
    source = app_source()

    assert "Put many PDFs into real_validation_input/" in source
    assert "real_validation_input/" in source
    assert "Run Blind Audit" in source


def test_phase52_report_archive_is_separate_from_current_run():
    source = app_source()

    assert "render_current_run_tab" in source
    assert "render_report_archive_tab" in source
    assert "Previous reports live here so current-run counters stay separate" in source


def test_phase52_ui_does_not_change_confidence_thresholds():
    assert config.EXTRACTION_ACCEPT_THRESHOLD == 0.65
    assert config.EXTRACTION_REVIEW_THRESHOLD == 0.50
    assert config.CONSENSUS_ACCEPT_THRESHOLD == 0.60
    assert config.CONSENSUS_REVIEW_THRESHOLD == 0.25


def test_phase52_ui_does_not_disable_local_only_privacy_behavior():
    labels = privacy_mode_labels()

    assert config.MEDAI_LOCAL_ONLY is True
    assert config.MEDAI_ALLOW_EXTERNAL_API is False
    assert labels.local_only == "ON"
    assert labels.external_apis == "DISABLED"
