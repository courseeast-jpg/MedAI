"""Tests for R27 visible MKB Explorer and R26 QA comparator wiring."""
from __future__ import annotations

import json
from pathlib import Path

from app.main import MKB_EXPLORER_TAB, operator_tab_labels
from app.mkb_all_records_qa_comparator import build_all_records_qa_comparator


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = REPO_ROOT / "app" / "main.py"
REPORT_DIR = REPO_ROOT / "reports" / "medai_r27_expose_mkb_explorer_and_r26_qa_comparator_in_visible_ui"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_mkb_explorer_is_visible_in_default_and_advanced_navigation() -> None:
    assert operator_tab_labels(False) == ["Run & Review", "MKB Explorer", "Review Queue"]
    assert MKB_EXPLORER_TAB in operator_tab_labels(True)
    assert operator_tab_labels(True)[:3] == ["Run & Review", "MKB Explorer", "Review Queue"]


def test_runtime_tabs_use_shared_visible_navigation_model() -> None:
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert "tab_labels = operator_tab_labels(show_advanced_tools)" in source
    assert "elif label == MKB_EXPLORER_TAB:" in source
    assert "render_mkb_tab(sys_components)" in source


def test_r26_all_record_qa_comparator_is_reachable_inside_mkb_explorer() -> None:
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert "#### All-record QA comparator" in source
    assert "Extracted Payload QA Queue" in source
    assert "Not-Extracted / Failure QA Queue" in source
    assert "mkb_all_records_qa_queue_mode" in source


def test_all_record_queues_are_populated_for_visible_ui() -> None:
    model = build_all_records_qa_comparator()
    assert model["counts"]["total_staging_records"] == 496
    assert len(model["extracted_queue"]) == 179
    assert len(model["not_extracted_queue"]) == 317
    assert model["counts"]["source_preview_available"] == 331
    assert model["counts"]["source_unavailable"] == 165


def test_r27_summary_required_fields_passed() -> None:
    s = _summary()
    assert s["overall_result"] == "PASS"
    assert s["mkb_explorer_tab_visible"] is True
    assert s["mkb_explorer_visible_by_default"] is True
    assert s["mkb_explorer_visible_when_advanced_enabled"] is True
    assert s["r26_all_record_qa_comparator_visible"] is True
    assert s["extracted_payload_queue_visible"] is True
    assert s["not_extracted_failure_queue_visible"] is True
    assert s["all_staging_records_indexed"] == 496
    assert s["extracted_records_inspectable"] == 179
    assert s["not_extracted_records_inspectable"] == 317
    assert s["source_preview_available"] == 331
    assert s["source_unavailable"] == 165
    assert s["provider_model_call_made"] is False
    assert s["live_extraction_started"] is False
    assert s["new_extraction_started"] is False
    assert s["active_verified_records_created"] == 0
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["privacy_result"] == "passed"
    assert s["safety_result"] == "passed"


def test_public_reports_have_no_private_paths_or_payloads() -> None:
    forbidden = ["C:\\", "GEMINI_API_KEY", "Bearer ", "Authorization:", "ya29.", "raw_provider_response"]
    for path in REPORT_DIR.glob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in forbidden:
            assert marker not in text
