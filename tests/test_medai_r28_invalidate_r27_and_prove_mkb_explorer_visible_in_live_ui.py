"""Tests for R28 live UI proof of MKB Explorer and R26 QA comparator visibility."""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_r28_invalidate_r27_and_prove_mkb_explorer_visible_in_live_ui"
MAIN_PATH = REPO_ROOT / "app" / "main.py"
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_medai_r28_invalidate_r27_and_prove_mkb_explorer_visible_in_live_ui.py"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_r28_invalidates_r27_static_only_pass() -> None:
    summary = _summary()
    assert summary["r27_invalid_pass_confirmed"] is True
    assert summary["static_code_only_validation"] is False
    assert summary["live_ui_proof_method"] == "playwright"
    assert "did not start Streamlit" in summary["r27_invalid_pass_explanation"]


def test_mkb_explorer_and_comparator_are_live_visible() -> None:
    summary = _summary()
    assert summary["overall_result"] == "PASS"
    assert summary["mkb_explorer_visible_in_live_ui"] is True
    assert summary["mkb_explorer_visible_by_default"] is True
    assert summary["mkb_explorer_visible_when_advanced_enabled"] is True
    assert summary["advanced_tools_enabled_in_live_probe"] is True
    assert summary["r26_all_record_qa_comparator_visible_in_live_ui"] is True
    assert summary["extracted_payload_queue_visible_in_live_ui"] is True
    assert summary["not_extracted_failure_queue_visible_in_live_ui"] is True


def test_required_counts_are_visible_in_live_probe() -> None:
    summary = _summary()
    assert summary["all_staging_records_count_visible"] == 496
    assert summary["extracted_payload_count_visible"] == 179
    assert summary["not_extracted_count_visible"] == 317
    assert summary["source_preview_available_count_visible"] == 331
    assert summary["source_unavailable_count_visible"] == 165


def test_comparator_queue_labels_are_explicit_in_rendered_ui_source() -> None:
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert "Extracted Payload QA Queue: {qa_counts['extracted_payload_records']}" in source
    assert "Not-Extracted / Failure QA Queue: {qa_counts['not_extracted_records']}" in source


def test_r28_validator_starts_streamlit_and_uses_playwright_not_static_only() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"streamlit"' in source
    assert "sync_playwright" in source
    assert "page.get_by_role(\"tab\", name=\"MKB Explorer\"" in source
    assert "static_code_only_validation" in source


def test_public_reports_have_no_private_paths_or_payloads() -> None:
    forbidden = ["C:\\", "GEMINI_API_KEY", "Bearer ", "Authorization:", "ya29.", "raw_provider_response"]
    for path in REPORT_DIR.glob("*"):
        if not path.is_file() or path.suffix.lower() == ".png":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in forbidden:
            assert marker not in text
