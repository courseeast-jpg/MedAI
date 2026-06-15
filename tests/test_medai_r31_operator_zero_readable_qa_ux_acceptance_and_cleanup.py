"""Tests for R31 operator-zero readable QA UX acceptance.

Validates the live operator-zero proof artifacts and the code-level UI ordering. Content-
free assertions only (counts, ordering booleans, heading presence, save-persistence).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_r31_operator_zero_readable_qa_ux_acceptance_and_cleanup"
MAIN_PY = REPO_ROOT / "app" / "main.py"
TEXT_REPORTS = ("summary.json", "implementation_report.md", "rendered_ui_text_probe.txt",
                "ui_evidence.json", "operator_zero_acceptance_matrix.json")
SCREENSHOTS = ("screenshot_default_mkb_explorer.png", "screenshot_extracted_record_detail.png",
               "screenshot_not_extracted_record_detail.png")
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"), re.compile(r"ya29\."),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\\\Users\\\\"), re.compile(r"\bMRN\b"), re.compile(r"\bDOB\b"),
]


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_artifacts_exist():
    for name in (*TEXT_REPORTS, *SCREENSHOTS):
        assert (REPORT_DIR / name).exists(), name


def test_overall_pass_operator_zero():
    s = _summary()
    assert s["overall_result"] == "PASS"
    assert s["operator_zero_validation"] is True
    assert s["static_code_only_validation"] is False
    assert s["operator_streamlit_command_tested"] is True
    assert s["operator_streamlit_url"] == "http://localhost:8561"
    assert s["mkb_explorer_visible"] is True


def test_ui_ordering_comparator_first():
    s = _summary()
    assert s["all_record_qa_comparator_first_section"] is True
    assert s["legacy_table_above_comparator"] is False
    assert s["review_staging_detail_above_comparator"] is False
    assert s["raw_json_default_detail"] is False
    assert s["advanced_raw_payload_collapsed"] is True


def test_code_level_ordering():
    # render_mkb_tab must call the comparator section before the legacy section.
    src = MAIN_PY.read_text(encoding="utf-8")
    marker = "def render_mkb_tab("
    body = src[src.index(marker):src.index(marker) + 1200]
    assert "_render_qa_comparator_section()" in body
    assert "_render_legacy_mkb_section(" in body
    assert body.index("_render_qa_comparator_section()") < body.index("_render_legacy_mkb_section(")
    assert "Advanced / legacy MKB staging table" in body


def test_counts_and_headings():
    s = _summary()
    assert s["extracted_payload_count_visible"] == 179
    assert s["not_extracted_count_visible"] == 317
    assert s["readable_extracted_content_visible"] is True
    assert s["readable_extracted_sections_visible"] is True
    assert s["readable_extracted_items_visible"] is True
    assert s["source_evidence_panel_visible"] is True
    assert s["qa_decision_panel_visible"] is True


def test_record_type_readability():
    s = _summary()
    assert s["sample_full_schema_non_placeholder_visible"] is True
    assert s["sample_minimal_review_readable_visible"] is True
    assert s["sample_not_extracted_terminal_reason_visible"] is True


def test_qa_status_save_persisted_live():
    s = _summary()
    assert s["qa_status_save_for_extracted_verified_live"] is True
    assert s["qa_status_save_for_not_extracted_verified_live"] is True
    qa = json.loads((REPORT_DIR / "operator_zero_acceptance_matrix.json").read_text(encoding="utf-8"))
    save = qa["proof_markers"].get("qa_save", {})
    assert save.get("extracted_status") == "needs_manual_review"
    assert save.get("not_extracted_status") == "not_extracted_reviewed"
    assert save.get("active_mkb_write") == "0"


def test_safety_no_side_effects():
    s = _summary()
    for k in ("provider_model_call_made", "live_extraction_started", "new_extraction_started",
              "auto_accept_enabled", "medical_decision_made"):
        assert s[k] is False, k
    assert s["active_verified_records_created"] == 0


def test_no_private_committed_flags():
    s = _summary()
    for k in ("private_artifacts_committed", "raw_text_committed", "rendered_source_images_committed",
              "tokenized_payloads_committed", "token_maps_committed", "pi_values_committed",
              "credentials_or_tokens_committed"):
        assert s[k] is False, k


def test_no_leaks():
    s = _summary()
    assert s["public_report_phi_leak_count"] == 0
    assert s["private_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
    for name in TEXT_REPORTS:
        text = (REPORT_DIR / name).read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            assert not pat.search(text), (name, pat.pattern)


def test_no_ssn_pattern():
    for name in TEXT_REPORTS:
        text = (REPORT_DIR / name).read_text(encoding="utf-8", errors="ignore")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", text)
