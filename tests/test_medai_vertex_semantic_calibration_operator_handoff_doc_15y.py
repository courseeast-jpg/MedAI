"""Docs-only tests for the 15Y Vertex synthetic calibration handoff guide."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "operator_handoff" / "MEDAI_VERTEX_SYNTHETIC_CALIBRATION_HANDOFF_15Y.md"

REQUIRED_SECTIONS = [
    "## 1. Purpose", "## 2. Calibration status", "## 3. What 15X-R4 proves",
    "## 4. What 15X-R4 does NOT prove", "## 5. Final synthetic calibration metrics",
    "## 6. Category coverage (12)", "## 7. Guardrails that held",
    "## 8. Contract hardenings created during calibration", "## 9. Operator interpretation",
    "## 10. Required gates before real-document routing", "## 11. Recommended next implementation stage",
    "## 12. Evidence chain", "## 13. Troubleshooting / stop conditions",
]
CATEGORIES = [
    "portal_result_cards", "cytology_pathology_narrative", "urinalysis_table_like_lab",
    "mixed_narrative_numeric_result", "short_clinical_note_negation", "short_clinical_note_uncertainty",
    "medication_mention_no_ddi", "bilingual_cyrillic_snippet", "sparse_low_information_result",
    "multi_section_explicit_unknowns", "abnormal_numeric_with_units", "normal_numeric_with_units",
]


def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def test_handoff_doc_exists() -> None:
    assert DOC.exists()


def test_all_required_sections_present() -> None:
    text = _text()
    for s in REQUIRED_SECTIONS:
        assert s in text, s


def test_15xr4_pass_and_fixture_pass_stated() -> None:
    text = _text()
    assert "15X-R4: full synthetic calibration PASS" in text
    assert "15/15 fixtures passed" in text


def test_twelve_categories_listed() -> None:
    text = _text()
    for cat in CATEGORIES:
        assert cat in text


def test_final_metrics_present() -> None:
    text = _text()
    for tok in (
        "fixture_count = **15**", "live_call_count = **15**", "schema_validation_pass_count = **15**",
        "verbatim_evidence_anchor_pass_count = **15**", "hallucinated_field_count = **0**",
        "review_required_count = **15**", "active_written_count = **0**", "active_mkb_record_created_count = **0**",
        "auto_accept = **false**", "total_token_count_all_calls = **7334**",
        "estimated_total_cost_usd_all_calls = **$0.0014486**", "billing_check_pending = **true**",
    ):
        assert tok in text


def test_token_cost_summary_present() -> None:
    text = _text()
    assert "7334" in text and "$0.0014486" in text


def test_evidence_anchor_guardrails_present() -> None:
    text = _text()
    assert "verbatim source span" in text
    assert "paraphrase is rejected" in text.lower()
    assert "no embeddings / llm judge / fuzzy evidence" in text.lower()


def test_bilingual_label_guardrails_present() -> None:
    text = _text()
    assert "declared label aliases only" in text
    assert "undeclared label drift rejected" in text.lower()
    assert "section mismatch rejected" in text.lower()


def test_real_document_boundary_present() -> None:
    text = _text()
    assert "clear unrestricted real medical documents" in text
    assert "safe to process real documents" in text
    assert "not real-document clearance" in text.lower()


def test_no_active_write_no_auto_accept_review_bound_present() -> None:
    text = _text()
    assert "No active MKB write" in text
    assert "No auto-accept" in text
    assert "Review-bound only" in text


def test_future_real_doc_gates_present() -> None:
    text = _text()
    for g in (
        "Real-doc privacy stripping proof", "PII vault isolation proof", "No raw private document leakage",
        "Synthetic-to-real adapter dry run", "Redacted real-like fixture replay", "Operator review queue handoff",
        "Real-doc refusal path", "Billing/cost cap check", "Human authorization gate for any real live call",
        "No active MKB write gate", "No auto-accept gate", "Medication safety gate non-bypass proof",
    ):
        assert g in text


def test_next_stage_15z_present() -> None:
    assert "MEDAI-VERTEX-REAL-DOC-READINESS-GATES-NO-LIVE-15Z" in _text()


def test_evidence_chain_15pc_through_15xr4_present() -> None:
    text = _text()
    for b in ("15P-C", "15P-D", "15Q", "15R", "15S", "15T", "15U", "15V", "15W", "15X", "15X-R1", "15X-R2", "15X-R3", "15X-R4"):
        assert b in text


def test_troubleshooting_stop_conditions_present() -> None:
    text = _text()
    assert "stop and treat as a safety failure" in text.lower()
    for sc in ("real PII in an outbound payload", "non-verbatim evidence accepted", "undeclared label drift accepted", "medication safety bypass"):
        assert sc in text


def test_no_credentials_or_paths_in_doc() -> None:
    text = _text()
    for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/"):
        assert token not in text


def test_no_live_gate_names_required_in_doc_instructions() -> None:
    # The doc may reference the dedicated gate name as documentation, but must not
    # instruct setting the other live gates.
    text = _text()
    for gate in (
        "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_REMAINING_FAMILIES_LIVE_ALLOWED",
    ):
        assert gate not in text


def test_script_report_passes() -> None:
    import scripts.run_medai_vertex_semantic_calibration_operator_handoff_doc_15y as mod

    text = DOC.read_text(encoding="utf-8")
    checklist = mod._build_checklist(text)
    assert checklist["required_sections_present_count"] == len(REQUIRED_SECTIONS)
    assert checklist["calibration_pass_summary_present"] is True
    assert checklist["final_metrics_present"] is True
    assert checklist["category_coverage_present"] is True
    assert checklist["real_document_boundary_present"] is True
    assert checklist["future_real_doc_gates_present"] is True
    assert checklist["evidence_chain_present"] is True
    assert checklist["troubleshooting_stop_conditions_present"] is True
