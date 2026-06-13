"""Docs-only tests for the 15W Vertex Decision Audit operator handoff guide."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_DOC = REPO_ROOT / "docs" / "operator_handoff" / "MEDAI_VERTEX_DECISION_AUDIT_OPERATOR_HANDOFF_15W.md"

REQUIRED_SECTIONS = [
    "## 1. Purpose",
    "## 2. How to reach the panel",
    "## 3. What the operator should see",
    "## 4. Export instructions",
    "## 5. Governance guarantees",
    "## 6. Operator do / don't",
    "## 7. Troubleshooting",
    "## 8. Evidence chain",
    "## 9. Current limits",
    "## 10. Next stage preview",
]


def _text() -> str:
    return HANDOFF_DOC.read_text(encoding="utf-8")


def test_handoff_doc_exists() -> None:
    assert HANDOFF_DOC.exists()


def test_all_required_sections_present() -> None:
    text = _text()
    for section in REQUIRED_SECTIONS:
        assert section in text, section


def test_navigation_and_title_present() -> None:
    text = _text()
    assert '"Vertex Decision Audit"' in text
    assert "Vertex Semantic Review Decision Audit" in text


def test_decision_counts_present() -> None:
    text = _text()
    for tok in ("total **15**", "accepted_for_review **13**", "rejected **1**", "deferred **1**"):
        assert tok in text


def test_all_four_package_family_counts_present() -> None:
    text = _text()
    for tok in (
        "`portal_result_cards`: 4",
        "`cytology_pathology_narrative`: 3",
        "`urinalysis_table_like_lab`: 4",
        "`mixed_narrative_numeric_result`: 4",
    ):
        assert tok in text


def test_provider_route_model_present() -> None:
    assert "`vertex` / `gemini-2.5-flash-lite`" in _text()


def test_governance_guarantees_present() -> None:
    text = _text()
    for guarantee in (
        "No live provider call",
        "No active MKB write",
        "No auto-accept",
        "No provider/network call",
        "Decision store remains unchanged",
    ):
        assert guarantee in text


def test_export_instructions_present() -> None:
    text = _text()
    assert "JSON export" in text
    assert "CSV export" in text
    assert "Markdown audit summary" in text
    assert "Exports are read-only. No active MKB records are created." in text


def test_real_document_limitation_present() -> None:
    text = _text()
    assert "Not yet cleared for unrestricted real medical documents" in text
    assert "No unrestricted real medical document route yet" in text


def test_troubleshooting_safety_failures_present() -> None:
    text = _text()
    assert "stop and report" in text
    assert "Tab missing" in text
    assert "Export missing" in text
    assert "Decision counts mismatch" in text


def test_evidence_chain_15pc_through_15v_present() -> None:
    text = _text()
    for block in ("15P-C", "15P-D", "15Q", "15R", "15S", "15T", "15U", "15V"):
        assert block in text


def test_no_credentials_or_paths_in_doc() -> None:
    text = _text()
    for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/"):
        assert token not in text


def test_no_live_gate_names_required_in_doc() -> None:
    # The doc should not instruct setting any live provider gate.
    text = _text()
    for gate in (
        "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_REMAINING_FAMILIES_LIVE_ALLOWED",
    ):
        assert gate not in text
