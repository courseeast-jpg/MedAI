"""Focused tests for MEDAI-AI-EXTRACTION-WORKFLOW-SEAM-15A."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from app.source_extraction_packages import package_action_plan
from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_extraction_adapter import (
    AIExtractionAdapterInput,
    ExtractionWorkflowContext,
    FakeAIExtractionAdapter,
)
from execution.extraction_workflow import run_ai_extraction_workflow, workflow_result_to_public_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_OCR_BODY = "Patient Jane Example DOB 01/02/1970 MRN 123456 private cytology body"


def _extract(source_class: str):
    return FakeAIExtractionAdapter().extract(
        AIExtractionAdapterInput(
            source_class=source_class,
            safe_source_document_id="source_fake_001",
            selected_document_category="AI-assisted extraction",
            selected_specialty_domain="urology",
            source_modality="fake_local_adapter",
        )
    )


def test_adapter_contract_exists_and_fake_adapter_returns_schema_valid_drafts() -> None:
    draft = _extract("urinalysis_table")

    assert draft.external_api_used is False
    assert draft.review_required is True
    assert draft.auto_accept is False
    assert draft.sections


def test_cytology_pathology_package_contains_required_sections() -> None:
    draft = _extract("cytology_pathology")
    headings = {section.heading for section in draft.sections}

    assert {
        "Tests Ordered",
        "Diagnosis",
        "Recommendation",
        "Clinical History / ICD",
        "Cytology Information",
        "Gross Description",
    }.issubset(headings)


def test_recommendation_section_is_source_text_only_not_medai_recommendation() -> None:
    draft = _extract("cytology_pathology")
    recommendation = next(section for section in draft.sections if section.heading == "Recommendation")

    assert "not MedAI recommendation" in recommendation.narrative_label


def test_urinalysis_package_reconstructs_rows_with_value_reference_flag_unit() -> None:
    draft = _extract("urinalysis_table")
    observations = draft.sections[0].observations
    rbc = next(obs for obs in observations if obs.label == "RBC")

    assert rbc.value == "3-10"
    assert rbc.flag == "abnormal"
    assert rbc.unit == "/hpf"
    assert rbc.reference_interval == "0-2"


def test_portal_card_package_reconstructs_card_style_observations() -> None:
    draft = _extract("portal_cards")
    observations = {obs.label: obs for obs in draft.sections[0].observations}

    assert observations["Specific Gravity"].value == "1.020"
    assert observations["Protein"].reference_interval == "Negative/Trace"
    assert observations["Ketones"].value == "Negative"


def test_workflow_result_is_review_bound_only_and_no_active_writes() -> None:
    result = run_ai_extraction_workflow(ExtractionWorkflowContext(source_class="urinalysis_table"))

    assert result.review_required is True
    assert result.auto_accept is False
    assert result.active_written_count == 0
    assert result.review_bound_package_count == 1
    assert result.packages[0]["package_status"] == "review-bound"


def test_external_api_false_and_privacy_gate_fields_exist() -> None:
    result = run_ai_extraction_workflow(ExtractionWorkflowContext(source_class="portal_cards"))

    assert result.external_api_used is False
    assert result.privacy_gate_status
    assert result.pii_redaction_required is False
    assert result.external_payload_allowed is False
    assert result.external_payload_preview_available is False


def test_public_reports_do_not_contain_raw_private_ocr_body() -> None:
    env = dict(os.environ)
    env["MEDAI_15A_SKIP_PYTEST"] = "1"
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_extraction_workflow_seam_15a.py"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    report_dir = REPO_ROOT / "reports" / "medai_ai_extraction_workflow_seam_15a"
    for path in report_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert PRIVATE_OCR_BODY not in text
        assert "Jane Example" not in text
        assert "MRN 123456" not in text
        assert "C:\\" not in text
        result = check_public_report_payload(json.loads(text) if path.suffix == ".json" else text)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_operator_preview_contains_visible_package_body() -> None:
    result = run_ai_extraction_workflow(ExtractionWorkflowContext(source_class="portal_cards"))
    preview = result.operator_preview

    assert preview["visible"] is True
    assert preview["packages"][0]["sections"][0]["observations"]
    assert preview["packages"][0]["label"] == "AI-assisted extraction draft - review-bound only"


def test_accept_reject_defer_semantics_are_not_broken() -> None:
    actions = package_action_plan(["record-1"])["actions"]
    keys = {action["key"] for action in actions}

    assert {
        "accept_package_after_source_comparison",
        "reject_package",
        "defer_package",
    }.issubset(keys)


def test_package_first_review_queue_semantics_are_preserved() -> None:
    import app.main as main

    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "Source extraction packages" in source
    assert "Atomic record actions" in source


def test_hidden_admin_default_operator_ui_constraints_are_not_regressed() -> None:
    import app.main as main

    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "Show advanced tools" in source
    assert "render_ai_extraction_operator_preview" in source


def test_workflow_public_dict_keeps_safety_contract() -> None:
    result = workflow_result_to_public_dict(
        run_ai_extraction_workflow(ExtractionWorkflowContext(source_class="cytology_pathology"))
    )

    assert result["external_api_used"] is False
    assert result["active_written_count"] == 0
    assert result["auto_accept"] is False
    assert result["review_required"] is True
