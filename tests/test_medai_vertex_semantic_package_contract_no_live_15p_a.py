"""Focused tests for MEDAI-VERTEX-SEMANTIC-PACKAGE-CONTRACT-NO-LIVE-15P-A."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_package_quality_eval import PACKAGE_FAMILIES
from execution.vertex_semantic_package_contract import (
    PROMPT_REQUIRED_PHRASES,
    build_fake_vertex_semantic_response,
    build_vertex_semantic_prompt,
    build_vertex_semantic_request_payload,
    evaluate_vertex_semantic_contract,
    prompt_privacy_check,
    validate_vertex_semantic_response,
)
from app.ai_package_run_review_preview import build_run_review_package_previews

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_package_contract_no_live_15p_a"


def test_contract_covers_all_15o_package_families() -> None:
    report = evaluate_vertex_semantic_contract()
    summary = report["summary"]
    assert summary["package_families_checked"] == list(PACKAGE_FAMILIES)
    assert summary["package_family_count"] == 4
    assert len(report["cases"]) == 4


def test_prompt_contains_required_no_live_contract_instructions() -> None:
    for preview in build_run_review_package_previews():
        prompt = build_vertex_semantic_prompt(preview)
        for phrase in PROMPT_REQUIRED_PHRASES.values():
            assert phrase in prompt
        assert preview.source_visible_body in prompt
        assert "Return JSON only." in prompt
        assert "diagnosis, treatment advice, clinical recommendations" in prompt
        assert prompt_privacy_check(prompt)["privacy_result"] == "passed"


def test_request_payload_shape_is_vertex_json_only_and_no_live() -> None:
    for preview in build_run_review_package_previews():
        payload = build_vertex_semantic_request_payload(preview)
        assert payload["provider_route"] == "vertex"
        assert payload["provider_name"] == "gemini_vertex"
        assert payload["model"] == "gemini-2.5-flash-lite"
        assert payload["location"] == "global"
        assert payload["endpoint_host"] == "aiplatform.googleapis.com"
        assert payload["live_call_made"] is False
        assert payload["generationConfig"]["temperature"] == 0
        assert payload["generationConfig"]["responseMimeType"] == "application/json"
        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][0]["parts"][0]["text"]


def test_fake_vertex_response_schema_validation_requires_source_evidence() -> None:
    for preview in build_run_review_package_previews():
        response = build_fake_vertex_semantic_response(preview)
        valid, errors = validate_vertex_semantic_response(response, preview)
        assert valid, errors
        assert response["review_required"] is True
        assert response["auto_accept"] is False
        assert response["semantic_findings"]
        for finding in response["semantic_findings"]:
            assert finding["evidence_text"]
            assert finding["source_faithful"] is True


def test_schema_validation_rejects_hallucinated_or_unanchored_fields() -> None:
    preview = build_run_review_package_previews()[0]
    response = build_fake_vertex_semantic_response(preview)
    response["semantic_findings"][0]["label"] = "Invented Field"
    valid, errors = validate_vertex_semantic_response(response, preview)
    assert valid is False
    assert "finding_0_not_source_anchored" in errors


def test_contract_preserves_15o_review_package_guarantees() -> None:
    summary = evaluate_vertex_semantic_contract()["summary"]
    assert summary["schema_validation_pass_count"] == 4
    assert summary["fake_vertex_response_valid_count"] == 4
    assert summary["source_visible_body_preserved_count"] == 4
    assert summary["evidence_anchor_preserved_count"] == 4
    assert summary["candidate_facts_separated_count"] == 4
    assert summary["unknown_values_explicit_count"] == 4
    assert summary["uncertainty_flags_visible_count"] == 4
    assert summary["under_1_minute_compare_preserved_count"] == 4
    assert summary["hallucinated_field_count"] == 0
    assert summary["all_contract_invariants_passed"] is True


def test_safety_invariants_and_old_gates_preserved() -> None:
    summary = evaluate_vertex_semantic_contract()["summary"]
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["active_written_count"] == 0
    assert summary["auto_accept"] is False
    assert summary["review_required"] is True
    source = (REPO_ROOT / "execution" / "vertex_semantic_package_contract.py").read_text(encoding="utf-8")
    assert "MEDAI" + "_VERTEX_LIVE_SMOKE_ALLOWED" not in source
    assert "run_gemini_vertex_live_smoke" not in source
    assert "GEMINI" + "_API_KEY" not in source


def test_contract_public_payloads_are_report_safe() -> None:
    report = evaluate_vertex_semantic_contract()
    check = check_public_report_payload(report)
    assert check.passed, check.leak_examples_redacted
    serialized = json.dumps(report, sort_keys=True)
    for token in ("Authorization", "Bearer ", "ya29.", "AIza", "token_map", "C:\\", "DOB", "MRN:"):
        assert token not in serialized


def test_reports_generated_and_public_safe() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_vertex_semantic_package_contract_no_live_15p_a.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "medai_vertex_semantic_package_contract_no_live_15p_a_ready" in proc.stdout
    expected = {
        "summary.json",
        "implementation_report.md",
        "semantic_contract_cases.json",
        "semantic_contract_matrix.md",
    }
    assert expected == {path.name for path in REPORT_DIR.iterdir()}
    for path in REPORT_DIR.iterdir():
        text = path.read_text(encoding="utf-8")
        for token in ("Authorization", "Bearer ", "ya29.", "AIza", "token_map", "C:\\", "DOB", "MRN:"):
            assert token not in text
        payload = json.loads(text) if path.suffix == ".json" else text
        check = check_public_report_payload(payload)
        assert check.passed, f"{path.name}: {check.leak_examples_redacted}"


def test_report_summary_contains_required_metrics() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_vertex_semantic_package_contract_no_live_15p_a.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["privacy_result"] == "passed"
    assert summary["package_family_count"] == 4
    assert summary["schema_validation_pass_count"] == 4
    assert summary["fake_vertex_response_valid_count"] == 4
    assert summary["source_visible_body_preserved_count"] == 4
    assert summary["evidence_anchor_preserved_count"] == 4
    assert summary["candidate_facts_separated_count"] == 4
    assert summary["unknown_values_explicit_count"] == 4
    assert summary["uncertainty_flags_visible_count"] == 4
    assert summary["hallucinated_field_count"] == 0
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["active_written_count"] == 0
    assert summary["auto_accept"] is False
    assert summary["review_required"] is True
    assert summary["billing_check_pending"] is True
