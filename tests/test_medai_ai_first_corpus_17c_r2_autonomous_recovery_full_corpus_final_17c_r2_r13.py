"""No-live tests for 17C-R2-R13 autonomous recovery."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.autonomous_recovery_runner import REPORT_DIR
from execution.sectioned_cost_planner import build_cost_plan
from execution.sectioned_extraction import (
    SECTION_NAMES,
    build_section_payload,
    salvage_one_complete_json_object,
    split_tokenized_window,
    validate_section_response,
)
from execution.sectioned_merge import merge_sections

REPO_ROOT = Path(__file__).resolve().parents[1]


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_local_only_script_exits_and_writes_required_public_reports() -> None:
    result = subprocess.run(
        ["python", "scripts/run_medai_ai_first_corpus_17c_r2_autonomous_recovery_full_corpus_final_17c_r2_r13.py", "--local-only"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "MEDAI-AI-FIRST-CORPUS-17C-R2-AUTONOMOUS-RECOVERY-AND-FULL-CORPUS-RUN-FINAL-17C-R2-R13_PASS" in result.stdout
    for name in (
        "summary.json",
        "implementation_report.md",
        "autonomous_recovery_policy_public.md",
        "sectioned_strategy_public.json",
        "recovery_ladder_public.json",
        "sectioned_merge_validation_public.json",
        "checkpoint_policy_public.json",
        "cost_plan_public.json",
        "live_entry_gate_public.md",
        "live_run_public_report.md",
        "failed_docs_public.json",
        "safety_boundary_public.md",
    ):
        assert (REPORT_DIR / name).exists(), name


def test_local_only_makes_no_provider_or_live_side_effect_claim() -> None:
    s = _summary()
    assert s["provider_model_call_made_during_local_phase"] is False
    assert s["provider_model_call_made_during_live_phase"] is False
    assert s["gemini_call_made_during_live_phase"] is False
    assert s["full_live_run_started"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["live_gate_environment_active_after_run"] is False


def test_sectioned_extraction_contract_and_payload_shape() -> None:
    payload = build_section_payload("clinical_findings", "[NAME_1] has symptom token.")
    cfg = payload["generationConfig"]
    assert cfg["responseMimeType"] == "application/json"
    assert cfg["temperature"] == 0
    assert cfg["maxOutputTokens"] <= 2048
    text = payload["contents"][0]["parts"][0]["text"]
    assert "strict compact JSON object only" in text
    assert "Do not infer" in text
    assert "clinical_findings" in text


def test_full_schema_truncation_falls_back_contract_exists() -> None:
    ladder = json.loads((REPORT_DIR / "recovery_ladder_public.json").read_text(encoding="utf-8"))
    assert "compact_full_schema" in ladder["strategies"]
    assert "sectioned_extraction" in ladder["strategies"]
    assert "adaptive_section_splitting" in ladder["strategies"]
    assert ladder["continue_to_next_doc_for_non_safety_schema_failures"] is True


def test_section_truncation_window_strategy_and_no_partial_json_salvage() -> None:
    windows = split_tokenized_window("a" * 13000, max_chars=5000)
    assert len(windows) >= 3
    ok, obj, reason = salvage_one_complete_json_object('{"items":[1]')
    assert ok is False
    assert obj is None
    assert reason == "truncated_or_invalid_json"


def test_missing_schema_keys_trigger_skeleton_retry_without_schema_weakening() -> None:
    valid, reason, obj = validate_section_response('{"section":"clinical_findings","items":[]}', "clinical_findings")
    assert valid is False
    assert obj is None
    assert reason == "missing_section_schema_fields"
    payload = build_section_payload("clinical_findings", "x", skeleton_retry=True)
    assert '"warnings":[]' in payload["contents"][0]["parts"][0]["text"]


def test_merge_requires_all_required_sections_and_no_synthesis() -> None:
    ok, reason, merged = merge_sections({})
    assert ok is False
    assert reason == "missing_required_sections"
    sections = {
        name: {"section": name, "items": [], "needs_review": True, "warnings": []}
        for name in SECTION_NAMES
    }
    ok, reason, merged = merge_sections(sections)
    assert ok is True
    assert reason == "ok"
    assert merged is not None
    assert merged["needs_review"] is True
    assert merged["extracted_labs"] == []
    assert merged["extracted_diagnoses"] == []
    assert merged["extracted_medications"] == []


def test_cost_planner_enforces_caps() -> None:
    plan = build_cost_plan(
        [100, 200, 300],
        total_cap_usd=10.00,
        per_chunk_cap_usd=0.05,
        full_max_output_tokens=8192,
        section_max_output_tokens=2048,
        input_usd_per_m=0.075,
        output_usd_per_m=0.30,
    )
    assert plan.full_schema_safe is True
    assert plan.selected_chunk_size >= 1
    blocked = build_cost_plan(
        [5_000_000],
        total_cap_usd=0.01,
        per_chunk_cap_usd=0.0001,
        full_max_output_tokens=8192,
        section_max_output_tokens=2048,
        input_usd_per_m=0.075,
        output_usd_per_m=0.30,
    )
    assert blocked.full_schema_safe is False


def test_public_redaction_and_private_artifact_flags() -> None:
    s = _summary()
    assert s["private_checkpoint_committed"] is False
    assert s["private_evidence_committed"] is False
    assert s["private_responses_committed"] is False
    assert s["parsed_private_responses_committed"] is False
    assert s["tokenized_payloads_written_to_repo"] is False
    assert s["raw_ocr_written_to_repo"] is False
    assert s["token_maps_written_to_repo"] is False
    assert s["private_identifier_values_written_to_repo"] is False
    assert s["credential_or_token_written_to_repo"] is False
    for report in REPORT_DIR.iterdir():
        if report.suffix.lower() not in {".json", ".md"}:
            continue
        result = check_public_report_payload(report.read_text(encoding="utf-8", errors="ignore"))
        assert result.passed, (report.name, result.private_filename_path_leaks, result.secret_leaks)


def test_existing_live_batch_regression_surface_is_compatible() -> None:
    s = _summary()
    assert s["checkpoint_document_aware"] is True
    assert s["checkpoint_section_aware"] is True
    assert s["completed_docs_skipped_on_resume"] is True
    assert s["completed_sections_skipped_on_resume"] is True
    assert s["schema_weakened"] is False
    assert s["clinical_values_inferred"] is False
    assert s["missing_required_values_synthesized"] is False
    assert s["future_17d_mkb_import_not_started"] is True
