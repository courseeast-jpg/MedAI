"""Tests for MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy.report_privacy import check_public_report_payload  # noqa: E402
from clinical_knowledge.terminology.term_match_hypothesis import (  # noqa: E402
    TERMINOLOGY_LOOKUP_ENV_VAR,
)
from scripts.run_medai_cka_term_integration_wiring_next_01 import (  # noqa: E402
    UI_ENV_VAR,
    _load_plan_functions,
    _safe_metadata,
    run_wiring_audit,
)

SCOPED_FILES = {
    "app/main.py",
    "scripts/run_medai_cka_term_integration_wiring_next_01.py",
    "tests/test_medai_cka_term_integration_wiring_next_01.py",
    "reports/medai_cka_term_integration_wiring_next_01/MEDAI_CKA_TERM_INTEGRATION_WIRING_NEXT_01.md",
    "reports/medai_cka_term_integration_wiring_next_01/medai_cka_term_integration_wiring_next_01_report.json",
    "reports/medai_cka_term_integration_wiring_next_01/medai_cka_term_integration_wiring_next_01_report.md",
}


def _plan_fn():
    return _load_plan_functions()["terminology_match_hypothesis_ui_plan"]


def _item():
    return {"terminology_match_hypothesis_metadata": _safe_metadata()}


def test_only_scoped_files_changed() -> None:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = {line[3:].replace("\\", "/") for line in result.stdout.splitlines() if line.strip()}
    assert changed <= SCOPED_FILES


def test_both_env_vars_unset_no_render_block() -> None:
    assert _plan_fn()(_item(), env={}) is None


def test_only_helper_env_truthy_no_render_block() -> None:
    assert _plan_fn()(_item(), env={TERMINOLOGY_LOOKUP_ENV_VAR: "true"}) is None


def test_only_ui_env_truthy_no_render_block() -> None:
    assert _plan_fn()(_item(), env={UI_ENV_VAR: "true"}) is None


def test_both_env_vars_truthy_safe_metadata_renders_read_only_plan() -> None:
    plan = _plan_fn()(_item(), env={TERMINOLOGY_LOOKUP_ENV_VAR: "true", UI_ENV_VAR: "true"})
    assert plan is not None
    assert plan["expander_label"] == "Terminology match hypothesis"
    assert "exact_terminology_match" in "\n".join(plan["markdown_lines"])
    assert "rxnorm" in "\n".join(plan["markdown_lines"])
    assert "Auto-accept allowed" in "\n".join(plan["markdown_lines"])


def test_no_unsafe_metadata_or_auto_accept_semantics_render() -> None:
    bad = {"terminology_match_hypothesis_metadata": {**_safe_metadata(), "auto_accept_allowed": True}}
    assert _plan_fn()(bad, env={TERMINOLOGY_LOOKUP_ENV_VAR: "true", UI_ENV_VAR: "true"}) is None


def test_only_markdown_and_caption_used_by_wiring_block() -> None:
    report = run_wiring_audit()
    assert report["allowed_streamlit_calls"] == ["st.markdown", "st.caption"]
    assert report["forbidden_streamlit_call_count"] == 0
    assert report["buttons_added"] is False
    assert report["forms_added"] is False
    assert report["callbacks_added"] is False
    assert report["actions_added"] is False
    assert report["state_mutation_added"] is False
    assert report["data_layer_write_added"] is False
    assert report["document_type_mutation_added"] is False


def test_no_licensed_or_raw_private_content_rendered() -> None:
    plan = _plan_fn()(_item(), env={TERMINOLOGY_LOOKUP_ENV_VAR: "true", UI_ENV_VAR: "true"})
    assert plan is not None
    serialized = json.dumps(plan, sort_keys=True).lower()
    forbidden = (
        "display_normalized",
        "synonym",
        "definition",
        "raw_text",
        "raw_ocr",
        "raw_document",
        "private_path",
        "secret",
        "patient",
    )
    assert not any(token in serialized for token in forbidden)
    assert check_public_report_payload(plan).passed is True


def test_no_clinical_interpretation_ddi_or_cue_expansion() -> None:
    report = run_wiring_audit()
    assert report["clinical_interpretation_performed"] is False
    assert report["diagnosis_inference_performed"] is False
    assert report["treatment_inference_performed"] is False
    assert report["medication_inference_performed"] is False
    assert report["ddi_behavior_changed"] is False
    assert report["cue_expansion_recommended"] is False
    assert report["cue_expansion_performed"] is False
    assert report["auto_accept_allowed"] is False


def test_app_main_has_one_narrow_cka_terminology_wiring_block() -> None:
    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert source.count("MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01") == 2
    assert "with st.expander(\"Advanced technical details\", expanded=False)" in source
    assert "terminology_match_hypothesis_ui_plan(item)" in source


def test_helper_remains_default_off_outside_explicit_env() -> None:
    from clinical_knowledge.terminology.term_match_hypothesis import derive_terminology_match_hypothesis

    assert derive_terminology_match_hypothesis({"terminology_candidate_text": "aspirin"}, env={}) is None


def test_freeze_and_park_tags_not_touched() -> None:
    for commit in ("7ef8ffd", "3e46461", "9f9e22d", "f4d3cc6", "748c32a"):
        result = subprocess.run(
            ["git", "tag", "--points-at", commit],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip(), f"missing expected tag at {commit}"


def test_generated_reports_privacy_clean_if_present() -> None:
    report_dir = REPO_ROOT / "reports" / "medai_cka_term_integration_wiring_next_01"
    if not report_dir.exists():
        pytest.skip("report script has not generated reports yet")
    for path in report_dir.iterdir():
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else {"report_text": path.read_text(encoding="utf-8")}
        assert check_public_report_payload(payload).passed is True
