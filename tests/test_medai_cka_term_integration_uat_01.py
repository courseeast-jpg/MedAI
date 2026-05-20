"""Tests for MEDAI-CKA-TERM-INTEGRATION-UAT-01."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy.report_privacy import check_public_report_payload  # noqa: E402
from clinical_knowledge.terminology.term05_read_only_adapter import (  # noqa: E402
    build_synthetic_read_only_adapter,
)
from clinical_knowledge.terminology.term_match_hypothesis import (  # noqa: E402
    TERMINOLOGY_LOOKUP_ENV_VAR,
    derive_terminology_match_hypothesis,
)
from scripts.run_medai_cka_term_integration_uat_01 import run_uat  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "medai_cka_term_integration_uat_01"
SCOPED_FILES = {
    "scripts/run_medai_cka_term_integration_uat_01.py",
    "tests/test_medai_cka_term_integration_uat_01.py",
    "reports/medai_cka_term_integration_uat_01/MEDAI_CKA_TERM_INTEGRATION_UAT_01.md",
    "reports/medai_cka_term_integration_uat_01/medai_cka_term_integration_uat_01_report.json",
    "reports/medai_cka_term_integration_uat_01/medai_cka_term_integration_uat_01_report.md",
}


def _adapter():
    return build_synthetic_read_only_adapter()


def _enabled_env():
    return {TERMINOLOGY_LOOKUP_ENV_VAR: "true"}


def test_uat_modifies_no_runtime_files() -> None:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = {line[3:].replace("\\", "/") for line in result.stdout.splitlines() if line.strip()}
    assert changed <= SCOPED_FILES
    assert "app/main.py" not in changed


def test_default_off_helper_returns_none() -> None:
    record = {"terminology_candidate_text": "aspirin"}
    assert derive_terminology_match_hypothesis(record, env={}, lookup_adapter=_adapter()) is None


def test_env_on_exact_synthetic_match_controlled_vocab_only() -> None:
    metadata = derive_terminology_match_hypothesis(
        {"terminology_candidate_text": "aspirin", "terminology_system_filter": ["rxnorm"]},
        env=_enabled_env(),
        lookup_adapter=_adapter(),
    )
    assert metadata is not None
    assert metadata["match_family"] == "exact_terminology_match"
    assert metadata["terminology_system_family"] == "rxnorm"
    assert metadata["matches_count"] == 1
    assert metadata["review_required"] is True
    assert metadata["auto_accept_allowed"] is False


def test_env_on_ambiguous_synthetic_match_controlled_vocab_only() -> None:
    metadata = derive_terminology_match_hypothesis(
        {"terminology_candidate_text": "aspirin"},
        env=_enabled_env(),
        lookup_adapter=_adapter(),
    )
    assert metadata is not None
    assert metadata["match_family"] == "ambiguous_terminology_match"
    assert metadata["review_required"] is True
    assert metadata["auto_accept_allowed"] is False


def test_no_adapter_fails_closed() -> None:
    metadata = derive_terminology_match_hypothesis(
        {"terminology_candidate_text": "aspirin"},
        env=_enabled_env(),
        lookup_adapter=None,
    )
    assert metadata is None


def test_raw_private_input_fields_are_not_lookup_signature() -> None:
    record = {
        "raw_text": "[redacted synthetic raw marker]",
        "ocr_text": "[redacted synthetic ocr marker]",
        "filename": "[redacted synthetic filename marker]",
        "private_path": "[redacted synthetic path marker]",
    }
    assert derive_terminology_match_hypothesis(record, env=_enabled_env(), lookup_adapter=_adapter()) is None


def test_no_licensed_row_content_or_raw_private_output() -> None:
    metadata = derive_terminology_match_hypothesis(
        {"terminology_candidate_text": "aspirin"},
        env=_enabled_env(),
        lookup_adapter=_adapter(),
    )
    assert metadata is not None
    false_flags = (
        "licensed_row_content_included",
        "raw_text_emitted",
        "raw_ocr_text_emitted",
        "raw_document_text_emitted",
        "raw_filename_emitted",
        "private_path_emitted",
        "phi_emitted",
        "secret_emitted",
    )
    for flag in false_flags:
        assert metadata[flag] is False
    forbidden_keys = {
        "code",
        "display",
        "display_normalized",
        "rxcui",
        "loinc_num",
        "synonym",
        "definition",
    }
    assert not (set(metadata) & forbidden_keys)


def test_review_bound_no_auto_accept_and_no_inference() -> None:
    metadata = derive_terminology_match_hypothesis(
        {"terminology_candidate_text": "aspirin"},
        env=_enabled_env(),
        lookup_adapter=_adapter(),
    )
    assert metadata is not None
    assert metadata["review_required"] is True
    assert metadata["auto_accept_allowed"] is False
    assert metadata["clinical_interpretation_performed"] is False
    assert metadata["diagnosis_inference_performed"] is False
    assert metadata["treatment_inference_performed"] is False
    assert metadata["medication_inference_performed"] is False
    assert metadata["ddi_behavior_changed"] is False
    assert metadata["external_api_used"] is False


def test_no_streamlit_ui_wiring_and_app_main_unmodified() -> None:
    helper = (REPO_ROOT / "clinical_knowledge" / "terminology" / "term_match_hypothesis.py").read_text(encoding="utf-8")
    app_main = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    forbidden_ui_tokens = ("import streamlit", "st.", "st_button", "st_form")
    assert not any(token in helper.lower() for token in forbidden_ui_tokens)
    assert "derive_terminology_match_hypothesis" not in app_main
    assert "term_match_hypothesis" not in app_main


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
    report = run_uat()
    assert report["freeze_tags_untouched"] is True
    assert report["cue_expansion_recommended"] is False


def test_os_environ_not_polluted_by_env_mapping() -> None:
    before = os.environ.get(TERMINOLOGY_LOOKUP_ENV_VAR)
    run_uat()
    after = os.environ.get(TERMINOLOGY_LOOKUP_ENV_VAR)
    assert after == before


def test_uat_report_payload_is_safe_and_aggregate_only() -> None:
    report = run_uat()
    assert report["total_synthetic_cases_evaluated"] == 6
    assert report["metadata_emission_count"] == 3
    assert report["unsafe_output_count"] == 0
    assert report["licensed_row_content_output_count"] == 0
    assert report["raw_text_output_count"] == 0
    assert report["raw_ocr_text_output_count"] == 0
    assert report["raw_document_text_output_count"] == 0
    assert report["raw_filename_output_count"] == 0
    assert report["private_path_output_count"] == 0
    assert report["phi_output_count"] == 0
    assert report["secret_output_count"] == 0
    assert report["external_api_used_count"] == 0
    assert report["auto_accept_allowed_count"] == 0
    assert check_public_report_payload(report).passed is True


def test_generated_reports_contain_no_raw_private_fields_if_present() -> None:
    if not REPORT_DIR.exists():
        pytest.skip("UAT reports not generated yet")
    for path in REPORT_DIR.iterdir():
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = {"report_text": path.read_text(encoding="utf-8")}
        assert check_public_report_payload(payload).passed is True
