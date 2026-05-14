from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from clinical_knowledge.terminology.b07_term_opt_in import (
    build_b07_terminology_metadata,
    read_b07_term_flag_state,
)
from clinical_knowledge.terminology.term05_read_only_adapter import build_synthetic_read_only_adapter

ROOT = Path(__file__).resolve().parents[1]
ENABLED_ENV = {
    "MEDAI_B07_TERMINOLOGY_OPT_IN": "1",
    "MEDAI_TERMINOLOGY_LOOKUP_ENABLED": "1",
    "MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION": "1",
    "MEDAI_TERMINOLOGY_READ_ONLY": "1",
    "MEDAI_TERMINOLOGY_ALLOW_WRITES": "0",
}


def test_flags_default_off_and_fail_closed() -> None:
    flags = read_b07_term_flag_state({})
    assert flags.enabled is False
    assert flags.b07_opt_in is False
    assert flags.lookup_enabled is False
    assert flags.hypothesis_annotation_enabled is False
    assert flags.read_only is True
    assert flags.allow_writes is False
    metadata = build_b07_terminology_metadata("aspirin", adapter=build_synthetic_read_only_adapter(), env={})
    assert metadata.enabled is False
    assert metadata.input_term is None
    assert metadata.terminology_status == "disabled"


def test_missing_or_inconsistent_flags_fail_closed() -> None:
    adapter = build_synthetic_read_only_adapter()
    missing_lookup = build_b07_terminology_metadata(
        "aspirin",
        adapter=adapter,
        env={"MEDAI_B07_TERMINOLOGY_OPT_IN": "1", "MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION": "1"},
    )
    assert missing_lookup.enabled is False
    assert "terminology_lookup_disabled" in missing_lookup.reason_codes
    writes_enabled = build_b07_terminology_metadata(
        "aspirin",
        adapter=adapter,
        env={**ENABLED_ENV, "MEDAI_TERMINOLOGY_ALLOW_WRITES": "1"},
    )
    assert writes_enabled.enabled is False
    assert "terminology_writes_forbidden" in writes_enabled.reason_codes


def test_exact_opt_in_metadata_is_hypothesis_only() -> None:
    metadata = build_b07_terminology_metadata(
        "aspirin",
        adapter=build_synthetic_read_only_adapter(),
        source_filter=["rxnorm"],
        env=ENABLED_ENV,
    )
    assert metadata.enabled is True
    assert metadata.terminology_status == "exact"
    assert metadata.source_system == "rxnorm"
    assert metadata.candidate_code_count == 1
    assert metadata.annotation_tier == "hypothesis"
    assert metadata.requires_review is True
    assert metadata.read_only_lookup is True
    assert metadata.writes_active_fact is False
    assert metadata.b07_authority_source is False


def test_unknown_and_ambiguous_behaviors_are_safe() -> None:
    adapter = build_synthetic_read_only_adapter()
    unknown = build_b07_terminology_metadata("b07 unknown remains unmapped", adapter=adapter, env=ENABLED_ENV)
    assert unknown.enabled is True
    assert unknown.terminology_status == "unmapped"
    assert unknown.candidate_code_count == 0
    assert unknown.no_code_hallucinated is True
    ambiguous = build_b07_terminology_metadata("aspirin", adapter=adapter, env=ENABLED_ENV)
    assert ambiguous.enabled is True
    assert ambiguous.terminology_status == "ambiguous"
    assert ambiguous.source_system is None
    assert ambiguous.requires_review is True
    assert "manual_review_required" in ambiguous.reason_codes


def test_rollback_restores_off_state() -> None:
    adapter = build_synthetic_read_only_adapter()
    off_before = build_b07_terminology_metadata("aspirin", adapter=adapter, env={})
    enabled = build_b07_terminology_metadata("aspirin", adapter=adapter, env=ENABLED_ENV)
    off_after = build_b07_terminology_metadata("aspirin", adapter=adapter, env={})
    assert enabled.enabled is True
    assert off_after.safe_public_summary() == off_before.safe_public_summary()


def test_no_clinical_advice_promotion_or_ddi_clear() -> None:
    metadata = build_b07_terminology_metadata("aspirin", adapter=build_synthetic_read_only_adapter(), env=ENABLED_ENV)
    assert metadata.writes_active_fact is False
    assert metadata.promotes_hypothesis is False
    assert metadata.clears_ddi_status is False
    assert metadata.clinical_advice_generated is False
    assert metadata.dosing_advice_generated is False
    assert metadata.prescribing_advice_generated is False
    assert metadata.external_api_used is False


def test_public_report_privacy_clean_for_metadata() -> None:
    metadata = build_b07_terminology_metadata(
        "aspirin",
        adapter=build_synthetic_read_only_adapter(),
        source_filter=["rxnorm"],
        env=ENABLED_ENV,
    )
    payload = metadata.safe_public_summary()
    assert check_public_report_payload(payload).passed is True


def test_validation_script_runs_and_report_safe() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_b07_term01_opt_in_integration_validation.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((ROOT / "reports" / "b07_term01_opt_in_integration" / "b07_term01_opt_in_integration_report.json").read_text(encoding="utf-8"))
    assert payload["conclusion"] == "b07_term01_opt_in_integration_ready"
    assert payload["cases_failed"] == 0
    assert payload["off_state_preservation_passed"] is True
    assert payload["opt_in_behavior_passed"] is True
    assert payload["rollback_behavior_passed"] is True
    assert payload["external_api_used"] is False
    assert payload["public_report_privacy_clean"] is True
    rendered = json.dumps(payload)
    assert "terminology_data/" not in rendered.replace("\\", "/")
    assert "data/terminology" not in rendered.replace("\\", "/")
    assert "LICENSE_ACK_PRIVATE" not in rendered
    assert "RXNCONSO" not in rendered
    assert "Loinc.csv" not in rendered


def test_b07_baseline_tests_pass_with_flags_off() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_cka_block07_medical_coding.py", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **dict(__import__("os").environ),
            "MEDAI_B07_TERMINOLOGY_OPT_IN": "0",
            "MEDAI_TERMINOLOGY_LOOKUP_ENABLED": "0",
            "MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION": "0",
            "MEDAI_TERMINOLOGY_READ_ONLY": "1",
            "MEDAI_TERMINOLOGY_ALLOW_WRITES": "0",
        },
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_no_private_or_runtime_files_staged() -> None:
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert staged.returncode == 0
    lowered = staged.stdout.lower().replace("\\", "/")
    assert "terminology_data" not in lowered
    assert "data/terminology" not in lowered
    assert "license_ack_private" not in lowered
    assert ".rrf" not in lowered
    assert ".csv" not in lowered
    assert ".sqlite" not in lowered
    assert ".db" not in lowered
