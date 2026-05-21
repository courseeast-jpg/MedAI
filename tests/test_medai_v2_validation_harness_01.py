"""Focused tests for MEDAI-V2-VALIDATION-HARNESS-01.

These tests prove that:

* all three reports exist and pass the public-report privacy check;
* JSON required booleans exist and have the correct values;
* prior V2 report folders exist (architecture / foundation / runtime
  contracts);
* the runtime contracts module imports silently;
* the validation_harness module imports silently;
* both modules use only standard-library imports;
* no forbidden import names appear in either module's source;
* the 32 required contract names resolve from the runtime contracts
  module;
* the 10 required ``Protocol`` classes are typing.Protocol-shaped;
* ``V2RuntimeSafetyProfile`` defaults match the foundation
  invariants;
* ``V2TerminologyMatchSummary`` is aggregate-only and forbids row-
  content fields;
* ``V2ReviewItem`` defaults are review-bound;
* the V1 five-validation health-check catalog is enumerated;
* the V2 validation matrix exposes all 8 categories;
* no runtime files are modified by this block.
"""
from __future__ import annotations

import dataclasses
import io
import json
import sys
from pathlib import Path
from typing import Protocol

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_validation_harness_01"
HARNESS_MODULE_PATH = (
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py"
)
CONTRACTS_MODULE_PATH = (
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py"
)

SHORT_SUMMARY = REPORT_DIR / "MEDAI_V2_VALIDATION_HARNESS_01.md"
JSON_REPORT = REPORT_DIR / "medai_v2_validation_harness_01_report.json"
LONG_REPORT = REPORT_DIR / "medai_v2_validation_harness_01_report.md"

REQUIRED_REPORT_FILES = (SHORT_SUMMARY, JSON_REPORT, LONG_REPORT)

REQUIRED_PRIOR_V2_REPORT_DIRS = (
    REPO_ROOT / "reports" / "medai_v2_architecture_spec_01",
    REPO_ROOT / "reports" / "medai_v2_foundation_spec_02",
    REPO_ROOT / "reports" / "medai_v2_runtime_contracts_01",
)

ALLOWED_STDLIB_TOP_LEVEL_MODULES = {
    "dataclasses",
    "datetime",
    "enum",
    "typing",
    "__future__",
}

FORBIDDEN_IMPORT_PREFIXES = (
    "streamlit",
    "requests",
    "httpx",
    "http.",
    "urllib",
    "socket",
    "sqlite3",
    "sqlcipher3",
    "anthropic",
    "google.generativeai",
    "openai",
    "boto3",
    "psycopg",
)

REQUIRED_PROTOCOLS = (
    "V2IngestionAdapterProtocol",
    "V2DocumentQualityProtocol",
    "V2ExtractionAdapterProtocol",
    "V2ClassifierProtocol",
    "V2TerminologyLookupProtocol",
    "V2ReviewQueueProtocol",
    "V2OperatorActionProtocol",
    "V2ObservabilitySinkProtocol",
    "V2ValidationHarnessProtocol",
    "V2PrivacyGateProtocol",
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(JSON_REPORT.read_text(encoding="utf-8"))


# ── Reports + privacy ─────────────────────────────────────────────────────


def test_all_three_reports_exist():
    for p in REQUIRED_REPORT_FILES:
        assert p.is_file(), f"missing report file: {p.relative_to(REPO_ROOT)}"


def test_prior_v2_report_folders_exist():
    for d in REQUIRED_PRIOR_V2_REPORT_DIRS:
        assert d.is_dir(), f"missing prior V2 report dir: {d.relative_to(REPO_ROOT)}"


def test_reports_pass_public_report_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for p in REQUIRED_REPORT_FILES:
        content = p.read_text(encoding="utf-8")
        target: object = content
        if p.suffix == ".json":
            target = json.loads(content)
        result = check_public_report_payload(target)
        assert result.passed, (
            f"privacy check failed for {p.name}: "
            f"phi={result.raw_phi_logged_in_public_reports} "
            f"private_ref={result.private_filename_path_leaks} "
            f"secrets={result.secret_leaks}"
        )


# ── JSON required booleans ────────────────────────────────────────────────


def test_json_required_false_booleans(payload):
    for key in (
        "runtime_behavior_changed",
        "app_main_changed",
        "ui_changed",
        "launcher_changed",
        "startup_config_changed",
        "extraction_changed",
        "ocr_changed",
        "classifier_changed",
        "threshold_scoring_changed",
        "cue_pack_changed",
        "clinical_behavior_changed",
        "ddi_behavior_changed",
        "terminology_behavior_changed",
        "private_adapter_implemented",
        "concrete_adapters_implemented",
        "runtime_wiring_added",
        "external_api_used",
        "private_data_accessed",
        "licensed_rows_read",
        "private_license_ack_read",
        "private_config_read",
        "runtime_db_accessed",
        "source_documents_opened",
        "raw_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "secrets_printed",
        "tags_touched",
        "cue_expansion_recommended",
    ):
        assert payload.get(key) is False, f"{key} expected False, got {payload.get(key)!r}"


def test_json_required_true_booleans(payload):
    for key in (
        "reports_only",
        "requires_preceding_spec",
        "preceding_architecture_spec_present",
        "preceding_foundation_spec_present",
        "preceding_runtime_contracts_present",
        "v1_validation_healthcheck_catalog_created",
        "v2_validation_matrix_created",
        "contract_conformance_harness_created",
        "contracts_importable",
        "contracts_side_effect_free",
        "contracts_standard_library_only",
        "validation_harness_side_effect_free",
        "v1_release_preserved",
    ):
        assert payload.get(key) is True, f"{key} expected True, got {payload.get(key)!r}"


def test_next_recommended_block_value(payload):
    assert payload.get("next_recommended_block") == "V2-UI-SHELL-SPEC-01_OR_V2-DATA-INFRA-SPEC-01"


# ── Module imports silent + stdlib-only ───────────────────────────────────


def _check_module_silent_import(module_dotted: str) -> tuple[str, str]:
    """Return (stdout, stderr) captured during a fresh import."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        sys.modules.pop(module_dotted, None)
        __import__(module_dotted)
        out = sys.stdout.getvalue()
        err = sys.stderr.getvalue()
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return out, err


def test_validation_harness_module_imports_silently():
    out, err = _check_module_silent_import(
        "clinical_knowledge.v2_contracts.validation_harness"
    )
    assert out == "", f"validation_harness import wrote stdout: {out!r}"
    assert err == "", f"validation_harness import wrote stderr: {err!r}"


def test_runtime_contracts_module_imports_silently():
    out, err = _check_module_silent_import(
        "clinical_knowledge.v2_contracts.runtime_contracts"
    )
    assert out == "", f"runtime_contracts import wrote stdout: {out!r}"
    assert err == "", f"runtime_contracts import wrote stderr: {err!r}"


def _static_scan(path: Path) -> list[tuple[int, str]]:
    """Return a list of (lineno, module_name) for non-stdlib or forbidden imports."""
    violations: list[tuple[int, str]] = []
    src = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        if stripped.startswith("from "):
            module_name = stripped.split(None, 2)[1]
        else:
            module_name = stripped.split(None, 2)[1].split(",")[0]
        top_level = module_name.split(".", 1)[0]
        if top_level not in ALLOWED_STDLIB_TOP_LEVEL_MODULES:
            violations.append((lineno, module_name))
        for prefix in FORBIDDEN_IMPORT_PREFIXES:
            if module_name == prefix or module_name.startswith(prefix + "."):
                violations.append((lineno, f"{module_name} (forbidden prefix {prefix})"))
    return violations


def test_validation_harness_module_standard_library_only():
    violations = _static_scan(HARNESS_MODULE_PATH)
    assert violations == [], f"non-stdlib / forbidden imports: {violations}"


def test_runtime_contracts_module_standard_library_only():
    violations = _static_scan(CONTRACTS_MODULE_PATH)
    assert violations == [], f"non-stdlib / forbidden imports: {violations}"


def test_no_forbidden_imports_in_harness_module():
    src = HARNESS_MODULE_PATH.read_text(encoding="utf-8")
    for fragment in (
        "import streamlit",
        "from streamlit",
        "import requests",
        "from requests",
        "import sqlite3",
        "from sqlite3",
        "from app.",
        "import app.",
        "from clinical_knowledge.terminology",
        "import clinical_knowledge.terminology",
    ):
        assert fragment not in src, f"found {fragment!r} in validation_harness module"


# ── Contract inventory + protocols + safety profile ───────────────────────


def test_all_v2_contract_names_resolve_from_runtime_contracts():
    import clinical_knowledge.v2_contracts.runtime_contracts as rc

    assert len(rc.CONTRACT_NAMES) == 32
    for name in rc.CONTRACT_NAMES:
        assert hasattr(rc, name), f"missing contract name: {name}"


def test_required_protocols_are_protocols():
    import clinical_knowledge.v2_contracts.runtime_contracts as rc

    for name in REQUIRED_PROTOCOLS:
        obj = getattr(rc, name)
        assert issubclass(obj, Protocol), f"{name} is not a Protocol"


def test_default_runtime_safety_profile_invariants():
    import clinical_knowledge.v2_contracts.runtime_contracts as rc

    p = rc.V2RuntimeSafetyProfile()
    assert p.local_only is True
    assert p.review_bound is True
    assert p.external_api_blocked is True
    assert p.auto_accept_allowed is False
    assert p.terminology_lookup_aggregate_only is True
    assert p.private_adapter_implemented is False
    assert p.cue_expansion_recommended is False
    assert p.clinical_decision_expansion is False


def test_terminology_summary_is_aggregate_only():
    import clinical_knowledge.v2_contracts.runtime_contracts as rc

    fields = {f.name for f in dataclasses.fields(rc.V2TerminologyMatchSummary)}
    assert {
        "anonymous_id",
        "match_family",
        "terminology_system_family",
        "matches_count",
        "review_required",
        "auto_accept_allowed",
        "licensed_row_content_included",
        "public_report_safe",
    }.issubset(fields)
    for forbidden in (
        "code",
        "display",
        "rxcui",
        "loinc_num",
        "synonym",
        "definition",
        "concept",
        "raw_text",
        "ocr_text",
    ):
        assert forbidden not in fields, (
            f"terminology contract unexpectedly exposes {forbidden!r}"
        )


def test_review_item_default_is_pending_review():
    import clinical_knowledge.v2_contracts.runtime_contracts as rc

    classification = rc.V2ClassificationResult(
        anonymous_id="record_001",
        candidates=(),
    )
    item = rc.V2ReviewItem(
        anonymous_id="record_001",
        classification=classification,
    )
    assert item.disposition == rc.V2ReviewDisposition.PENDING_REVIEW
    assert item.review_required is True
    assert item.auto_accept_allowed is False


# ── V1 health-check catalog + V2 matrix ───────────────────────────────────


def test_v1_health_check_catalog_has_five_entries():
    from clinical_knowledge.v2_contracts import validation_harness as vh

    assert len(vh.V1_HEALTH_CHECKS) == 5
    names = vh.v1_health_check_names()
    assert set(names) == {
        "cka_final_mvp_release",
        "b07_term01_opt_in_integration",
        "medai_route_fix01",
        "medai_ui_ops_panel",
        "medai_ui_boot_fix_startup_resilience",
    }


def test_v1_health_check_external_api_used_is_false_for_every_entry():
    from clinical_knowledge.v2_contracts import validation_harness as vh

    for hc in vh.V1_HEALTH_CHECKS:
        assert hc.expected_external_api_used is False, hc.name


def test_v2_validation_matrix_has_all_eight_categories():
    from clinical_knowledge.v2_contracts import validation_harness as vh

    assert len(vh.V2_VALIDATION_MATRIX) == 8
    cats = vh.v2_validation_matrix_categories()
    assert set(cats) == {
        "A_contract_import_and_side_effect_safety",
        "B_contract_inventory_conformance",
        "C_safety_profile_conformance",
        "D_terminology_aggregate_only_conformance",
        "E_review_hitl_conformance",
        "F_reports_privacy_conformance",
        "G_runtime_non_modification_conformance",
        "H_parking_freeze_preservation_conformance",
    }


def test_v2_validation_matrix_in_json_report_matches(payload):
    matrix = payload["section_d_v2_validation_matrix"]
    assert set(matrix.keys()) == {
        "A_contract_import_and_side_effect_safety",
        "B_contract_inventory_conformance",
        "C_safety_profile_conformance",
        "D_terminology_aggregate_only_conformance",
        "E_review_hitl_conformance",
        "F_reports_privacy_conformance",
        "G_runtime_non_modification_conformance",
        "H_parking_freeze_preservation_conformance",
    }
    for cat, entries in matrix.items():
        assert isinstance(entries, list) and len(entries) >= 3, (
            f"category {cat} expected at least 3 invariants, got {len(entries)}"
        )


# ── No runtime files modified by this block ───────────────────────────────


def test_app_main_unchanged_by_v2_validation_harness_01():
    src = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "MEDAI-V2-VALIDATION-HARNESS-01" not in src
    assert "validation_harness" not in src


def test_launchers_unchanged_by_v2_validation_harness_01():
    for launcher in (
        "Start_MedAI_UI.bat",
        "Start_MedAI_UI_Silent.vbs",
        "Start_MedAI_Test_UI.bat",
        "Start_MedAI_UI_Encrypted.bat",
    ):
        p = REPO_ROOT / launcher
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        assert "MEDAI-V2-VALIDATION-HARNESS-01" not in src


def test_startup_preflight_unchanged_by_v2_validation_harness_01():
    p = REPO_ROOT / "app" / "startup_preflight.py"
    if not p.is_file():
        return
    src = p.read_text(encoding="utf-8")
    assert "MEDAI-V2-VALIDATION-HARNESS-01" not in src


def test_config_unchanged_by_v2_validation_harness_01():
    p = REPO_ROOT / "app" / "config.py"
    if not p.is_file():
        return
    src = p.read_text(encoding="utf-8")
    assert "MEDAI-V2-VALIDATION-HARNESS-01" not in src


def test_v1_validation_scripts_unchanged_by_this_block():
    """The V1 five-validation scripts must not mention the V2 harness."""
    for s in (
        "run_cka_final_mvp_release_validation.py",
        "run_b07_term01_opt_in_integration_validation.py",
        "run_medai_route_fix01_validation.py",
        "run_medai_ui_ops_panel_validation.py",
        "run_medai_ui_boot_fix_validation.py",
    ):
        p = REPO_ROOT / "scripts" / s
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8")
        assert "MEDAI-V2-VALIDATION-HARNESS-01" not in src
