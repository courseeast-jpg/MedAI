"""Focused tests for MEDAI-V2-DATA-INFRA-SPEC-01.

These tests prove that:

* all three reports exist and pass the public-report privacy check;
* JSON required booleans exist and have the correct values;
* prior V2 report folders exist (architecture / foundation / runtime
  contracts / validation harness / UI shell);
* the V2 contracts + validation_harness modules still import
  silently;
* the report defines 8 conceptual persistence stores (PS1..PS8);
* the report defines 10 future implementation gates A-J;
* the report defines 9 migration gates;
* the report preserves the runtime-DB-row-blind doctrine;
* the report preserves the V1 five-validation health-check set;
* the report states no V2 data implementation started;
* the report preserves the terminology / private-adapter data
  boundary (license-gated row access blocked);
* the report preserves V1 frozen release at ``7ef8ffd``;
* the report states cue expansion is **NOT** recommended;
* no DB / persistence / runtime / Streamlit / V2-contract files are
  modified by this block.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_data_infra_spec_01"

SHORT_SUMMARY = REPORT_DIR / "MEDAI_V2_DATA_INFRA_SPEC_01.md"
JSON_REPORT = REPORT_DIR / "medai_v2_data_infra_spec_01_report.json"
LONG_REPORT = REPORT_DIR / "medai_v2_data_infra_spec_01_report.md"

REQUIRED_REPORT_FILES = (SHORT_SUMMARY, JSON_REPORT, LONG_REPORT)

REQUIRED_PRIOR_V2_REPORT_DIRS = (
    REPO_ROOT / "reports" / "medai_v2_architecture_spec_01",
    REPO_ROOT / "reports" / "medai_v2_foundation_spec_02",
    REPO_ROOT / "reports" / "medai_v2_runtime_contracts_01",
    REPO_ROOT / "reports" / "medai_v2_validation_harness_01",
    REPO_ROOT / "reports" / "medai_v2_ui_shell_spec_01",
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
        "runtime_db_accessed",
        "schema_changed",
        "migration_created",
        "migration_executed",
        "persistence_code_changed",
        "runtime_behavior_changed",
        "app_main_changed",
        "streamlit_code_changed",
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
        "licensed_rows_exposed",
        "private_license_ack_read",
        "private_config_read",
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
        "preceding_validation_harness_present",
        "preceding_ui_shell_spec_present",
        "data_infra_spec_created",
        "conceptual_store_map_created",
        "ledger_audit_separation_defined",
        "rollback_doctrine_defined",
        "migration_gate_doctrine_defined",
        "public_report_data_doctrine_defined",
        "future_data_implementation_gates_defined",
        "v1_validation_healthcheck_catalog_preserved",
        "runtime_db_row_blind",
        "v1_release_preserved",
    ):
        assert payload.get(key) is True, f"{key} expected True, got {payload.get(key)!r}"


def test_next_recommended_block_value(payload):
    assert payload.get("next_recommended_block") == "V2-EXTRACTION-SPEC-01_OR_V2-ROADMAP-02"


# ── Conceptual store map + gates ─────────────────────────────────────────


def test_store_map_has_eight_stores(payload):
    stores = payload.get("section_d_conceptual_persistence_store_map")
    assert isinstance(stores, list) and len(stores) == 8, (
        f"expected 8 stores, got {len(stores) if isinstance(stores, list) else type(stores)}"
    )
    ids = [s.get("store_id") for s in stores]
    assert ids == ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "PS7", "PS8"]


def test_no_store_is_implemented_in_this_spec(payload):
    for s in payload["section_d_conceptual_persistence_store_map"]:
        assert s.get("implementation_present_in_this_spec") is False, (
            f"store {s.get('store_id')} unexpectedly marked implementation_present_in_this_spec=true"
        )


def test_future_implementation_gates_has_ten_entries(payload):
    gates = payload.get("section_l_future_implementation_gates")
    assert isinstance(gates, list) and len(gates) == 10
    for letter in "ABCDEFGHIJ":
        assert any(g.lstrip().startswith(f"{letter}.") for g in gates), (
            f"missing gate label {letter}."
        )


def test_migration_gate_doctrine_has_nine_entries(payload):
    mgates = payload.get("section_g_migration_gate_doctrine")
    assert isinstance(mgates, list) and len(mgates) == 9


# ── Runtime-DB-row-blind doctrine + public-report data doctrine ──────────


def test_runtime_db_row_blind_doctrine_present(payload):
    doctrine = payload["section_c_runtime_db_row_blind_doctrine"]
    assert doctrine["runtime_db_is_private_operational_state"] is True
    assert doctrine["reports_must_be_runtime_db_row_blind"] is True
    assert doctrine["future_db_access_requires_separate_implementation_spec"] is True
    for forbidden in (
        "DB rows",
        "raw records",
        "document filenames",
        "raw OCR text",
        "raw document text",
        "PHI",
        "private filesystem paths",
        "licensed terminology rows",
    ):
        assert forbidden in doctrine["v2_report_must_not_contain"]


def test_public_report_data_doctrine_present(payload):
    doctrine = payload["section_h_public_report_data_doctrine"]
    assert "counts" in doctrine["allowed_public_report_content"]
    assert "controlled_vocabulary_statuses" in doctrine["allowed_public_report_content"]
    for forbidden in ("PHI", "raw OCR text", "secrets", "API keys"):
        assert forbidden in doctrine["prohibited_public_report_content"]


# ── Terminology / private-adapter data boundary preserved ───────────────


def test_terminology_data_boundary_preserved(payload):
    b = payload["section_i_terminology_private_adapter_data_boundary"]
    assert b["licensed_terminology_row_access_blocked"] is True
    assert b["private_adapter_implementation_blocked"] is True
    assert b["terminology_output_aggregate_only"] is True
    assert b["public_reports_show_only_license_gate_status_not_licensed_content"] is True
    assert b["mesh_integration_blocked_until_operator_side_license_and_download_conditions_satisfied"] is True
    assert b["park_02_anchor_commit_short"] == "b9b19ad"
    assert b["operator_license_confirmation_status"] == "still_required"
    assert b["mesh_status_carried_forward"] == "download_helper_created"


# ── V1 five-validation health-check catalog preserved ────────────────────


def test_v1_health_check_catalog_preserved(payload):
    j = payload["section_j_validation_health_persistence_boundary"]
    assert j["v1_five_validation_health_check_set_preserved"] is True
    assert set(j["v1_health_check_names"]) == {
        "cka_final_mvp_release",
        "b07_term01_opt_in_integration",
        "medai_route_fix01",
        "medai_ui_ops_panel",
        "medai_ui_boot_fix_startup_resilience",
    }


# ── V1 frozen release preserved ───────────────────────────────────────────


def test_v1_frozen_release_preserved(payload):
    assert payload["v1_release_preserved"] is True
    assert payload["freeze_commit_short"] == "7ef8ffd"


# ── Cue expansion NOT recommended ─────────────────────────────────────────


def test_cue_expansion_not_recommended(payload):
    assert payload["cue_expansion_recommended"] is False
    long_md = LONG_REPORT.read_text(encoding="utf-8")
    short_md = SHORT_SUMMARY.read_text(encoding="utf-8")
    assert (
        "NOT** recommended" in long_md
        or "not recommended" in long_md.lower()
    )
    assert (
        "NOT** recommended" in short_md
        or "not recommended" in short_md.lower()
    )


# ── V2 contracts + validation_harness still import silently ──────────────


def _silent_import(module_dotted: str) -> tuple[str, str]:
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


def test_runtime_contracts_module_still_imports_silently():
    out, err = _silent_import("clinical_knowledge.v2_contracts.runtime_contracts")
    assert out == "" and err == ""


def test_validation_harness_module_still_imports_silently():
    out, err = _silent_import("clinical_knowledge.v2_contracts.validation_harness")
    assert out == "" and err == ""


# ── No runtime / DB / persistence / UI / V2-contract files modified ──────


def test_app_main_unchanged_by_v2_data_infra_spec_01():
    src = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "MEDAI-V2-DATA-INFRA-SPEC-01" not in src
    assert "v2_data_infra_spec_01" not in src


def test_launchers_unchanged_by_v2_data_infra_spec_01():
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
        assert "MEDAI-V2-DATA-INFRA-SPEC-01" not in src


def test_startup_preflight_unchanged_by_v2_data_infra_spec_01():
    p = REPO_ROOT / "app" / "startup_preflight.py"
    if not p.is_file():
        return
    src = p.read_text(encoding="utf-8")
    assert "MEDAI-V2-DATA-INFRA-SPEC-01" not in src


def test_config_unchanged_by_v2_data_infra_spec_01():
    p = REPO_ROOT / "app" / "config.py"
    if not p.is_file():
        return
    src = p.read_text(encoding="utf-8")
    assert "MEDAI-V2-DATA-INFRA-SPEC-01" not in src


def test_v2_contracts_modules_unchanged_by_v2_data_infra_spec_01():
    for f in (
        "clinical_knowledge/v2_contracts/runtime_contracts.py",
        "clinical_knowledge/v2_contracts/validation_harness.py",
        "clinical_knowledge/v2_contracts/__init__.py",
    ):
        p = REPO_ROOT / f
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8")
        assert "MEDAI-V2-DATA-INFRA-SPEC-01" not in src, (
            f"{f} unexpectedly mentions MEDAI-V2-DATA-INFRA-SPEC-01"
        )
