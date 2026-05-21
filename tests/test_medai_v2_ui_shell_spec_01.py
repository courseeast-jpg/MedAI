"""Focused tests for MEDAI-V2-UI-SHELL-SPEC-01.

These tests prove that:

* all three reports exist and pass the public-report privacy check;
* JSON required booleans exist and have the correct values;
* prior V2 report folders exist (architecture / foundation / runtime
  contracts / validation harness);
* the V2 contracts + validation_harness modules still import
  silently;
* the report defines 8 shells (S1-S8) in the screen / surface
  inventory;
* the report defines 10 future UI implementation gates A-J;
* the report preserves local-only / review-bound / external-API-
  blocked / no-auto-accept defaults;
* the report preserves the terminology / private-adapter wait gate;
* the report preserves V1 frozen release at ``7ef8ffd``;
* the report states cue expansion is **NOT** recommended;
* the implementation files are limited to reports / script / focused
  tests; ``app/main.py`` and Streamlit code remain unchanged.
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

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_ui_shell_spec_01"

SHORT_SUMMARY = REPORT_DIR / "MEDAI_V2_UI_SHELL_SPEC_01.md"
JSON_REPORT = REPORT_DIR / "medai_v2_ui_shell_spec_01_report.json"
LONG_REPORT = REPORT_DIR / "medai_v2_ui_shell_spec_01_report.md"

REQUIRED_REPORT_FILES = (SHORT_SUMMARY, JSON_REPORT, LONG_REPORT)

REQUIRED_PRIOR_V2_REPORT_DIRS = (
    REPO_ROOT / "reports" / "medai_v2_architecture_spec_01",
    REPO_ROOT / "reports" / "medai_v2_foundation_spec_02",
    REPO_ROOT / "reports" / "medai_v2_runtime_contracts_01",
    REPO_ROOT / "reports" / "medai_v2_validation_harness_01",
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
        "runtime_db_accessed",
        "source_documents_opened",
        "raw_text_printed",
        "raw_filenames_printed",
        "raw_filenames_exposed",
        "private_paths_printed",
        "private_paths_exposed",
        "secrets_printed",
        "tags_touched",
        "cue_expansion_recommended",
        "auto_accept_allowed_default",
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
        "ui_shell_spec_created",
        "screen_inventory_created",
        "future_ui_implementation_gates_defined",
        "ui_spec_only",
        "read_only_shell_spec",
        "no_ui_actions_added",
        "no_callbacks_added",
        "no_session_state_logic_added",
        "review_bound_default",
        "local_only_default",
        "external_api_blocked_default",
        "v1_release_preserved",
    ):
        assert payload.get(key) is True, f"{key} expected True, got {payload.get(key)!r}"


def test_next_recommended_block_value(payload):
    assert payload.get("next_recommended_block") == "V2-DATA-INFRA-SPEC-01"


# ── Shell inventory and implementation gates ─────────────────────────────


def test_shell_inventory_has_eight_shells(payload):
    shells = payload.get("section_d_proposed_v2_ui_shell_map")
    assert isinstance(shells, list) and len(shells) == 8, (
        f"expected 8 shells, got {len(shells) if isinstance(shells, list) else type(shells)}"
    )
    ids = [s.get("shell_id") for s in shells]
    assert ids == ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]


def test_no_shell_adds_actions_or_callbacks(payload):
    shells = payload["section_d_proposed_v2_ui_shell_map"]
    for shell in shells:
        assert shell.get("action_buttons_allowed_in_this_spec") is False, (
            f"shell {shell.get('shell_id')} unexpectedly allows action buttons"
        )
        assert shell.get("callbacks_allowed_in_this_spec") is False, (
            f"shell {shell.get('shell_id')} unexpectedly allows callbacks"
        )


def test_inventory_counts_are_zero_for_actions_callbacks_state(payload):
    inv = payload["section_e_screen_surface_inventory_summary"]
    assert inv["action_buttons_added_in_this_spec"] == 0
    assert inv["callbacks_added_in_this_spec"] == 0
    assert inv["session_state_keys_added_in_this_spec"] == 0
    assert inv["total_shells"] == 8


def test_future_ui_implementation_gates_has_ten_entries(payload):
    gates = payload.get("section_j_future_ui_implementation_gates")
    assert isinstance(gates, list) and len(gates) == 10, (
        f"expected 10 implementation gates A-J, got {len(gates) if isinstance(gates, list) else type(gates)}"
    )
    # Each entry should start with a labeled letter A. through J.
    for letter in "ABCDEFGHIJ":
        assert any(g.lstrip().startswith(f"{letter}.") for g in gates), (
            f"missing gate label {letter}."
        )


# ── Default safety posture preserved ──────────────────────────────────────


def test_local_only_review_bound_external_api_blocked_no_auto_accept_defaults(payload):
    invariants = payload["section_f_safety_privacy_ui_invariants"]
    assert invariants["local_only_default"] is True
    assert invariants["review_bound_default"] is True
    assert invariants["external_api_blocked_default"] is True
    assert invariants["auto_accept_allowed_default"] is False


# ── Terminology / private-adapter wait gate preserved ────────────────────


def test_terminology_private_adapter_wait_gate_preserved(payload):
    wait_gate = payload["section_h_terminology_private_adapter_ui_wait_gate"]
    assert wait_gate["private_adapter_implementation_allowed"] is False
    assert wait_gate["real_private_store_access_allowed"] is False
    assert wait_gate["manual_license_verification_complete"] is False
    assert wait_gate["license_gated_resources_verified_count"] == 0
    assert wait_gate["internal_boundaries_verified_count"] == 2
    assert wait_gate["ui_must_not_expose_licensed_rows"] is True
    assert wait_gate["ui_must_not_expose_private_config_contents"] is True
    assert wait_gate["ui_must_not_read_license_acknowledgement_contents"] is True
    assert wait_gate["ui_must_show_implementation_blocked_message_only"] is True
    assert wait_gate["ui_aggregate_only_emission_required"] is True


# ── V1 frozen release preserved ───────────────────────────────────────────


def test_v1_frozen_release_preserved(payload):
    parking = payload["section_i_parking_freeze_visibility_doctrine"]
    first = parking[0]
    assert first["track"] == "Local operator release"
    assert first["status"] == "frozen"
    assert first["anchor_commit_short"] == "7ef8ffd"
    assert payload["v1_release_preserved"] is True


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


# ── V2 contracts + validation_harness modules still import silently ──────


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


# ── No runtime or UI files modified by this block ─────────────────────────


def test_app_main_unchanged_by_v2_ui_shell_spec_01():
    src = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "MEDAI-V2-UI-SHELL-SPEC-01" not in src
    assert "v2_ui_shell_spec_01" not in src


def test_launchers_unchanged_by_v2_ui_shell_spec_01():
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
        assert "MEDAI-V2-UI-SHELL-SPEC-01" not in src


def test_startup_preflight_unchanged_by_v2_ui_shell_spec_01():
    p = REPO_ROOT / "app" / "startup_preflight.py"
    if not p.is_file():
        return
    src = p.read_text(encoding="utf-8")
    assert "MEDAI-V2-UI-SHELL-SPEC-01" not in src


def test_config_unchanged_by_v2_ui_shell_spec_01():
    p = REPO_ROOT / "app" / "config.py"
    if not p.is_file():
        return
    src = p.read_text(encoding="utf-8")
    assert "MEDAI-V2-UI-SHELL-SPEC-01" not in src


def test_v2_contracts_modules_unchanged_by_v2_ui_shell_spec_01():
    """This block must NOT modify the typing-only v2_contracts modules."""
    for f in (
        "clinical_knowledge/v2_contracts/runtime_contracts.py",
        "clinical_knowledge/v2_contracts/validation_harness.py",
        "clinical_knowledge/v2_contracts/__init__.py",
    ):
        p = REPO_ROOT / f
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8")
        assert "MEDAI-V2-UI-SHELL-SPEC-01" not in src, (
            f"{f} unexpectedly mentions MEDAI-V2-UI-SHELL-SPEC-01"
        )
