"""Focused tests for MEDAI-V2-FOUNDATION-DEFAULT-OFF-STATUS-REGISTRY-01."""
from __future__ import annotations

import ast
import contextlib
import importlib
import io
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_foundation_default_off_status_registry_01"
REPORTS = (
    REPORT_DIR / "MEDAI_V2_FOUNDATION_DEFAULT_OFF_STATUS_REGISTRY_01.md",
    REPORT_DIR / "medai_v2_foundation_default_off_status_registry_01_report.json",
    REPORT_DIR / "medai_v2_foundation_default_off_status_registry_01_report.md",
)
REGISTRY_PATH = REPO_ROOT / "clinical_knowledge" / "v2_foundation" / "status_registry.py"
ALLOWED_CHANGED_PATHS = {
    "clinical_knowledge/v2_foundation/__init__.py",
    "clinical_knowledge/v2_foundation/status_registry.py",
    "scripts/run_medai_v2_foundation_default_off_status_registry_01.py",
    "tests/test_medai_v2_foundation_default_off_status_registry_01.py",
    "reports/medai_v2_foundation_default_off_status_registry_01/MEDAI_V2_FOUNDATION_DEFAULT_OFF_STATUS_REGISTRY_01.md",
    "reports/medai_v2_foundation_default_off_status_registry_01/medai_v2_foundation_default_off_status_registry_01_report.json",
    "reports/medai_v2_foundation_default_off_status_registry_01/medai_v2_foundation_default_off_status_registry_01_report.md",
}
PRIOR_V2_REPORT_DIRS = (
    "medai_v2_architecture_spec_01",
    "medai_v2_foundation_spec_02",
    "medai_v2_runtime_contracts_01",
    "medai_v2_validation_harness_01",
    "medai_v2_ui_shell_spec_01",
    "medai_v2_data_infra_spec_01",
    "medai_v2_extraction_spec_01",
    "medai_v2_roadmap_02",
    "medai_v2_foundation_implementation_readiness_01",
    "medai_v2_packaging_spec_01",
    "medai_v2_roadmap_park_01",
    "medai_v2_foundation_default_off_implementation_plan_01",
)


def _payload() -> dict:
    return json.loads(
        (REPORT_DIR / "medai_v2_foundation_default_off_status_registry_01_report.json").read_text(
            encoding="utf-8"
        )
    )


def _report_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in REPORTS)


def _registry_module():
    return importlib.import_module("clinical_knowledge.v2_foundation.status_registry")


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_all_three_reports_exist():
    for path in REPORTS:
        assert path.is_file(), path


def test_json_required_booleans_have_expected_values():
    payload = _payload()
    false_keys = (
        "runtime_behavior_changed",
        "default_off_helper_runtime_wired",
        "default_off_helper_ui_wired",
        "app_main_changed",
        "streamlit_code_changed",
        "ui_changed",
        "launcher_changed",
        "installer_changed",
        "deployment_script_changed",
        "startup_config_changed",
        "extraction_changed",
        "ocr_changed",
        "ocr_routing_changed",
        "classifier_changed",
        "threshold_scoring_changed",
        "parser_behavior_changed",
        "fallback_behavior_changed",
        "cue_pack_changed",
        "db_schema_changed",
        "migration_created",
        "migration_executed",
        "persistence_code_changed",
        "clinical_behavior_changed",
        "ddi_behavior_changed",
        "terminology_behavior_changed",
        "private_adapter_implemented",
        "concrete_adapters_implemented",
        "runtime_wiring_added",
        "external_api_used",
        "private_data_accessed",
        "source_documents_opened",
        "raw_text_read",
        "raw_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "secrets_printed",
        "licensed_rows_read",
        "licensed_rows_exposed",
        "private_license_ack_read",
        "private_config_read",
        "runtime_db_accessed",
        "tags_touched",
        "tags_created",
        "tags_modified",
        "cue_expansion_recommended",
        "auto_accept_allowed_default",
        "status_registry_stdout_on_import",
        "status_registry_stderr_on_import",
    )
    true_keys = (
        "status_registry_created",
        "implementation_started",
        "default_off_helper_created",
        "v1_release_preserved",
        "local_only_default",
        "review_bound_default",
        "external_api_blocked_default",
        "all_entries_public_report_safe",
        "terminology_private_adapter_blocked",
        "cue_expansion_blocked",
        "clinical_decision_logic_expansion_blocked",
        "status_registry_standard_library_only",
        "status_registry_import_side_effect_free",
        "preceding_implementation_plan_present",
    )
    for key in false_keys:
        assert payload.get(key) is False, key
    for key in true_keys:
        assert payload.get(key) is True, key


def test_prior_v2_report_folders_exist():
    for name in PRIOR_V2_REPORT_DIRS:
        assert (REPO_ROOT / "reports" / name).is_dir(), name


def test_registry_imports_silently_and_uses_only_stdlib_imports():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        module = _registry_module()
    assert module is not None
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == ""
    assert _import_roots(REGISTRY_PATH) <= {"__future__", "dataclasses", "enum", "typing"}
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "streamlit",
        "requests",
        "httpx",
        "urllib",
        "socket",
        "http.",
        "sqlite3",
        "sqlcipher3",
        "psycopg",
        "anthropic",
        "openai",
        "google.generativeai",
        "boto3",
        "clinical_knowledge.terminology",
    ):
        assert forbidden not in text


def test_registry_exposes_required_api_and_entries():
    module = _registry_module()
    for name in (
        "V2CapabilityStatus",
        "V2CapabilityCategory",
        "V2CapabilityEntry",
        "get_v2_capability_registry",
        "get_v2_capability_by_id",
        "list_blocked_v2_capabilities",
        "list_default_enabled_v2_capabilities",
        "summarize_v2_capability_registry",
    ):
        assert hasattr(module, name), name
    registry = module.get_v2_capability_registry()
    assert isinstance(registry, tuple)
    assert len(registry) == 13
    assert module.get_v2_capability_by_id("missing") is None
    assert module.get_v2_capability_by_id("v2_foundation_status_registry") is not None


def test_registry_entries_are_default_off_unwired_and_public_safe():
    module = _registry_module()
    registry = module.get_v2_capability_registry()
    assert all(entry.default_enabled is False for entry in registry)
    assert all(entry.runtime_wired is False for entry in registry)
    assert all(entry.ui_wired is False for entry in registry)
    assert all(entry.public_report_safe is True for entry in registry)
    assert module.list_default_enabled_v2_capabilities() == ()


def test_blocked_entries_and_summary_are_public_safe_aggregates():
    module = _registry_module()
    blocked_ids = {entry.capability_id for entry in module.list_blocked_v2_capabilities()}
    assert blocked_ids == {
        "terminology_private_adapter",
        "cue_expansion",
        "clinical_decision_logic_expansion",
    }
    summary = module.summarize_v2_capability_registry()
    assert summary["total_count"] == 13
    assert summary["blocked_count"] == 3
    assert summary["default_enabled_count"] == 0
    assert summary["runtime_wired_count"] == 0
    assert summary["ui_wired_count"] == 0
    assert summary["public_report_safe_count"] == 13
    assert summary["all_default_off"] is True
    assert summary["no_runtime_wiring"] is True
    assert summary["no_ui_wiring"] is True
    assert summary["cue_expansion_blocked"] is True
    assert summary["terminology_private_adapter_blocked"] is True
    assert summary["clinical_decision_logic_expansion_blocked"] is True
    assert set(summary) == {
        "total_count",
        "counts_by_status",
        "counts_by_category",
        "blocked_count",
        "default_enabled_count",
        "runtime_wired_count",
        "ui_wired_count",
        "public_report_safe_count",
        "all_default_off",
        "no_runtime_wiring",
        "no_ui_wiring",
        "cue_expansion_blocked",
        "terminology_private_adapter_blocked",
        "clinical_decision_logic_expansion_blocked",
    }


def test_reports_capture_required_registry_posture():
    payload = _payload()
    text = _report_text()
    assert payload["registry_entry_count"] == 13
    assert payload["default_enabled_count"] == 0
    assert payload["runtime_wired_count"] == 0
    assert payload["ui_wired_count"] == 0
    assert payload["blocked_entry_count"] == 3
    assert payload["recommended_next_block"] == "V2-ROADMAP-03"
    assert payload["recommended_next_3_blocks"] == [
        "V2-ROADMAP-03",
        "V2-ROADMAP-PARK-02_OR_RELEASE-FREEZE-SNAPSHOT",
        "FREEZE-MAINTENANCE-ONLY_OR_NEXT_DEFAULT-OFF_PLAN",
    ]
    for phrase in (
        "Registry Implementation Summary",
        "Default-Off Guarantees",
        "Import And Side-Effect Guarantees",
        "Runtime And UI Non-Wiring Guarantees",
        "Cue expansion remains blocked and explicitly not recommended",
        "V1 frozen release remains",
    ):
        assert phrase in text


def test_reports_pass_public_report_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for path in REPORTS:
        content = path.read_text(encoding="utf-8")
        target: object = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, path.name


def test_forbidden_runtime_ui_launcher_db_extraction_files_do_not_import_registry():
    needles = ("v2_foundation.status_registry", "status_registry")
    forbidden_roots = (
        REPO_ROOT / "app",
        REPO_ROOT / "launchers",
        REPO_ROOT / "clinical_knowledge" / "extraction",
        REPO_ROOT / "clinical_knowledge" / "extractors",
        REPO_ROOT / "clinical_knowledge" / "ocr",
        REPO_ROOT / "clinical_knowledge" / "terminology",
        REPO_ROOT / "clinical_knowledge" / "classification",
        REPO_ROOT / "clinical_knowledge" / "classifier",
        REPO_ROOT / "clinical_knowledge" / "persistence",
        REPO_ROOT / "clinical_knowledge" / "db",
    )
    hits: list[str] = []
    for root in forbidden_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(needle in text for needle in needles):
                hits.append(str(path.relative_to(REPO_ROOT)))
    assert hits == []


def test_implementation_files_limited_to_allowed_scope():
    result = subprocess.run(
        ["git", "status", "--short", "-uall"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = {
        line[3:].replace("\\", "/")
        for line in result.stdout.splitlines()
        if line and line[:2].strip()
    }
    assert changed <= ALLOWED_CHANGED_PATHS
