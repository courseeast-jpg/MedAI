"""Focused tests for MEDAI-V2-FOUNDATION-SPEC-02.

These tests prove the foundation SPEC is reports-only, that the three
required reports exist and carry the correct invariant booleans, that
public reports do not embed PHI / private paths / secrets / raw
filenames / licensed rows, that cue expansion is recorded as NOT
recommended, that V2 implementation did not start, that all parked /
frozen / blocked track anchors are referenced, that the stop-on-
failure doctrine and future block taxonomy are present, and that no
runtime files were modified by this block.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_foundation_spec_02"

SHORT_SUMMARY = REPORT_DIR / "MEDAI_V2_FOUNDATION_SPEC_02.md"
JSON_REPORT = REPORT_DIR / "medai_v2_foundation_spec_02_report.json"
LONG_REPORT = REPORT_DIR / "medai_v2_foundation_spec_02_report.md"

REQUIRED_REPORT_FILES = (SHORT_SUMMARY, JSON_REPORT, LONG_REPORT)

REQUIRED_FALSE_BOOLEANS = (
    "v2_implementation_started",
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
)

REQUIRED_TRUE_BOOLEANS = (
    "reports_only",
    "v1_release_preserved",
    "foundation_doctrine_created",
    "stop_on_failure_rules_defined",
    "future_block_taxonomy_defined",
)

REQUIRED_PARKED_FROZEN_COMMIT_ANCHORS = (
    "7ef8ffd",
    "3e46461",
    "9f9e22d",
    "f4d3cc6",
    "748c32a",
    "1b14ffe",
    "6b31678",
    "91b9eba",
    "e398a75",
    "b9b19ad",
    "60f1114",
    "cedbbd3",
    "376ca4e",
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(JSON_REPORT.read_text(encoding="utf-8"))


# ── 1. All three reports exist ─────────────────────────────────────────────


def test_all_three_reports_exist():
    for p in REQUIRED_REPORT_FILES:
        assert p.is_file(), f"missing report file: {p.relative_to(REPO_ROOT)}"


# ── 2. JSON required booleans exist and have correct values ────────────────


def test_required_false_booleans(payload):
    for key in REQUIRED_FALSE_BOOLEANS:
        assert key in payload, f"missing required key: {key}"
        assert payload[key] is False, f"{key} expected False, got {payload[key]!r}"


def test_required_true_booleans(payload):
    for key in REQUIRED_TRUE_BOOLEANS:
        assert key in payload, f"missing required key: {key}"
        assert payload[key] is True, f"{key} expected True, got {payload[key]!r}"


# ── 3. Reports do not embed PHI / private paths / secrets / raw filenames / licensed rows ──


def test_public_report_privacy_check_passes_for_all_three_reports():
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


# ── 4. Reports state cue expansion is not recommended ──────────────────────


def test_reports_state_cue_expansion_not_recommended(payload):
    assert payload["cue_expansion_recommended"] is False
    long_md = LONG_REPORT.read_text(encoding="utf-8")
    short_md = SHORT_SUMMARY.read_text(encoding="utf-8")
    assert "NOT** recommended" in long_md or "not recommended" in long_md.lower()
    assert "NOT** recommended" in short_md or "not recommended" in short_md.lower()


# ── 5. Reports state V2 implementation did not start ───────────────────────


def test_reports_state_v2_implementation_did_not_start(payload):
    assert payload["v2_implementation_started"] is False
    assert payload["reports_only"] is True


# ── 6. Reports preserve frozen / parked track anchors ──────────────────────


def test_all_parked_frozen_anchors_referenced_in_json(payload):
    payload_text = json.dumps(payload, indent=2, ensure_ascii=False)
    for anchor in REQUIRED_PARKED_FROZEN_COMMIT_ANCHORS:
        assert anchor in payload_text, f"missing anchor {anchor!r} in JSON report"


def test_v1_release_preserved_flag_set(payload):
    assert payload["v1_release_preserved"] is True


# ── 7. Reports include stop-on-failure doctrine ────────────────────────────


def test_stop_on_failure_doctrine_present(payload):
    assert payload["stop_on_failure_rules_defined"] is True
    rules = payload.get("section_d_stop_on_failure_rules")
    assert isinstance(rules, list) and len(rules) >= 8, (
        "expected at least 8 stop-on-failure rules"
    )


# ── 8. Reports include future block taxonomy ───────────────────────────────


def test_future_block_taxonomy_present(payload):
    assert payload["future_block_taxonomy_defined"] is True
    taxonomy = payload.get("section_c_block_taxonomy")
    assert isinstance(taxonomy, dict)
    allowed = taxonomy.get("allowed_block_classes")
    disallowed = taxonomy.get("disallowed_block_classes_unless_separately_approved")
    assert isinstance(allowed, list) and len(allowed) >= 8, (
        "expected at least 8 allowed block classes"
    )
    assert isinstance(disallowed, list) and len(disallowed) >= 10, (
        "expected at least 10 disallowed block classes"
    )


# ── 9. No runtime files modified by this block ─────────────────────────────


def test_app_main_unchanged_by_v2_foundation_spec_02():
    src = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "MEDAI-V2-FOUNDATION-SPEC-02" not in src
    assert "medai_v2_foundation_spec_02" not in src


def test_launchers_unchanged_by_v2_foundation_spec_02():
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
        assert "MEDAI-V2-FOUNDATION-SPEC-02" not in src


def test_startup_preflight_unchanged_by_v2_foundation_spec_02():
    p = REPO_ROOT / "app" / "startup_preflight.py"
    if not p.is_file():
        return
    src = p.read_text(encoding="utf-8")
    assert "MEDAI-V2-FOUNDATION-SPEC-02" not in src


def test_config_unchanged_by_v2_foundation_spec_02():
    p = REPO_ROOT / "app" / "config.py"
    if not p.is_file():
        return
    src = p.read_text(encoding="utf-8")
    assert "MEDAI-V2-FOUNDATION-SPEC-02" not in src


# ── Recommended next block ────────────────────────────────────────────────


def test_next_recommended_block_is_runtime_contracts_01(payload):
    assert payload["next_recommended_block"] == "MEDAI-V2-RUNTIME-CONTRACTS-01"
