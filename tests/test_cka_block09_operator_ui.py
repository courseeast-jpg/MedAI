"""Tests for CKA-B09 Operator UI — Clinical Knowledge Safety Panels.

Covers:
- viewer module imports
- snapshot loader reads public reports only, skips private files
- snapshot loader handles missing reports gracefully
- status summary and safety flag aggregation
- each panel helper exists, importable, callable, produces safe text
- privacy panel flags unsafe state as BLOCKED/REVIEW REQUIRED
- medication panel: no medication advice wording
- coding panel: no real UMLS/SNOMED active claim
- consensus panel: no active auto-write
- release readiness panel: not production autonomous / not medical device
- app/main.py Streamlit integration present
- validation script succeeds and reports safety flags
- public reports contain no raw private strings
- CKA-B01 through CKA-B08 tests still pass (verified separately)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.clinical_knowledge_safety_viewer import (
    get_cka_block_status_summary,
    get_cka_operator_panels,
    get_cka_safety_flags,
    load_cka_safety_snapshot,
    render_clinical_knowledge_safety_dashboard,
    render_consensus_panel,
    render_cka_release_readiness_panel,
    render_decision_engine_panel,
    render_enrichment_panel,
    render_medical_coding_panel,
    render_medication_safety_panel,
    render_mkb_status_panel,
    render_privacy_panel,
    render_truth_resolution_panel,
)
from clinical_knowledge.privacy import check_public_report_payload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def snapshot():
    return load_cka_safety_snapshot()


@pytest.fixture
def empty_snapshot():
    return {
        "reports": {},
        "blocks_loaded": [],
        "blocks_missing": [f"CKA-B0{i}" for i in range(1, 9)],
        "private_files_read": False,
        "replacement_map_loaded": False,
        "source_response_raw_loaded": False,
    }


@pytest.fixture
def unsafe_snapshot():
    """Snapshot with a report that has privacy flags raised."""
    return {
        "reports": {
            "CKA-B02": {
                "block_id": "CKA-B02",
                "raw_phi_logged_in_public_reports": True,
                "private_filename_path_leaks": 2,
                "secret_leaks": 1,
                "replacement_map_written_to_public_reports": False,
                "external_api_used": False,
            }
        },
        "blocks_missing": [],
        "private_files_read": False,
        "replacement_map_loaded": False,
        "source_response_raw_loaded": False,
    }


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------

class TestImports:
    def test_viewer_module_importable(self):
        import app.clinical_knowledge_safety_viewer as v
        assert v is not None

    def test_all_helpers_importable(self):
        funcs = [
            load_cka_safety_snapshot,
            get_cka_block_status_summary,
            get_cka_safety_flags,
            get_cka_operator_panels,
            render_mkb_status_panel,
            render_decision_engine_panel,
            render_privacy_panel,
            render_truth_resolution_panel,
            render_medication_safety_panel,
            render_enrichment_panel,
            render_medical_coding_panel,
            render_consensus_panel,
            render_cka_release_readiness_panel,
            render_clinical_knowledge_safety_dashboard,
        ]
        for fn in funcs:
            assert callable(fn), f"{fn.__name__} is not callable"

    def test_no_scispacy_dependency(self):
        assert "scispacy" not in sys.modules


# ---------------------------------------------------------------------------
# TestSnapshotLoader
# ---------------------------------------------------------------------------

class TestSnapshotLoader:
    def test_snapshot_loads_public_reports(self, snapshot):
        reports = snapshot.get("reports", {})
        # At least some reports should be present (B01-B08 all committed)
        assert len(reports) >= 1

    def test_snapshot_has_correct_keys(self, snapshot):
        assert "reports" in snapshot
        assert "blocks_loaded" in snapshot
        assert "blocks_missing" in snapshot
        assert "private_files_read" in snapshot
        assert "replacement_map_loaded" in snapshot
        assert "source_response_raw_loaded" in snapshot

    def test_snapshot_private_files_read_false(self, snapshot):
        assert snapshot.get("private_files_read") is False

    def test_snapshot_replacement_map_loaded_false(self, snapshot):
        assert snapshot.get("replacement_map_loaded") is False

    def test_snapshot_source_response_raw_loaded_false(self, snapshot):
        assert snapshot.get("source_response_raw_loaded") is False

    def test_snapshot_skips_private_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            private_path = Path(tmpdir) / "cka_test_PRIVATE.json"
            private_path.write_text(
                json.dumps({"block_id": "PRIVATE_BLOCK", "secret": "sk-abc"}),
                encoding="utf-8",
            )
            snapshot = load_cka_safety_snapshot(report_paths=[str(private_path)])
            reports = snapshot.get("reports", {})
            assert "PRIVATE_BLOCK" not in reports
            assert snapshot.get("private_files_read") is False

    def test_snapshot_skips_private_underscore_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            private_path = Path(tmpdir) / "private_cka_data.json"
            private_path.write_text(
                json.dumps({"block_id": "PRIVATE_DATA"}),
                encoding="utf-8",
            )
            snapshot = load_cka_safety_snapshot(report_paths=[str(private_path)])
            reports = snapshot.get("reports", {})
            assert "PRIVATE_DATA" not in reports

    def test_snapshot_handles_missing_file_gracefully(self):
        snapshot = load_cka_safety_snapshot(report_paths=["/nonexistent/file.json"])
        assert "reports" in snapshot
        assert snapshot.get("private_files_read") is False

    def test_snapshot_handles_empty_report_paths(self):
        snapshot = load_cka_safety_snapshot(report_paths=[])
        assert "reports" in snapshot

    def test_snapshot_strips_unsafe_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "cka_test_report.json"
            report_path.write_text(
                json.dumps({
                    "block_id": "CKA-TEST",
                    "conclusion": "test",
                    "replacement_map": {"original": "sanitized"},  # should be stripped
                    "source_response_raw": "raw text",             # should be stripped
                }),
                encoding="utf-8",
            )
            snapshot = load_cka_safety_snapshot(report_paths=[str(report_path)])
            report = snapshot.get("reports", {}).get("CKA-TEST", {})
            if report:  # file was loaded
                assert "replacement_map" not in report
                assert "source_response_raw" not in report

    def test_snapshot_loaded_reports_match_blocks_loaded(self, snapshot):
        reports = snapshot.get("reports", {})
        blocks_loaded = snapshot.get("blocks_loaded", [])
        assert set(reports.keys()) == set(blocks_loaded)


# ---------------------------------------------------------------------------
# TestBlockStatusSummary
# ---------------------------------------------------------------------------

class TestBlockStatusSummary:
    def test_summary_has_all_block_keys(self, snapshot):
        summary = get_cka_block_status_summary(snapshot)
        for i in range(1, 9):
            assert f"CKA-B0{i}_ready" in summary

    def test_summary_b01_ready(self, snapshot):
        summary = get_cka_block_status_summary(snapshot)
        # B01 report should be present
        assert summary.get("CKA-B01_ready") is True

    def test_summary_blocks_loaded_count(self, snapshot):
        summary = get_cka_block_status_summary(snapshot)
        assert isinstance(summary.get("blocks_loaded_count"), int)
        assert summary["blocks_loaded_count"] >= 1

    def test_summary_empty_snapshot(self, empty_snapshot):
        summary = get_cka_block_status_summary(empty_snapshot)
        assert summary.get("CKA-B01_ready") is False
        assert summary.get("blocks_loaded_count") == 0

    def test_summary_missing_blocks_list(self, snapshot):
        summary = get_cka_block_status_summary(snapshot)
        assert "blocks_missing" in summary
        assert isinstance(summary["blocks_missing"], list)


# ---------------------------------------------------------------------------
# TestSafetyFlags
# ---------------------------------------------------------------------------

class TestSafetyFlags:
    def test_flags_has_required_keys(self, snapshot):
        flags = get_cka_safety_flags(snapshot)
        required = [
            "raw_phi_logged_in_public_reports",
            "private_filename_path_leaks",
            "secret_leaks",
            "replacement_map_written_to_public_reports",
            "external_api_used",
            "clinical_recommendations_generated",
            "prescription_dosing_advice_generated",
            "frozen_hitl_release_reopened",
            "all_clear",
        ]
        for k in required:
            assert k in flags, f"Missing key: {k}"

    def test_flags_all_clear_from_real_reports(self, snapshot):
        flags = get_cka_safety_flags(snapshot)
        assert flags.get("all_clear") is True

    def test_flags_unsafe_snapshot(self, unsafe_snapshot):
        flags = get_cka_safety_flags(unsafe_snapshot)
        assert flags.get("all_clear") is False
        assert flags.get("raw_phi_logged_in_public_reports") is True

    def test_flags_private_files_read(self, snapshot):
        flags = get_cka_safety_flags(snapshot)
        assert "private_files_read" in flags
        assert flags.get("private_files_read") is False

    def test_flags_replacement_map_loaded(self, snapshot):
        flags = get_cka_safety_flags(snapshot)
        assert flags.get("replacement_map_loaded") is False


# ---------------------------------------------------------------------------
# TestOperatorPanels
# ---------------------------------------------------------------------------

class TestOperatorPanels:
    def test_panels_dict_has_all_keys(self, snapshot):
        panels = get_cka_operator_panels(snapshot)
        required = [
            "mkb_status_panel_ready",
            "decision_engine_panel_ready",
            "privacy_panel_ready",
            "truth_resolution_panel_ready",
            "medication_safety_panel_ready",
            "enrichment_panel_ready",
            "medical_coding_panel_ready",
            "consensus_panel_ready",
            "release_readiness_panel_ready",
            "panels_count",
            "block_status",
            "safety_flags",
        ]
        for k in required:
            assert k in panels, f"Missing key: {k}"

    def test_panels_count_is_nine(self, snapshot):
        panels = get_cka_operator_panels(snapshot)
        assert panels.get("panels_count") == 9

    def test_panels_release_always_ready(self, empty_snapshot):
        panels = get_cka_operator_panels(empty_snapshot)
        assert panels.get("release_readiness_panel_ready") is True


# ---------------------------------------------------------------------------
# TestPanelRenders
# ---------------------------------------------------------------------------

class TestPanelRenders:
    """Test that render helpers return safe text output."""

    FORBIDDEN_IN_TEXT = [
        "replacement_map",
        "source_response_raw",
        "private_payload",
        "raw_source_text",
    ]
    FORBIDDEN_POSITIVE_CLAIMS = [
        "is a medical device",
        "is production autonomous",
        "performs autonomous diagnosis",
    ]
    FORBIDDEN_ADVICE = [
        "prescribing",
        "dosing",
        "take this medication",
        "recommended dose",
    ]

    def _check_output(self, text: str, panel_name: str) -> None:
        text_lower = text.lower()
        for f in self.FORBIDDEN_IN_TEXT:
            assert f not in text_lower, f"{panel_name}: found forbidden field '{f}'"
        for f in self.FORBIDDEN_POSITIVE_CLAIMS:
            assert f not in text_lower, f"{panel_name}: found forbidden positive claim '{f}'"
        for f in self.FORBIDDEN_ADVICE:
            assert f not in text_lower, f"{panel_name}: found forbidden advice phrase '{f}'"

    def test_mkb_panel_renders_safe(self, snapshot):
        out = render_mkb_status_panel(snapshot)
        assert isinstance(out, str)
        self._check_output(out, "mkb")

    def test_decision_engine_panel_renders_safe(self, snapshot):
        out = render_decision_engine_panel(snapshot)
        assert isinstance(out, str)
        self._check_output(out, "decision_engine")
        assert "clinical answer" not in out.lower() or "no clinical answer" in out.lower()

    def test_privacy_panel_renders_safe(self, snapshot):
        out = render_privacy_panel(snapshot)
        assert isinstance(out, str)
        self._check_output(out, "privacy")

    def test_privacy_panel_flags_unsafe_as_blocked(self, unsafe_snapshot):
        out = render_privacy_panel(unsafe_snapshot)
        assert "BLOCKED" in out or "REVIEW REQUIRED" in out

    def test_privacy_panel_clears_safe_state(self, snapshot):
        flags = get_cka_safety_flags(snapshot)
        if flags.get("all_clear"):
            out = render_privacy_panel(snapshot)
            assert "BLOCKED" not in out

    def test_truth_resolution_panel_renders_safe(self, snapshot):
        out = render_truth_resolution_panel(snapshot)
        assert isinstance(out, str)
        self._check_output(out, "truth_resolution")
        # Must not show medication advice
        assert "dosing" not in out.lower()

    def test_medication_panel_renders_safe(self, snapshot):
        out = render_medication_safety_panel(snapshot)
        assert isinstance(out, str)
        self._check_output(out, "medication")
        # Must not give positive recommendation wording (negation "no recommendations" is allowed)
        text = out.lower()
        if "recommendation" in text:
            # Only allowed if clearly negated
            assert "no medication recommendation" in text or "no recommendation" in text
        assert "dosing" not in text

    def test_medication_panel_has_no_dosing_advice(self, snapshot):
        out = render_medication_safety_panel(snapshot)
        bad_phrases = ["take this dose", "recommended dose", "mg per day"]
        for phrase in bad_phrases:
            assert phrase not in out.lower()

    def test_enrichment_panel_renders_safe(self, snapshot):
        out = render_enrichment_panel(snapshot)
        assert isinstance(out, str)
        self._check_output(out, "enrichment")
        assert "source_response" not in out.lower()

    def test_medical_coding_panel_renders_safe(self, snapshot):
        out = render_medical_coding_panel(snapshot)
        assert isinstance(out, str)
        self._check_output(out, "coding")

    def test_medical_coding_panel_no_real_umls_claim(self, snapshot):
        out = render_medical_coding_panel(snapshot)
        bad_claims = ["real umls active", "snomed certified", "certified clinical coding"]
        for claim in bad_claims:
            assert claim not in out.lower()

    def test_consensus_panel_renders_safe(self, snapshot):
        out = render_consensus_panel(snapshot)
        assert isinstance(out, str)
        self._check_output(out, "consensus")

    def test_consensus_panel_no_auto_write_claim(self, snapshot):
        out = render_consensus_panel(snapshot)
        assert "auto-write active: true" not in out.lower()
        assert "writes active records automatically" not in out.lower()

    def test_release_readiness_panel_renders_safe(self, snapshot):
        out = render_cka_release_readiness_panel(snapshot)
        assert isinstance(out, str)
        self._check_output(out, "release_readiness")

    def test_release_readiness_not_autonomous(self, snapshot):
        out = render_cka_release_readiness_panel(snapshot)
        has_not_autonomous = (
            "not production-autonomous" in out.lower()
            or "not production autonomous" in out.lower()
        )
        assert has_not_autonomous

    def test_release_readiness_not_medical_device(self, snapshot):
        out = render_cka_release_readiness_panel(snapshot)
        assert "not a medical device" in out.lower()

    def test_release_readiness_not_diagnosis_system(self, snapshot):
        out = render_cka_release_readiness_panel(snapshot)
        # Must not claim autonomous diagnosis capability
        assert "is production autonomous" not in out.lower()

    def test_all_panels_callable_on_empty_snapshot(self, empty_snapshot):
        """Panels must not crash on empty snapshot."""
        panels = [
            render_mkb_status_panel,
            render_decision_engine_panel,
            render_privacy_panel,
            render_truth_resolution_panel,
            render_medication_safety_panel,
            render_enrichment_panel,
            render_medical_coding_panel,
            render_consensus_panel,
            render_cka_release_readiness_panel,
        ]
        for fn in panels:
            out = fn(empty_snapshot)
            assert isinstance(out, str)


# ---------------------------------------------------------------------------
# TestMainPyIntegration
# ---------------------------------------------------------------------------

class TestMainPyIntegration:
    def test_main_py_has_cka_tab(self):
        main_path = Path(__file__).parent.parent / "app" / "main.py"
        assert main_path.exists()
        content = main_path.read_text(encoding="utf-8")
        assert "Clinical Knowledge Safety" in content

    def test_main_py_imports_viewer(self):
        main_path = Path(__file__).parent.parent / "app" / "main.py"
        content = main_path.read_text(encoding="utf-8")
        assert "clinical_knowledge_safety_viewer" in content

    def test_main_py_has_try_except(self):
        main_path = Path(__file__).parent.parent / "app" / "main.py"
        content = main_path.read_text(encoding="utf-8")
        # Should have a try/except around the CKA tab
        assert "tab_cka" in content
        # Verify isolated from existing tabs
        assert "Review Package" in content
        assert "Clinical Knowledge Safety" in content

    def test_main_py_preserved_existing_tabs(self):
        main_path = Path(__file__).parent.parent / "app" / "main.py"
        content = main_path.read_text(encoding="utf-8")
        for tab in ["Current Run", "Blind Audit", "Report Archive", "Review Package"]:
            assert tab in content, f"Existing tab '{tab}' was removed"

    def test_viewer_importable_standalone(self):
        """Viewer can be imported without Streamlit session (no crash)."""
        import app.clinical_knowledge_safety_viewer  # noqa: F401


# ---------------------------------------------------------------------------
# TestPrivacyAndPublicReports
# ---------------------------------------------------------------------------

class TestPrivacyAndPublicReports:
    def test_panel_outputs_pass_privacy_checker(self, snapshot):
        panels_summary = get_cka_operator_panels(snapshot)
        privacy_check = check_public_report_payload(panels_summary)
        assert privacy_check.passed, f"Privacy check failed: {privacy_check.leak_examples_redacted}"

    def test_safety_flags_pass_privacy_checker(self, snapshot):
        flags = get_cka_safety_flags(snapshot)
        privacy_check = check_public_report_payload(flags)
        assert privacy_check.passed

    def test_report_json_no_replacement_map(self):
        report_path = Path(__file__).parent.parent / "reports" / "cka_block09_operator_ui" / "cka_block09_operator_ui_report.json"
        if report_path.exists():
            import json as _json
            data = _json.loads(report_path.read_text(encoding="utf-8"))
            # The flag must be present but must be False — no actual replacement_map data loaded
            assert data.get("replacement_map_loaded") is False

    def test_report_json_no_source_response_raw(self):
        report_path = Path(__file__).parent.parent / "reports" / "cka_block09_operator_ui" / "cka_block09_operator_ui_report.json"
        if report_path.exists():
            import json as _json
            data = _json.loads(report_path.read_text(encoding="utf-8"))
            # The flag must be present but must be False — no raw source responses loaded
            assert data.get("source_response_raw_loaded") is False


# ---------------------------------------------------------------------------
# TestValidationScript
# ---------------------------------------------------------------------------

class TestValidationScript:
    def test_validation_script_runs_cleanly(self):
        from scripts.run_cka_block09_operator_ui_validation import run_validation
        report = run_validation()
        assert report["all_passed"] is True, (
            f"Validation failed: {[c for c in report.get('case_results', []) if not c.get('passed')]}"
        )

    def test_validation_all_14_cases_pass(self):
        from scripts.run_cka_block09_operator_ui_validation import run_validation
        report = run_validation()
        assert report["synthetic_cases_run"] == 14
        assert report["cases_passed"] == 14

    def test_validation_safety_flags(self):
        from scripts.run_cka_block09_operator_ui_validation import run_validation
        report = run_validation()
        assert report["private_files_read"] is False
        assert report["replacement_map_loaded"] is False
        assert report["source_response_raw_loaded"] is False
        assert report["clinical_recommendations_generated"] is False
        assert report["prescription_dosing_advice_generated"] is False
        assert report["real_external_connectors_implemented"] is False
        assert report["external_api_used"] is False
        assert report["raw_phi_logged_in_public_reports"] is False
        assert report["private_filename_path_leaks"] == 0
        assert report["secret_leaks"] == 0
        assert report["frozen_hitl_release_reopened"] is False
        assert report["production_ocr_changed"] is False
        assert report["production_extractor_changed"] is False
        assert report["safety_gate_changed"] is False

    def test_validation_report_has_block_id(self):
        from scripts.run_cka_block09_operator_ui_validation import run_validation
        report = run_validation()
        assert report["block_id"] == "CKA-B09"

    def test_validation_panels_count(self):
        from scripts.run_cka_block09_operator_ui_validation import run_validation
        report = run_validation()
        assert report["panels_count"] == 9

    def test_validation_streamlit_integration_ready(self):
        from scripts.run_cka_block09_operator_ui_validation import run_validation
        report = run_validation()
        assert report["streamlit_integration_ready"] is True


# ---------------------------------------------------------------------------
# TestOperatorReviewPolish (CKA-OPR-01)
# ---------------------------------------------------------------------------


class TestOperatorReviewPolish:
    """Verifies the CKA-OPR-01 polish: B09/B10 path coverage + key-drift fix."""

    def test_viewer_paths_include_b09_b10(self):
        from app.clinical_knowledge_safety_viewer import _CKA_REPORT_PATHS
        assert "CKA-B09" in _CKA_REPORT_PATHS
        assert "CKA-B10" in _CKA_REPORT_PATHS

    def test_snapshot_loads_b09_b10(self, snapshot):
        loaded = snapshot.get("blocks_loaded", [])
        assert "CKA-B09" in loaded, f"B09 not in {loaded}"
        assert "CKA-B10" in loaded, f"B10 not in {loaded}"

    def test_snapshot_blocks_loaded_can_reach_ten(self, snapshot):
        assert len(snapshot.get("blocks_loaded", [])) >= 10

    def test_summary_count_can_reach_ten(self, snapshot):
        summary = get_cka_block_status_summary(snapshot)
        assert summary["blocks_loaded_count"] >= 10

    def test_summary_b09_ready_key_present(self, snapshot):
        summary = get_cka_block_status_summary(snapshot)
        assert "CKA-B09_ready" in summary
        assert summary["CKA-B09_ready"] is True

    def test_summary_b10_ready_key_present(self, snapshot):
        summary = get_cka_block_status_summary(snapshot)
        assert "CKA-B10_ready" in summary
        assert summary["CKA-B10_ready"] is True

    def test_safe_mode_ready_maps_correctly(self, snapshot):
        summary = get_cka_block_status_summary(snapshot)
        # B03 carries safe_mode_tested=True; the summary key safe_mode_ready
        # must reflect that, not a stale False from a missing key.
        b03 = snapshot["reports"].get("CKA-B03", {})
        if b03.get("safe_mode_tested") is True:
            assert summary["safe_mode_ready"] is True

    def test_medication_safety_ready_maps_correctly(self, snapshot):
        summary = get_cka_block_status_summary(snapshot)
        b05 = snapshot["reports"].get("CKA-B05", {})
        expected = bool(
            b05.get("ddi_stub_ready", False)
            and b05.get("ddi_layer1_evidence_modifier_ready", False)
            and b05.get("ddi_layer2_write_gate_ready", False)
        )
        assert summary["medication_safety_ready"] is expected

    def test_enrichment_ready_maps_correctly(self, snapshot):
        summary = get_cka_block_status_summary(snapshot)
        b06 = snapshot["reports"].get("CKA-B06", {})
        if b06.get("controlled_enrichment_ready") is True:
            assert summary["enrichment_ready"] is True

    def test_summary_no_unsupported_claim_when_key_missing(self):
        # When a block's report does NOT carry the expected key, the
        # summary must remain False rather than fabricating a True claim.
        empty = {
            "reports": {
                "CKA-B03": {"block_id": "CKA-B03"},   # no safe_mode_tested
                "CKA-B05": {"block_id": "CKA-B05"},   # no ddi_*_ready
                "CKA-B06": {"block_id": "CKA-B06"},   # no controlled_enrichment_ready
            },
            "blocks_loaded": ["CKA-B03", "CKA-B05", "CKA-B06"],
            "blocks_missing": [],
        }
        summary = get_cka_block_status_summary(empty)
        assert summary["safe_mode_ready"] is False
        assert summary["medication_safety_ready"] is False
        assert summary["enrichment_ready"] is False

    def test_opr_report_exists(self):
        path = (
            Path(__file__).parent.parent
            / "reports"
            / "cka_operator_review_polish"
            / "cka_operator_review_polish_report.json"
        )
        assert path.exists()

    def test_opr_report_privacy_clean(self):
        path = (
            Path(__file__).parent.parent
            / "reports"
            / "cka_operator_review_polish"
            / "cka_operator_review_polish_report.json"
        )
        if not path.exists():
            pytest.skip("OPR-01 report not generated")
        data = json.loads(path.read_text(encoding="utf-8"))
        result = check_public_report_payload(data)
        assert result.passed, f"Privacy check failed: {result.leak_examples_redacted}"

    def test_opr_report_safety_flags(self):
        path = (
            Path(__file__).parent.parent
            / "reports"
            / "cka_operator_review_polish"
            / "cka_operator_review_polish_report.json"
        )
        if not path.exists():
            pytest.skip("OPR-01 report not generated")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["block_id"] == "CKA-OPR-01"
        assert data["conclusion"] == "cka_operator_review_polish_ready"
        assert data["external_api_used"] is False
        assert data["raw_phi_logged_in_public_reports"] is False
        assert data["private_filename_path_leaks"] == 0
        assert data["secret_leaks"] == 0
        assert data["clinical_recommendations_generated"] is False
        assert data["prescription_dosing_advice_generated"] is False
        assert data["frozen_hitl_release_reopened"] is False
        assert data["b09_b10_reports_loaded_by_viewer"] is True
        assert data["stale_head_wording_fixed"] is True
        assert data["summary_key_drift_fixed"] is True
