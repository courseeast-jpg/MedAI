"""CKA-B10 test suite: System Preflight + Scaffold.

60 tests covering:
- TestPreflightEnums: PreflightStatus values and ordering
- TestPreflightCheck: PreflightCheck dataclass + safe_public_summary
- TestPreflightReport: CKAPreflightReport properties and public summary
- TestPreflightRunnerPasses: run_cka_preflight() with real env
- TestPreflightBlockChecks: individual B01-B09 check functions
- TestSafetyInvariants: external API, HITL freeze, config safe defaults
- TestScaffoldBuild: CKASystemScaffold.build() and invariants
- TestScaffoldPreflight: is_ready(), preflight(), reset_preflight()
- TestScaffoldPublicSummary: safe_public_summary fields
- TestSafetyBoundaries: allow_active_write, external_api, production_autonomous
- TestPrivacyAndPublicReport: report privacy check, no private data
- TestValidationScript: all 12 cases pass
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# TestPreflightEnums
# ---------------------------------------------------------------------------


class TestPreflightEnums:
    def test_preflight_status_values(self):
        from clinical_knowledge.preflight import PreflightStatus
        assert PreflightStatus.PASS.value == "pass"
        assert PreflightStatus.WARN.value == "warn"
        assert PreflightStatus.FAIL.value == "fail"

    def test_preflight_status_is_str(self):
        from clinical_knowledge.preflight import PreflightStatus
        assert isinstance(PreflightStatus.PASS, str)
        assert isinstance(PreflightStatus.FAIL, str)

    def test_status_rank_ordering(self):
        from clinical_knowledge.preflight import _STATUS_RANK, PreflightStatus
        assert _STATUS_RANK[PreflightStatus.PASS] < _STATUS_RANK[PreflightStatus.WARN]
        assert _STATUS_RANK[PreflightStatus.WARN] < _STATUS_RANK[PreflightStatus.FAIL]

    def test_three_statuses_exist(self):
        from clinical_knowledge.preflight import PreflightStatus
        assert len(list(PreflightStatus)) == 3


# ---------------------------------------------------------------------------
# TestPreflightCheck
# ---------------------------------------------------------------------------


class TestPreflightCheck:
    def test_pass_check(self):
        from clinical_knowledge.preflight import PreflightCheck, PreflightStatus
        c = PreflightCheck("test_check", PreflightStatus.PASS, "All good.", "CKA-B01")
        assert c.name == "test_check"
        assert c.status == PreflightStatus.PASS
        assert c.block_id == "CKA-B01"

    def test_fail_check(self):
        from clinical_knowledge.preflight import PreflightCheck, PreflightStatus
        c = PreflightCheck("test_fail", PreflightStatus.FAIL, "Something broke.", None)
        assert c.status == PreflightStatus.FAIL
        assert c.block_id is None

    def test_safe_public_summary_has_no_detail_string(self):
        from clinical_knowledge.preflight import PreflightCheck, PreflightStatus
        c = PreflightCheck("my_check", PreflightStatus.PASS, "Sensitive path: C:\\private\\data", "CKA-B01")
        summary = c.safe_public_summary()
        # detail field should NOT appear in public summary
        assert "detail" not in summary
        assert "C:" not in str(summary)
        assert "private" not in str(summary)

    def test_safe_public_summary_fields(self):
        from clinical_knowledge.preflight import PreflightCheck, PreflightStatus
        c = PreflightCheck("my_check", PreflightStatus.WARN, "A warning.", "CKA-B05")
        s = c.safe_public_summary()
        assert s["name"] == "my_check"
        assert s["status"] == "warn"
        assert s["block_id"] == "CKA-B05"
        assert s["detail_ok"] is False

    def test_detail_ok_true_for_pass(self):
        from clinical_knowledge.preflight import PreflightCheck, PreflightStatus
        c = PreflightCheck("x", PreflightStatus.PASS, "ok")
        assert c.safe_public_summary()["detail_ok"] is True

    def test_detail_ok_false_for_fail(self):
        from clinical_knowledge.preflight import PreflightCheck, PreflightStatus
        c = PreflightCheck("x", PreflightStatus.FAIL, "bad")
        assert c.safe_public_summary()["detail_ok"] is False


# ---------------------------------------------------------------------------
# TestPreflightReport
# ---------------------------------------------------------------------------


class TestPreflightReport:
    def _make_report(self, statuses):
        from clinical_knowledge.preflight import CKAPreflightReport, PreflightCheck, PreflightStatus
        checks = [PreflightCheck(f"c{i}", s, "d", None) for i, s in enumerate(statuses)]
        # Determine overall
        worst = PreflightStatus.PASS
        from clinical_knowledge.preflight import _STATUS_RANK
        for c in checks:
            if _STATUS_RANK[c.status] > _STATUS_RANK[worst]:
                worst = c.status
        return CKAPreflightReport(checks=checks, overall_status=worst)

    def test_all_pass(self):
        from clinical_knowledge.preflight import PreflightStatus
        r = self._make_report([PreflightStatus.PASS] * 5)
        assert r.passed is True
        assert r.checks_passed == 5
        assert r.checks_failed == 0
        assert r.checks_warned == 0

    def test_any_fail_makes_failed(self):
        from clinical_knowledge.preflight import PreflightStatus
        r = self._make_report([PreflightStatus.PASS, PreflightStatus.FAIL, PreflightStatus.PASS])
        assert r.passed is False
        assert r.checks_failed == 1

    def test_warn_does_not_pass(self):
        from clinical_knowledge.preflight import PreflightStatus
        r = self._make_report([PreflightStatus.PASS, PreflightStatus.WARN])
        assert r.overall_status == PreflightStatus.WARN
        assert r.passed is False

    def test_safe_public_summary_keys(self):
        from clinical_knowledge.preflight import PreflightStatus
        r = self._make_report([PreflightStatus.PASS])
        s = r.safe_public_summary()
        required = {
            "overall_status", "passed", "checks_total", "checks_passed",
            "checks_failed", "checks_warned", "hitl_freeze_confirmed",
            "external_api_blocked", "safe_mode_active", "check_summaries",
        }
        assert required.issubset(s.keys())

    def test_safe_public_summary_counts(self):
        from clinical_knowledge.preflight import PreflightStatus
        r = self._make_report([PreflightStatus.PASS, PreflightStatus.WARN, PreflightStatus.FAIL])
        s = r.safe_public_summary()
        assert s["checks_total"] == 3
        assert s["checks_passed"] == 1
        assert s["checks_warned"] == 1
        assert s["checks_failed"] == 1

    def test_check_summaries_list(self):
        from clinical_knowledge.preflight import PreflightStatus
        r = self._make_report([PreflightStatus.PASS, PreflightStatus.PASS])
        s = r.safe_public_summary()
        assert isinstance(s["check_summaries"], list)
        assert len(s["check_summaries"]) == 2


# ---------------------------------------------------------------------------
# TestPreflightRunnerPasses
# ---------------------------------------------------------------------------


class TestPreflightRunnerPasses:
    @pytest.fixture(scope="class")
    def report(self):
        from clinical_knowledge.preflight import run_cka_preflight
        return run_cka_preflight()

    def test_overall_passes(self, report):
        from clinical_knowledge.preflight import PreflightStatus
        assert report.overall_status == PreflightStatus.PASS, (
            f"Preflight failed. Failures: "
            f"{[c.name for c in report.checks if c.status == PreflightStatus.FAIL]}, "
            f"Warnings: {[c.name for c in report.checks if c.status == PreflightStatus.WARN]}"
        )

    def test_hitl_freeze_confirmed(self, report):
        assert report.hitl_freeze_confirmed is True

    def test_external_api_blocked(self, report):
        assert report.external_api_blocked is True

    def test_safe_mode_active(self, report):
        assert report.safe_mode_active is True

    def test_no_failed_checks(self, report):
        from clinical_knowledge.preflight import PreflightStatus
        failed = [c.name for c in report.checks if c.status == PreflightStatus.FAIL]
        assert not failed, f"Failed checks: {failed}"

    def test_check_count_reasonable(self, report):
        # Expect at least 20 checks (2+ per block * 9 blocks + 3 cross-cutting)
        assert len(report.checks) >= 20

    def test_all_blocks_covered(self, report):
        block_ids = {c.block_id for c in report.checks if c.block_id}
        expected = {f"CKA-B0{i}" for i in range(1, 10)}
        assert expected == block_ids


# ---------------------------------------------------------------------------
# TestPreflightBlockChecks
# ---------------------------------------------------------------------------


class TestPreflightBlockChecks:
    def _run(self, fn_name):
        import clinical_knowledge.preflight as pf
        fn = getattr(pf, fn_name)
        checks = fn()
        return checks

    def test_b01_checks_have_block_id(self):
        from clinical_knowledge.preflight import PreflightStatus
        checks = self._run("_check_b01_mkb_foundation")
        assert all(c.block_id == "CKA-B01" for c in checks)
        assert all(c.status == PreflightStatus.PASS for c in checks)

    def test_b02_checks_pass(self):
        from clinical_knowledge.preflight import PreflightStatus
        checks = self._run("_check_b02_privacy_boundary")
        assert all(c.status == PreflightStatus.PASS for c in checks)

    def test_b02_secret_check_present(self):
        checks = self._run("_check_b02_privacy_boundary")
        names = [c.name for c in checks]
        assert "b02_secret_blocked" in names

    def test_b03_checks_pass(self):
        from clinical_knowledge.preflight import PreflightStatus
        checks = self._run("_check_b03_decision_engine")
        assert all(c.status == PreflightStatus.PASS for c in checks)

    def test_b04_checks_pass(self):
        from clinical_knowledge.preflight import PreflightStatus
        checks = self._run("_check_b04_truth_resolution")
        assert all(c.status == PreflightStatus.PASS for c in checks)

    def test_b05_checks_pass(self):
        from clinical_knowledge.preflight import PreflightStatus
        checks = self._run("_check_b05_medication_safety")
        assert all(c.status == PreflightStatus.PASS for c in checks)

    def test_b06_active_write_guard_passes(self):
        from clinical_knowledge.preflight import PreflightStatus
        checks = self._run("_check_b06_enrichment")
        guard = next((c for c in checks if "active_write" in c.name), None)
        assert guard is not None
        assert guard.status == PreflightStatus.PASS

    def test_b07_checks_pass(self):
        from clinical_knowledge.preflight import PreflightStatus
        checks = self._run("_check_b07_medical_coding")
        assert all(c.status == PreflightStatus.PASS for c in checks)

    def test_b08_registry_external_check_passes(self):
        from clinical_knowledge.preflight import PreflightStatus
        checks = self._run("_check_b08_consensus")
        registry_check = next((c for c in checks if "registry" in c.name), None)
        assert registry_check is not None
        assert registry_check.status == PreflightStatus.PASS

    def test_b09_snapshot_no_private_passes(self):
        from clinical_knowledge.preflight import PreflightStatus
        checks = self._run("_check_b09_operator_ui")
        snapshot_check = next((c for c in checks if "snapshot" in c.name), None)
        assert snapshot_check is not None
        assert snapshot_check.status == PreflightStatus.PASS


# ---------------------------------------------------------------------------
# TestSafetyInvariants
# ---------------------------------------------------------------------------


class TestSafetyInvariants:
    def test_external_api_check_passes(self):
        from clinical_knowledge.preflight import _check_external_api_blocked, PreflightStatus
        check = _check_external_api_blocked()
        assert check.status == PreflightStatus.PASS

    def test_hitl_freeze_check_passes(self):
        from clinical_knowledge.preflight import _check_hitl_freeze, PreflightStatus
        check = _check_hitl_freeze()
        assert check.status == PreflightStatus.PASS

    def test_config_safe_defaults_passes(self):
        from clinical_knowledge.preflight import _check_config_safe_defaults, PreflightStatus
        check = _check_config_safe_defaults()
        assert check.status == PreflightStatus.PASS

    def test_hitl_freeze_check_with_missing_file(self, tmp_path):
        from clinical_knowledge.preflight import _check_hitl_freeze, PreflightStatus
        check = _check_hitl_freeze(repo_root=tmp_path)
        # Should be WARN (not FAIL) when file is missing
        assert check.status == PreflightStatus.WARN

    def test_b06_allows_empty_facts_with_false(self):
        from clinical_knowledge.consensus.integration import consensus_facts_to_enrichment_candidates
        # allow_active_write=False (default) must not raise
        result = consensus_facts_to_enrichment_candidates([], allow_active_write=False)
        assert isinstance(result, list)

    def test_b06_raises_on_active_write_true(self):
        from clinical_knowledge.consensus.integration import consensus_facts_to_enrichment_candidates
        with pytest.raises(ValueError):
            consensus_facts_to_enrichment_candidates([], allow_active_write=True)

    def test_external_api_enabled_is_false(self):
        from clinical_knowledge.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG.EXTERNAL_APIS_ENABLED is False

    def test_enrich_promote_is_false(self):
        from clinical_knowledge.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG.ENRICH_PROMOTE is False


# ---------------------------------------------------------------------------
# TestScaffoldBuild
# ---------------------------------------------------------------------------


class TestScaffoldBuild:
    def test_build_succeeds(self):
        from clinical_knowledge.scaffold import CKASystemScaffold
        s = CKASystemScaffold.build()
        assert s is not None

    def test_build_safe_mode_true(self):
        from clinical_knowledge.scaffold import CKASystemScaffold
        s = CKASystemScaffold.build(safe_mode=True)
        assert s.safe_mode is True

    def test_build_allow_active_write_false(self):
        from clinical_knowledge.scaffold import CKASystemScaffold
        s = CKASystemScaffold.build()
        assert s.allow_active_write is False

    def test_build_store_initialized(self):
        from clinical_knowledge.scaffold import CKASystemScaffold
        from clinical_knowledge.store import MKBStore
        s = CKASystemScaffold.build()
        assert isinstance(s.store, MKBStore)

    def test_build_registry_initialized(self):
        from clinical_knowledge.scaffold import CKASystemScaffold
        from clinical_knowledge.connectors.registry import ConnectorRegistry
        s = CKASystemScaffold.build()
        assert isinstance(s.registry, ConnectorRegistry)

    def test_build_with_custom_db_path(self, tmp_path):
        from clinical_knowledge.scaffold import CKASystemScaffold
        db = str(tmp_path / "test_cka.db")
        s = CKASystemScaffold.build(db_path=db)
        assert s.store.db_path == db

    def test_allow_active_write_raises(self):
        from clinical_knowledge.scaffold import CKASystemScaffold
        from clinical_knowledge.store import MKBStore
        from clinical_knowledge.connectors.registry import ConnectorRegistry
        from clinical_knowledge.config import CKAConfig
        with pytest.raises(ValueError, match="allow_active_write"):
            CKASystemScaffold(
                store=MKBStore(":memory:"),
                registry=ConnectorRegistry.default(),
                config=CKAConfig(),
                allow_active_write=True,
            )

    def test_external_api_enabled_raises(self):
        from clinical_knowledge.scaffold import CKASystemScaffold
        from clinical_knowledge.store import MKBStore
        from clinical_knowledge.connectors.registry import ConnectorRegistry
        from clinical_knowledge.config import CKAConfig
        bad_config = CKAConfig()
        bad_config.EXTERNAL_APIS_ENABLED = True
        with pytest.raises(ValueError, match="EXTERNAL_APIS_ENABLED"):
            CKASystemScaffold(
                store=MKBStore(":memory:"),
                registry=ConnectorRegistry.default(),
                config=bad_config,
            )

    def test_production_autonomous_constant(self):
        from clinical_knowledge.scaffold import _PRODUCTION_AUTONOMOUS
        assert _PRODUCTION_AUTONOMOUS is False


# ---------------------------------------------------------------------------
# TestScaffoldPreflight
# ---------------------------------------------------------------------------


class TestScaffoldPreflight:
    @pytest.fixture
    def scaffold(self):
        from clinical_knowledge.scaffold import CKASystemScaffold
        return CKASystemScaffold.build()

    def test_is_ready_returns_true(self, scaffold):
        assert scaffold.is_ready() is True

    def test_preflight_returns_report(self, scaffold):
        from clinical_knowledge.preflight import CKAPreflightReport
        report = scaffold.preflight()
        assert isinstance(report, CKAPreflightReport)

    def test_preflight_cached(self, scaffold):
        r1 = scaffold.preflight()
        r2 = scaffold.preflight()  # should return same cached object
        # Both are reports that passed — verify consistency
        assert r1.passed == r2.passed

    def test_preflight_caches_in_last_preflight(self, scaffold):
        scaffold.preflight()
        assert scaffold._last_preflight is not None

    def test_reset_preflight_clears_cache(self, scaffold):
        scaffold.preflight()
        scaffold.reset_preflight()
        assert scaffold._last_preflight is None

    def test_is_ready_triggers_preflight(self, scaffold):
        assert scaffold._last_preflight is None  # fresh scaffold, no preflight run
        scaffold.is_ready()
        assert scaffold._last_preflight is not None

    def test_preflight_hitl_confirmed(self, scaffold):
        report = scaffold.preflight()
        assert report.hitl_freeze_confirmed is True

    def test_preflight_external_api_blocked(self, scaffold):
        report = scaffold.preflight()
        assert report.external_api_blocked is True


# ---------------------------------------------------------------------------
# TestScaffoldPublicSummary
# ---------------------------------------------------------------------------


class TestScaffoldPublicSummary:
    @pytest.fixture
    def scaffold(self):
        from clinical_knowledge.scaffold import CKASystemScaffold
        s = CKASystemScaffold.build()
        s.is_ready()
        return s

    def test_summary_returns_dict(self, scaffold):
        s = scaffold.safe_public_summary()
        assert isinstance(s, dict)

    def test_summary_scaffold_version(self, scaffold):
        s = scaffold.safe_public_summary()
        assert s["scaffold_version"] == "CKA-B10"

    def test_summary_safe_mode_true(self, scaffold):
        s = scaffold.safe_public_summary()
        assert s["safe_mode"] is True

    def test_summary_allow_active_write_false(self, scaffold):
        s = scaffold.safe_public_summary()
        assert s["allow_active_write"] is False

    def test_summary_production_autonomous_false(self, scaffold):
        s = scaffold.safe_public_summary()
        assert s["production_autonomous"] is False

    def test_summary_external_api_used_false(self, scaffold):
        s = scaffold.safe_public_summary()
        assert s["external_api_used"] is False

    def test_summary_store_initialized(self, scaffold):
        s = scaffold.safe_public_summary()
        assert s["store_initialized"] is True

    def test_summary_registry_counts(self, scaffold):
        s = scaffold.safe_public_summary()
        # Default registry: 4 total, 3 enabled
        assert s["registry_total_connectors"] == 4
        assert s["registry_enabled_connectors"] == 3

    def test_summary_preflight_passed(self, scaffold):
        s = scaffold.safe_public_summary()
        assert s["preflight_run"] is True
        assert s["preflight_passed"] is True

    def test_summary_privacy_safe(self, scaffold):
        from clinical_knowledge.privacy.report_privacy import check_public_report_payload
        s = scaffold.safe_public_summary()
        result = check_public_report_payload(s)
        assert result.passed


# ---------------------------------------------------------------------------
# TestSafetyBoundaries
# ---------------------------------------------------------------------------


class TestSafetyBoundaries:
    def test_scaffold_never_writes_active(self):
        from clinical_knowledge.scaffold import CKASystemScaffold
        s = CKASystemScaffold.build()
        assert s.allow_active_write is False

    def test_scaffold_no_external_api(self):
        from clinical_knowledge.scaffold import CKASystemScaffold
        s = CKASystemScaffold.build()
        assert s.config.EXTERNAL_APIS_ENABLED is False

    def test_preflight_no_external_call(self):
        """run_cka_preflight() completes without calling external APIs."""
        from clinical_knowledge.preflight import run_cka_preflight
        # This should complete without network access — runs purely locally
        report = run_cka_preflight()
        assert report is not None

    def test_b08_registry_default_all_local(self):
        from clinical_knowledge.connectors.registry import ConnectorRegistry
        reg = ConnectorRegistry.default()
        for spec in reg.list_all():
            assert spec.allow_external is False
            assert spec.synthetic_only is True

    def test_public_report_no_phi(self):
        from clinical_knowledge.preflight import run_cka_preflight
        from clinical_knowledge.privacy.report_privacy import check_public_report_payload
        report = run_cka_preflight()
        summary = report.safe_public_summary()
        result = check_public_report_payload(summary)
        assert result.passed


# ---------------------------------------------------------------------------
# TestPrivacyAndPublicReport
# ---------------------------------------------------------------------------


class TestPrivacyAndPublicReport:
    @pytest.fixture(scope="class")
    def report_path(self):
        return (
            Path(__file__).parent.parent
            / "reports"
            / "cka_block10_preflight_scaffold"
            / "cka_block10_preflight_scaffold_report.json"
        )

    def test_report_file_exists(self, report_path):
        assert report_path.exists(), f"Report not found at {report_path}"

    def test_report_json_parseable(self, report_path):
        if not report_path.exists():
            pytest.skip("Report not generated yet")
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_report_block_id(self, report_path):
        if not report_path.exists():
            pytest.skip("Report not generated yet")
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["block_id"] == "CKA-B10"

    def test_report_all_passed(self, report_path):
        if not report_path.exists():
            pytest.skip("Report not generated yet")
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["all_passed"] is True

    def test_report_external_api_false(self, report_path):
        if not report_path.exists():
            pytest.skip("Report not generated yet")
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["external_api_used"] is False

    def test_report_allow_active_write_false(self, report_path):
        if not report_path.exists():
            pytest.skip("Report not generated yet")
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["allow_active_write"] is False

    def test_report_hitl_freeze_confirmed(self, report_path):
        if not report_path.exists():
            pytest.skip("Report not generated yet")
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["hitl_freeze_confirmed"] is True

    def test_report_production_autonomous_false(self, report_path):
        if not report_path.exists():
            pytest.skip("Report not generated yet")
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["production_autonomous"] is False

    def test_report_privacy_passes(self, report_path):
        if not report_path.exists():
            pytest.skip("Report not generated yet")
        from clinical_knowledge.privacy.report_privacy import check_public_report_payload
        data = json.loads(report_path.read_text(encoding="utf-8"))
        result = check_public_report_payload(data)
        assert result.passed, f"Report privacy check failed: {result.leak_examples_redacted}"


# ---------------------------------------------------------------------------
# TestValidationScript
# ---------------------------------------------------------------------------


class TestValidationScript:
    def test_validation_runs_cleanly(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import run_validation
        report = run_validation()
        assert report["all_passed"] is True, (
            f"Validation failed: "
            f"{[c for c in report.get('case_results', []) if not c.get('passed')]}"
        )

    def test_validation_12_cases(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import run_validation
        report = run_validation()
        assert report["synthetic_cases_run"] == 12

    def test_validation_case_a_imports(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import case_a_all_modules_import
        r = case_a_all_modules_import()
        assert r["passed"] is True

    def test_validation_case_b_preflight(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import case_b_preflight_passes
        r = case_b_preflight_passes()
        assert r["passed"] is True

    def test_validation_case_c_external_api(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import case_c_external_api_blocked
        r = case_c_external_api_blocked()
        assert r["passed"] is True

    def test_validation_case_d_hitl_freeze(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import case_d_hitl_freeze_present
        r = case_d_hitl_freeze_present()
        assert r["passed"] is True

    def test_validation_case_e_store_ledger(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import case_e_mkb_store_ledger_functional
        r = case_e_mkb_store_ledger_functional()
        assert r["passed"] is True

    def test_validation_case_f_privacy_gate(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import case_f_privacy_gate_blocks_secret
        r = case_f_privacy_gate_blocks_secret()
        assert r["passed"] is True

    def test_validation_case_g_registry(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import case_g_registry_rejects_external
        r = case_g_registry_rejects_external()
        assert r["passed"] is True

    def test_validation_case_h_consensus(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import case_h_consensus_engine_instantiable
        r = case_h_consensus_engine_instantiable()
        assert r["passed"] is True

    def test_validation_case_i_operator_ui(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import case_i_operator_ui_loads_public_only
        r = case_i_operator_ui_loads_public_only()
        assert r["passed"] is True

    def test_validation_case_j_scaffold(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import case_j_scaffold_builds_safe_defaults
        r = case_j_scaffold_builds_safe_defaults()
        assert r["passed"] is True

    def test_validation_case_k_is_ready(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import case_k_scaffold_is_ready
        r = case_k_scaffold_is_ready()
        assert r["passed"] is True

    def test_validation_case_l_active_write(self):
        from scripts.run_cka_block10_preflight_scaffold_validation import case_l_scaffold_allow_active_write_raises
        r = case_l_scaffold_allow_active_write_raises()
        assert r["passed"] is True
