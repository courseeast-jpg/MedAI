"""CKA-B11 final MVP release tests.

Verifies:
- final release script imports
- B01-B10 commit list is represented in the report
- final docs are generated
- continuation snapshot is generated
- final report contains required safety flags
- final report privacy checker passes
- no private files are included
- no raw PHI/path/secret appears in the public report
- no clinical recommendation wording appears
- no prescription dosing advice appears
- production_autonomous remains false
- frozen HITL release remains closed
- external APIs remain disabled
- scaffold rejects EXTERNAL_APIS_ENABLED=True
- scaffold rejects allow_active_write=True
- full CKA package imports
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent
RELEASE_DIR = REPO_ROOT / "reports" / "cka_final_mvp_release"

EXPECTED_BLOCKS = [
    ("CKA-B01", "04477ca"),
    ("CKA-B02", "f42be80"),
    ("CKA-B03", "da45b71"),
    ("CKA-B04", "7011079"),
    ("CKA-B05", "398568e"),
    ("CKA-B06", "02b7955"),
    ("CKA-B07", "0ad2815"),
    ("CKA-B08", "65aa131"),
    ("CKA-B09", "ff0adf2"),
    ("CKA-B10", "27d940e"),
]

REQUIRED_DOCS = [
    "CKA_OPERATOR_GUIDE.md",
    "CKA_LIMITATIONS_AND_SAFETY.md",
    "CKA_ARCHITECTURE_MANIFEST.md",
    "CKA_CONTINUATION_SNAPSHOT.md",
]


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------


class TestImports:
    def test_final_release_script_imports(self):
        from scripts import run_cka_final_mvp_release_validation as mod
        assert hasattr(mod, "run_validation")

    def test_full_cka_package_imports(self):
        import clinical_knowledge.preflight
        import clinical_knowledge.scaffold
        import clinical_knowledge.store
        import clinical_knowledge.privacy.sanitizer
        import clinical_knowledge.privacy.report_privacy
        import clinical_knowledge.decision_engine.engine
        import clinical_knowledge.truth_resolution.engine
        import clinical_knowledge.medication_safety.ddi_stub
        import clinical_knowledge.enrichment.integration
        import clinical_knowledge.medical_coding.synthetic_mapper
        import clinical_knowledge.connectors.registry
        import clinical_knowledge.consensus.engine
        import app.clinical_knowledge_safety_viewer
        # All imports complete without error
        assert True


# ---------------------------------------------------------------------------
# TestReleaseDocs
# ---------------------------------------------------------------------------


class TestReleaseDocs:
    def test_release_dir_exists(self):
        assert RELEASE_DIR.exists() and RELEASE_DIR.is_dir()

    @pytest.mark.parametrize("doc", REQUIRED_DOCS)
    def test_doc_present(self, doc):
        assert (RELEASE_DIR / doc).exists()

    @pytest.mark.parametrize("doc", REQUIRED_DOCS)
    def test_doc_non_empty(self, doc):
        path = RELEASE_DIR / doc
        if path.exists():
            assert len(path.read_text(encoding="utf-8")) > 200

    def test_continuation_snapshot_lists_b01_b10(self):
        snap = (RELEASE_DIR / "CKA_CONTINUATION_SNAPSHOT.md").read_text(encoding="utf-8")
        for block, sha in EXPECTED_BLOCKS:
            assert block in snap, f"{block} not in continuation snapshot"
            assert sha in snap, f"{sha} not in continuation snapshot"

    def test_manifest_lists_all_blocks(self):
        man = (RELEASE_DIR / "CKA_ARCHITECTURE_MANIFEST.md").read_text(encoding="utf-8")
        for block, sha in EXPECTED_BLOCKS:
            assert block in man
            assert sha in man

    def test_limitations_states_not_medical_device(self):
        lim = (RELEASE_DIR / "CKA_LIMITATIONS_AND_SAFETY.md").read_text(encoding="utf-8").lower()
        assert "not a medical device" in lim

    def test_limitations_states_not_prescribing(self):
        lim = (RELEASE_DIR / "CKA_LIMITATIONS_AND_SAFETY.md").read_text(encoding="utf-8").lower()
        assert "not prescribing" in lim or "not prescribing software" in lim

    def test_limitations_states_no_real_connectors(self):
        lim = (RELEASE_DIR / "CKA_LIMITATIONS_AND_SAFETY.md").read_text(encoding="utf-8").lower()
        assert "no real external connectors" in lim or "synthetic, local-only" in lim


# ---------------------------------------------------------------------------
# TestReleaseReport
# ---------------------------------------------------------------------------


def _load_or_run_release_report():
    """Read the release JSON from disk; if missing, run validation once."""
    report_path = RELEASE_DIR / "cka_final_mvp_release_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    import os
    os.environ["CKA_B11_SKIP_NESTED_PYTEST"] = "1"
    from scripts.run_cka_final_mvp_release_validation import run_validation
    return run_validation()


@pytest.fixture(scope="session")
def release_report():
    return _load_or_run_release_report()


class TestReleaseReport:
    @pytest.fixture(scope="class")
    def report(self, release_report):
        return release_report

    def test_report_dict(self, report):
        assert isinstance(report, dict)

    def test_block_id(self, report):
        assert report["block_id"] == "CKA-B11"

    def test_conclusion(self, report):
        assert report["conclusion"] == "cka_mvp_release_package_ready"

    def test_branch(self, report):
        assert report["branch"] == "clinical-knowledge-architecture"

    def test_head_commit_hex(self, report):
        assert isinstance(report["head_commit"], str)
        # 7-char short SHA (chosen to avoid matching the SECRET regex's
        # 40+-alnum branch in the public-report privacy checker).
        assert 7 <= len(report["head_commit"]) <= 12
        int(report["head_commit"], 16)  # valid hex

    def test_completed_blocks_b01_b10(self, report):
        expected = [b for b, _ in EXPECTED_BLOCKS]
        assert report["completed_blocks"] == expected

    def test_block_commits_listed(self, report):
        commits_in_report = {
            (e["block"], e["sha"]) for e in report["block_commits"]
        }
        assert commits_in_report == set(EXPECTED_BLOCKS)

    def test_all_tests_passed_flag(self, report):
        assert report["all_tests_passed"] is True

    def test_total_tests_passed_positive(self, report):
        assert report["total_tests_passed"] > 0

    def test_preflight_checks_passed_positive(self, report):
        assert report["preflight_checks_passed"] > 0

    def test_validation_scripts_passed_positive(self, report):
        assert report["validation_scripts_passed"] >= 1

    def test_external_api_used_false(self, report):
        assert report["external_api_used"] is False

    def test_raw_phi_logged_false(self, report):
        assert report["raw_phi_logged_in_public_reports"] is False

    def test_private_path_leaks_zero(self, report):
        assert report["private_filename_path_leaks"] == 0

    def test_secret_leaks_zero(self, report):
        assert report["secret_leaks"] == 0

    def test_clinical_recs_false(self, report):
        assert report["clinical_recommendations_generated"] is False

    def test_prescription_advice_false(self, report):
        assert report["prescription_dosing_advice_generated"] is False

    def test_production_ocr_unchanged(self, report):
        assert report["production_ocr_changed"] is False

    def test_production_extractor_unchanged(self, report):
        assert report["production_extractor_changed"] is False

    def test_safety_gate_unchanged(self, report):
        assert report["safety_gate_changed"] is False

    def test_frozen_hitl_release_closed(self, report):
        assert report["frozen_hitl_release_reopened"] is False

    def test_production_autonomous_false(self, report):
        assert report["production_autonomous"] is False

    def test_not_ready_for_real_connector_activation(self, report):
        assert report["cka_ready_for_real_connector_activation"] is False

    def test_ready_for_operator_review(self, report):
        assert report["cka_ready_for_operator_review"] is True

    def test_next_decision_present(self, report):
        assert isinstance(report["next_recommended_decision"], str)
        assert len(report["next_recommended_decision"]) > 30

    def test_continuation_snapshot_created(self, report):
        assert report["continuation_snapshot_created"] is True

    def test_release_docs_listed(self, report):
        assert set(report["final_release_docs_created"]) == set(REQUIRED_DOCS)


# ---------------------------------------------------------------------------
# TestPrivacyAndPublicReport
# ---------------------------------------------------------------------------


class TestPrivacyAndPublicReport:
    @pytest.fixture(scope="class")
    def report_path(self, release_report):
        # `release_report` ensures the JSON file is on disk before this fixture
        # is consumed (either it already existed, or run_validation wrote it).
        return RELEASE_DIR / "cka_final_mvp_release_report.json"

    def test_report_file_exists(self, report_path):
        assert report_path.exists()

    def test_report_json_parseable(self, report_path):
        if not report_path.exists():
            pytest.skip("Report not generated yet")
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_report_passes_privacy_checker(self, report_path):
        if not report_path.exists():
            pytest.skip("Report not generated yet")
        from clinical_knowledge.privacy.report_privacy import check_public_report_payload
        data = json.loads(report_path.read_text(encoding="utf-8"))
        result = check_public_report_payload(data)
        assert result.passed, f"Privacy check failed: {result.leak_examples_redacted}"

    def test_no_private_files_in_release_dir(self):
        if not RELEASE_DIR.exists():
            pytest.skip("Release dir not present")
        for p in RELEASE_DIR.iterdir():
            nm = p.name.lower()
            assert "_private." not in nm
            assert not nm.startswith("private_")
            assert not nm.endswith(".pdf")
            assert not nm.endswith(".jpg")
            assert not nm.endswith(".jpeg")
            assert not nm.endswith(".png")

    def test_report_no_replacement_map(self, report_path):
        if not report_path.exists():
            pytest.skip("Report not generated yet")
        content = report_path.read_text(encoding="utf-8")
        assert "replacement_map" not in content

    def test_report_no_source_response_raw(self, report_path):
        if not report_path.exists():
            pytest.skip("Report not generated yet")
        content = report_path.read_text(encoding="utf-8")
        assert "source_response_raw" not in content

    def test_report_no_windows_paths(self, report_path):
        if not report_path.exists():
            pytest.skip("Report not generated yet")
        content = report_path.read_text(encoding="utf-8")
        # No drive-letter paths
        import re
        assert not re.search(r"[A-Za-z]:\\\\", content)


# ---------------------------------------------------------------------------
# TestSafetyText
# ---------------------------------------------------------------------------


class TestSafetyText:
    @pytest.fixture(scope="class")
    def all_doc_text(self):
        text = ""
        for d in REQUIRED_DOCS:
            p = RELEASE_DIR / d
            if p.exists():
                text += "\n" + p.read_text(encoding="utf-8")
        return text.lower()

    def test_no_dosing_advice(self, all_doc_text):
        for forbidden in ("take this dose", "recommended dose is", "mg per day", "you should take"):
            assert forbidden not in all_doc_text

    def test_no_clinical_recommendation_phrases(self, all_doc_text):
        # "we recommend" or "the system recommends" should NOT appear as a clinical claim
        for phrase in ("we recommend administering", "the system recommends prescribing"):
            assert phrase not in all_doc_text

    def test_no_real_connector_claim(self, all_doc_text):
        assert "real dxgpt api active" not in all_doc_text
        assert "real sage api active" not in all_doc_text
        assert "real patientnotes api active" not in all_doc_text

    def test_explicit_synthetic_disclosure(self, all_doc_text):
        assert "synthetic" in all_doc_text


# ---------------------------------------------------------------------------
# TestSafetyBoundaries
# ---------------------------------------------------------------------------


class TestSafetyBoundaries:
    def test_external_apis_enabled_default_false(self):
        from clinical_knowledge.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG.EXTERNAL_APIS_ENABLED is False

    def test_enrich_promote_default_false(self):
        from clinical_knowledge.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG.ENRICH_PROMOTE is False

    def test_production_autonomous_constant(self):
        from clinical_knowledge.scaffold import _PRODUCTION_AUTONOMOUS
        assert _PRODUCTION_AUTONOMOUS is False

    def test_scaffold_rejects_external_apis_enabled(self):
        from clinical_knowledge.scaffold import CKASystemScaffold
        from clinical_knowledge.store import MKBStore
        from clinical_knowledge.connectors.registry import ConnectorRegistry
        from clinical_knowledge.config import CKAConfig
        bad = CKAConfig()
        bad.EXTERNAL_APIS_ENABLED = True
        with pytest.raises(ValueError, match="EXTERNAL_APIS_ENABLED"):
            CKASystemScaffold(
                store=MKBStore(":memory:"),
                registry=ConnectorRegistry.default(),
                config=bad,
            )

    def test_scaffold_rejects_allow_active_write(self):
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

    def test_consensus_active_write_raises(self):
        from clinical_knowledge.consensus.integration import (
            consensus_facts_to_enrichment_candidates,
        )
        with pytest.raises(ValueError):
            consensus_facts_to_enrichment_candidates([], allow_active_write=True)

    def test_registry_default_all_local_synthetic(self):
        from clinical_knowledge.connectors.registry import ConnectorRegistry
        for spec in ConnectorRegistry.default().list_all():
            assert spec.allow_external is False
            assert spec.synthetic_only is True

    def test_preflight_overall_passes(self):
        from clinical_knowledge.preflight import run_cka_preflight, PreflightStatus
        report = run_cka_preflight()
        assert report.overall_status == PreflightStatus.PASS


# ---------------------------------------------------------------------------
# TestFrozenHITLRelease
# ---------------------------------------------------------------------------


class TestFrozenHITLRelease:
    """Verify the frozen HITL release artifacts are not modified by CKA."""

    def test_freeze_doc_present(self):
        freeze = REPO_ROOT / "MEDAI_OCR_LAYOUT_HITL_RELEASE_FREEZE.md"
        assert freeze.exists()

    def test_freeze_doc_content_states_freeze(self):
        freeze = REPO_ROOT / "MEDAI_OCR_LAYOUT_HITL_RELEASE_FREEZE.md"
        if not freeze.exists():
            pytest.skip("Freeze doc absent")
        content = freeze.read_text(encoding="utf-8").lower()
        # The freeze doc must continue to assert freeze status
        assert "freeze" in content or "frozen" in content


# ---------------------------------------------------------------------------
# TestValidationScript
# ---------------------------------------------------------------------------


class TestValidationScript:
    def test_validation_all_passed(self, release_report):
        assert release_report["all_passed"] is True, (
            f"Validation failed: "
            f"{[c for c in release_report.get('case_results', []) if not c.get('passed')]}"
        )

    def test_validation_12_cases(self, release_report):
        assert release_report["synthetic_cases_run"] == 12


# ---------------------------------------------------------------------------
# TestDocFreshnessAfterPolish (CKA-OPR-01)
# ---------------------------------------------------------------------------


class TestDocFreshnessAfterPolish:
    """Verifies the CKA-OPR-01 doc-drift fixes are applied."""

    def test_continuation_snapshot_lists_b11_commit(self):
        snap = (RELEASE_DIR / "CKA_CONTINUATION_SNAPSHOT.md").read_text(encoding="utf-8")
        assert "07860eb" in snap, "B11 commit hash missing from continuation snapshot"

    def test_continuation_snapshot_no_stale_head_claim(self):
        snap = (RELEASE_DIR / "CKA_CONTINUATION_SNAPSHOT.md").read_text(encoding="utf-8")
        assert "Until that commit is" not in snap, (
            "Stale 'until that commit is created' wording still present"
        )
        assert "HEAD is `27d940e`" not in snap, (
            "Stale 'HEAD is 27d940e' claim still present"
        )

    def test_manifest_b11_has_concrete_hash(self):
        man = (RELEASE_DIR / "CKA_ARCHITECTURE_MANIFEST.md").read_text(encoding="utf-8")
        assert "(this commit)" not in man
        assert "07860eb" in man

    def test_operator_guide_mentions_windows_launcher(self):
        guide = (RELEASE_DIR / "CKA_OPERATOR_GUIDE.md").read_text(encoding="utf-8")
        assert "Start_MedAI_UI.bat" in guide

    def test_operator_guide_mentions_final_validation_command(self):
        guide = (RELEASE_DIR / "CKA_OPERATOR_GUIDE.md").read_text(encoding="utf-8")
        assert "run_cka_final_mvp_release_validation.py" in guide

    def test_operator_guide_states_b01_b10_panel_coverage(self):
        guide = (RELEASE_DIR / "CKA_OPERATOR_GUIDE.md").read_text(encoding="utf-8")
        # Either explicit "B01-B10" wording or the path range must appear
        assert "B01-B10" in guide or "block01" in guide and "block10" in guide
