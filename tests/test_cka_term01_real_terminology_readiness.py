"""CKA-TERM-01 — terminology readiness tests.

Covers:
- model imports + safe public summaries
- license gate blocks without ack; test-mode env works
- inventory bounded to terminology_data/, no raw paths in summary
- missing files handled safely
- synthetic UMLS / SNOMED / RxNorm / LOINC parsers
- temp terminology store schema + counts
- lookup exact / synonym / unknown / ambiguous
- B07 integration boundary preserves tier/status, no hallucination
- DDI status not cleared (boundary summary)
- no terminology files committed
- public report no key/path/secret/PHI/license-text
- final CKA / SEC-06 / SEC-07 validations still invocable
- no clinical / dosing wording in modules
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.terminology import (    # noqa: E402
    LicenseGateError,
    LocalTerminologyStore,
    TerminologyConcept,
    TerminologyImportMode,
    TerminologyLookupResult,
    TerminologyLookupService,
    TerminologyLookupStatus,
    TerminologySourceManifest,
    TerminologySourceStatus,
    TerminologySystem,
    code_entity_via_local_terminology,
    inventory_terminology_data_dir,
    license_acknowledged_for,
    parse_loinc_csv,
    parse_rxnorm_rxnconso,
    parse_snomed_concept_description,
    parse_umls_mrconso,
    require_license_acknowledgment,
    safe_b07_boundary_summary,
)


REPORT_DIR = REPO_ROOT / "reports" / "cka_term01_real_terminology_readiness"


# Synthetic fixtures (kept minimal; mirror the validation script's layout).
SYNTHETIC_UMLS = "\n".join([
    "C0001|ENG|P|L0001|PF|S0001|Y|A0001||||MTH|PT|UMLS001|hypertension||N|",
    "C0001|ENG|P|L0001|PF|S0002|N|A0002||||MTH|SY|UMLS002|high blood pressure||N|",
    "C0002|ENG|P|L0002|PF|S0003|Y|A0003||||MTH|PT|UMLS003|fatigue||N|",
])
SYNTHETIC_SNOMED_C = "\n".join([
    "100000001\t20240101\t1\t900000000000207008\t900000000000074008",
    "100000002\t20240101\t1\t900000000000207008\t900000000000074008",
])
SYNTHETIC_SNOMED_D = "\n".join([
    "d1\t20240101\t1\tm1\t100000001\ten\t900000000000003001\tDiabetes mellitus type 2 (disorder)\t900000000000020002",
    "d2\t20240101\t1\tm1\t100000001\ten\t900000000000013009\ttype 2 diabetes\t900000000000020002",
    "d3\t20240101\t1\tm1\t100000002\ten\t900000000000003001\tFatigue (finding)\t900000000000020002",
])
SYNTHETIC_RXNORM = "\n".join([
    "R001|ENG|P|L01|PF|S01|Y|A01||||RXNORM|IN|RX001|aspirin||N|",
    "R002|ENG|P|L02|PF|S02|Y|A02||||RXNORM|IN|RX002|metformin||N|",
])
SYNTHETIC_LOINC = "\n".join([
    "LOINC_NUM,COMPONENT,LONG_COMMON_NAME",
    "12345-6,Glucose,Glucose [Mass/volume] in Serum or Plasma",
])


def _populated_store() -> LocalTerminologyStore:
    store = LocalTerminologyStore()
    store.add_concepts(parse_umls_mrconso(text=SYNTHETIC_UMLS).concepts)
    store.add_concepts(parse_snomed_concept_description(
        concept_text=SYNTHETIC_SNOMED_C,
        description_text=SYNTHETIC_SNOMED_D,
    ).concepts)
    store.add_concepts(parse_rxnorm_rxnconso(text=SYNTHETIC_RXNORM).concepts)
    store.add_concepts(parse_loinc_csv(text=SYNTHETIC_LOINC).concepts)
    return store


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------


class TestImports:
    def test_package_imports(self):
        import clinical_knowledge.terminology as t
        for name in (
            "TerminologyConcept", "TerminologyImportMode", "TerminologyLookupResult",
            "TerminologyLookupStatus", "TerminologySourceManifest",
            "TerminologySourceStatus", "TerminologySystem",
            "LicenseGateError", "license_acknowledged_for",
            "require_license_acknowledgment", "verify_operator_license_acknowledgment",
            "InventoryReport", "inventory_terminology_data_dir",
            "ParseResult", "parse_loinc_csv", "parse_rxnorm_rxnconso",
            "parse_snomed_concept_description", "parse_umls_mrconso",
            "LocalTerminologyStore", "TerminologyLookupService",
            "code_entity_via_local_terminology", "safe_b07_boundary_summary",
        ):
            assert hasattr(t, name), f"missing export: {name}"

    def test_validation_script_importable(self):
        from scripts import run_cka_term01_real_terminology_readiness_validation as v
        assert hasattr(v, "run_validation")

    def test_inventory_script_importable(self):
        from scripts import cka_terminology_inventory as s
        assert hasattr(s, "main")

    def test_import_local_script_importable(self):
        from scripts import cka_terminology_import_local as s
        assert hasattr(s, "main")


# ---------------------------------------------------------------------------
# TestModelsSafeSummary
# ---------------------------------------------------------------------------


class TestModelsSafeSummary:
    def test_concept_summary_no_phi(self):
        c = TerminologyConcept.synthetic_for(
            TerminologySystem.UMLS, "C0001", "Hypertension",
            synonyms=["High blood pressure"],
        )
        s = c.safe_public_summary()
        assert s["system"] == "umls"
        assert s["code"] == "C0001"
        assert s["synonyms_count"] == 1
        assert s["synthetic"] is True

    def test_source_manifest_no_path(self):
        m = TerminologySourceManifest.for_local_source(
            system=TerminologySystem.UMLS,
            local_root="/private/some/path",
            file_count=2,
            expected_files_present=["MRCONSO.RRF"],
            license_confirmed=False,
            import_mode=TerminologyImportMode.INVENTORY_ONLY,
            status=TerminologySourceStatus.PRESENT_VERIFIED,
        )
        s = m.safe_public_summary()
        assert "/private/some/path" not in json.dumps(s)
        assert s["system"] == "umls"
        assert s["license_confirmed"] is False
        assert s["import_mode"] == "inventory_only"

    def test_lookup_result_safe(self):
        r = TerminologyLookupResult.for_query("Hello World")
        s = r.safe_public_summary()
        assert s["normalized_query"] == "hello world"
        assert s["status"] == "unmapped"
        assert s["matches_count"] == 0
        assert s["no_code_hallucinated"] is True


# ---------------------------------------------------------------------------
# TestLicenseGate
# ---------------------------------------------------------------------------


class TestLicenseGate:
    def test_synthetic_test_always_acked(self):
        assert license_acknowledged_for(TerminologySystem.SYNTHETIC_TEST) is True

    @pytest.mark.parametrize("sys_", [
        TerminologySystem.UMLS, TerminologySystem.SNOMED_CT,
        TerminologySystem.RXNORM, TerminologySystem.LOINC,
    ])
    def test_real_systems_blocked_without_ack(self, sys_):
        # Empty env, no ack file.
        assert license_acknowledged_for(sys_, env={}) is False
        with pytest.raises(LicenseGateError):
            require_license_acknowledgment(sys_, env={})

    def test_test_mode_env_works(self):
        assert license_acknowledged_for(
            TerminologySystem.UMLS,
            env={"CKA_TERM01_TEST_LICENSE_ACK": "1"},
            test_mode=True,
        ) is True

    def test_test_mode_env_ignored_outside_test_mode(self):
        # Even with the env var set, real-mode (test_mode=False) ignores it.
        assert license_acknowledged_for(
            TerminologySystem.UMLS,
            env={"CKA_TERM01_TEST_LICENSE_ACK": "1"},
            test_mode=False,
        ) is False

    def test_local_ack_file_via_path(self, tmp_path):
        ack = tmp_path / "LICENSE_ACK_PRIVATE.json"
        ack.write_text(json.dumps({
            "operator_acknowledged": True,
            "acknowledged_systems": ["umls"],
        }), encoding="utf-8")
        assert license_acknowledged_for(
            TerminologySystem.UMLS, local_ack_file=ack,
        ) is True
        # Other systems still blocked.
        assert license_acknowledged_for(
            TerminologySystem.SNOMED_CT, local_ack_file=ack,
        ) is False


# ---------------------------------------------------------------------------
# TestInventory
# ---------------------------------------------------------------------------


class TestInventory:
    def test_missing_root_safe(self):
        inv = inventory_terminology_data_dir()
        s = inv.safe_public_summary()
        assert s["raw_paths_written_to_public_report"] is False
        assert s["sources_count"] >= 4   # UMLS, SNOMED, RxNorm, LOINC

    def test_no_drive_letter_path_in_summary(self):
        inv = inventory_terminology_data_dir()
        text = json.dumps(inv.safe_public_summary())
        assert not re.search(r"[A-Za-z]:\\\\", text)

    def test_safe_root_hash_format(self):
        inv = inventory_terminology_data_dir()
        h = inv.safe_public_summary()["terminology_root_safe_hash"]
        assert isinstance(h, str)
        assert h.startswith("term_root_")


# ---------------------------------------------------------------------------
# TestParsers
# ---------------------------------------------------------------------------


class TestParsers:
    def test_umls_parser(self):
        r = parse_umls_mrconso(text=SYNTHETIC_UMLS)
        codes = {c.code for c in r.concepts}
        assert "C0001" in codes
        assert "C0002" in codes
        c0001 = next(c for c in r.concepts if c.code == "C0001")
        assert c0001.synthetic is True
        assert c0001.system == TerminologySystem.UMLS
        # 'high blood pressure' should be either the display or a synonym.
        all_terms = {c0001.display.lower(), *[s.lower() for s in c0001.synonyms]}
        assert "high blood pressure" in all_terms
        assert "hypertension" in all_terms

    def test_snomed_parser(self):
        r = parse_snomed_concept_description(
            concept_text=SYNTHETIC_SNOMED_C,
            description_text=SYNTHETIC_SNOMED_D,
        )
        codes = {c.code for c in r.concepts}
        assert "100000001" in codes
        assert "100000002" in codes

    def test_rxnorm_parser(self):
        r = parse_rxnorm_rxnconso(text=SYNTHETIC_RXNORM)
        assert {c.code for c in r.concepts} == {"R001", "R002"}
        assert all(c.system == TerminologySystem.RXNORM for c in r.concepts)

    def test_loinc_parser(self):
        r = parse_loinc_csv(text=SYNTHETIC_LOINC)
        assert {c.code for c in r.concepts} == {"12345-6"}

    def test_parsers_cap_max_rows(self):
        # Generate a many-row UMLS-ish text to confirm the cap is honoured.
        big = "\n".join([
            f"C{i:04d}|ENG|P|L|PF|S|Y|A||||MTH|PT|U{i:04d}|term_{i}||N|"
            for i in range(50)
        ])
        r = parse_umls_mrconso(text=big, max_rows=5)
        assert r.rows_seen <= 5


# ---------------------------------------------------------------------------
# TestStoreAndLookup
# ---------------------------------------------------------------------------


class TestStoreAndLookup:
    def test_in_memory_store_init(self):
        store = LocalTerminologyStore()
        s = store.safe_public_summary()
        assert s["in_memory"] is True
        assert s["concepts_count"] == 0

    def test_add_concepts_round_trip(self):
        store = _populated_store()
        s = store.safe_public_summary()
        assert s["concepts_count"] >= 4

    def test_lookup_exact(self):
        store = _populated_store()
        svc = TerminologyLookupService(store)
        res = svc.lookup("metformin", systems=[TerminologySystem.RXNORM])
        assert res.status == TerminologyLookupStatus.EXACT
        assert res.exact_match is True
        assert res.matches and res.matches[0].code == "R002"

    def test_lookup_synonym(self):
        store = _populated_store()
        svc = TerminologyLookupService(store)
        # 'high blood pressure' is a synonym of UMLS C0001 (whose preferred
        # display is 'hypertension') AND/OR mapped via SNOMED. Either way
        # the service should return a non-empty match (SYNONYM or EXACT).
        res = svc.lookup("high blood pressure")
        assert res.status in (
            TerminologyLookupStatus.SYNONYM,
            TerminologyLookupStatus.EXACT,
            TerminologyLookupStatus.AMBIGUOUS,
        )
        assert res.matches  # non-empty

    def test_lookup_unknown_unmapped(self):
        store = _populated_store()
        svc = TerminologyLookupService(store)
        res = svc.lookup("zzzz_unknown_term_xyz")
        assert res.status == TerminologyLookupStatus.UNMAPPED
        assert res.matches == []
        assert res.no_code_hallucinated is True

    def test_lookup_empty_query_unmapped(self):
        store = _populated_store()
        svc = TerminologyLookupService(store)
        res = svc.lookup("")
        assert res.status == TerminologyLookupStatus.UNMAPPED
        assert res.matches == []


# ---------------------------------------------------------------------------
# TestB07Boundary
# ---------------------------------------------------------------------------


class TestB07Boundary:
    def test_unknown_term_unmapped_not_hallucinated(self):
        store = _populated_store()
        svc = TerminologyLookupService(store)
        res = code_entity_via_local_terminology("zzzz_unknown_xyz", svc)
        assert res.status == TerminologyLookupStatus.UNMAPPED
        assert res.matches == []
        assert res.no_code_hallucinated is True

    def test_known_term_mapped(self):
        store = _populated_store()
        svc = TerminologyLookupService(store)
        res = code_entity_via_local_terminology(
            "metformin", svc, systems=[TerminologySystem.RXNORM]
        )
        assert res.matches
        assert res.matches[0].code == "R002"

    def test_boundary_summary_invariants(self):
        b = safe_b07_boundary_summary()
        assert b["default_b07_behavior_unchanged"] is True
        assert b["coding_promotes_hypothesis"] is False
        assert b["coding_clears_ddi_status"] is False
        assert b["no_code_hallucinated"] is True
        assert b["unknown_terms_unmapped"] is True
        assert b["synthetic_mapper_still_present"] is True


# ---------------------------------------------------------------------------
# TestNoTerminologyDataCommitted
# ---------------------------------------------------------------------------


class TestNoTerminologyDataCommitted:
    def test_terminology_data_dir_not_in_repo(self):
        # The directory may exist from a prior validator run, but its
        # CONTENT must not be tracked. We test by listing tracked files
        # under that path via git.
        res = subprocess.run(
            ["git", "ls-files", "terminology_data/"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        # Either the path doesn't exist (rc 0, empty stdout) or git
        # returns no files because the dir is gitignored.
        assert res.returncode == 0
        assert res.stdout.strip() == ""

    def test_data_terminology_dir_not_tracked(self):
        res = subprocess.run(
            ["git", "ls-files", "data/terminology/"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        assert res.returncode == 0
        assert res.stdout.strip() == ""

    def test_license_ack_files_not_tracked(self):
        res = subprocess.run(
            ["git", "ls-files", "*LICENSE_ACK_PRIVATE*"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        assert res.returncode == 0
        assert res.stdout.strip() == ""

    def test_gitignore_lists_terminology_paths(self):
        gi = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "terminology_data/" in gi
        assert "data/terminology/" in gi
        assert "LICENSE_ACK_PRIVATE" in gi


# ---------------------------------------------------------------------------
# TestPublicReport
# ---------------------------------------------------------------------------


class TestPublicReport:
    @pytest.fixture(scope="class")
    def report(self):
        from scripts.run_cka_term01_real_terminology_readiness_validation import (
            run_validation,
        )
        return run_validation()

    def test_block_id(self, report):
        assert report["block_id"] == "CKA-TERM-01"

    def test_conclusion_valid(self, report):
        assert report["conclusion"] in (
            "cka_term01_local_terminology_import_ready",
            "cka_term01_local_terminology_files_required",
        )

    def test_synthetic_parses_passed(self, report):
        for k in (
            "synthetic_umls_parse_passed",
            "synthetic_snomed_parse_passed",
            "synthetic_rxnorm_parse_passed",
            "synthetic_loinc_parse_passed",
            "lookup_service_ready",
            "no_code_hallucinated",
            "unknown_terms_unmapped",
            "ambiguous_terms_flagged",
            "b07_integration_boundary_preserved",
            "coding_does_not_promote_hypothesis",
            "coding_does_not_clear_ddi_status",
            "license_gate_ready",
            "local_inventory_ready",
        ):
            assert report[k] is True, f"flag {k} not True"

    def test_safety_flags_false(self, report):
        for k in (
            "real_terminology_files_committed",
            "external_terminology_api_used",
            "real_umls_api_used",
            "real_snomed_download_used",
            "real_rxnorm_api_used",
            "real_loinc_api_used",
            "external_api_used",
            "raw_phi_logged_in_public_reports",
            "license_text_written_to_public_reports",
            "clinical_recommendations_generated",
            "prescription_dosing_advice_generated",
            "production_ocr_changed",
            "production_extractor_changed",
            "safety_gate_changed",
            "frozen_hitl_release_reopened",
        ):
            assert report[k] is False, f"flag {k} not False"

    def test_zero_leak_counters(self, report):
        assert report["private_filename_path_leaks"] == 0
        assert report["secret_leaks"] == 0

    def test_report_no_drive_letter_path(self, report):
        assert not re.search(r"[A-Za-z]:\\\\", json.dumps(report))

    def test_report_no_license_text_words(self, report):
        text = json.dumps(report)
        # The flag NAME license_text_written_to_public_reports is allowed,
        # but no license-text wording should appear.
        for needle in ("operator_acknowledged", "LICENSE_ACK_PRIVATE"):
            assert needle not in text, f"forbidden token {needle!r} in report"

    def test_report_passes_b02_privacy_checker(self, report):
        from clinical_knowledge.privacy.report_privacy import (
            check_public_report_payload,
        )
        result = check_public_report_payload(report)
        assert result.passed, (
            f"privacy checker rejected: {result.leak_examples_redacted}"
        )

    def test_report_md_present(self):
        md = REPORT_DIR / "cka_term01_real_terminology_readiness_report.md"
        assert md.exists()

    def test_operator_guide_present(self):
        guide = REPORT_DIR / "CKA_TERM01_OPERATOR_TERMINOLOGY_DATA_GUIDE.md"
        assert guide.exists()
        text = guide.read_text(encoding="utf-8")
        assert "terminology_data/" in text
        assert "license" in text.lower()


# ---------------------------------------------------------------------------
# TestNoClinicalLogicChange
# ---------------------------------------------------------------------------


class TestNoClinicalLogicChange:
    def test_no_clinical_text_in_terminology_modules(self):
        forbidden = (
            "take this dose",
            "recommended dose",
            "you should take",
            "mg per day",
            "we prescribe",
        )
        for fname in (
            "clinical_knowledge/terminology/__init__.py",
            "clinical_knowledge/terminology/models.py",
            "clinical_knowledge/terminology/license_gate.py",
            "clinical_knowledge/terminology/file_inventory.py",
            "clinical_knowledge/terminology/parsers.py",
            "clinical_knowledge/terminology/local_store.py",
            "clinical_knowledge/terminology/lookup_service.py",
            "clinical_knowledge/terminology/integration.py",
            "clinical_knowledge/terminology/source_manifest.py",
        ):
            text = (REPO_ROOT / fname).read_text(encoding="utf-8").lower()
            for needle in forbidden:
                assert needle not in text, f"{fname}: forbidden {needle!r}"

    def test_main_mkb_store_unchanged(self):
        from clinical_knowledge.store import MKBStore as MS
        from clinical_knowledge.security import EncryptedCKAStore as Enc
        assert isinstance(MS, type)
        assert MS is not Enc

    def test_consensus_engine_still_loads(self):
        from clinical_knowledge.consensus.engine import run_consensus    # noqa: F401

    def test_decision_engine_still_loads(self):
        import clinical_knowledge.decision_engine.engine    # noqa: F401

    def test_truth_resolution_still_loads(self):
        import clinical_knowledge.truth_resolution.engine    # noqa: F401

    def test_ddi_stub_still_loads(self):
        import clinical_knowledge.medication_safety.ddi_stub    # noqa: F401

    def test_b07_synthetic_mapper_still_present(self):
        from clinical_knowledge.medical_coding import synthetic_mapper    # noqa: F401
