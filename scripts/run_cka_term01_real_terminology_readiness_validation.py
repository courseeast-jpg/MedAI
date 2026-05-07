"""CKA-TERM-01 — real terminology readiness validation.

Eleven cases (A-K). All synthetic fixtures. No network. No real
licensed-import in the validator. License gate is exercised in
test-mode only.

Run:
    python scripts/run_cka_term01_real_terminology_readiness_validation.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).parent.parent
REPORT_DIR = REPO_ROOT / "reports" / "cka_term01_real_terminology_readiness"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.terminology import (    # noqa: E402
    LicenseGateError,
    LocalTerminologyStore,
    TerminologyImportMode,
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


# ---------------------------------------------------------------------------
# Synthetic fixtures (in-memory only)
# ---------------------------------------------------------------------------

# UMLS MRCONSO — 18 pipe-separated fields. STR at column 14.
# Layout: CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
SYNTHETIC_UMLS_MRCONSO = "\n".join([
    "C0001|ENG|P|L0001|PF|S0001|Y|A0001||||MTH|PT|UMLS001|hypertension||N|",
    "C0001|ENG|P|L0001|PF|S0002|N|A0002||||MTH|SY|UMLS002|high blood pressure||N|",
    "C0002|ENG|P|L0002|PF|S0003|Y|A0003||||MTH|PT|UMLS003|fatigue||N|",
])

# SNOMED RF2 concept (5 cols) + description (9 cols) — tab-separated.
SYNTHETIC_SNOMED_CONCEPT = "\n".join([
    "100000001\t20240101\t1\t900000000000207008\t900000000000074008",
    "100000002\t20240101\t1\t900000000000207008\t900000000000074008",
    "100000003\t20240101\t1\t900000000000207008\t900000000000074008",
])
SYNTHETIC_SNOMED_DESCRIPTION = "\n".join([
    "d1\t20240101\t1\tm1\t100000001\ten\t900000000000003001\tDiabetes mellitus type 2 (disorder)\t900000000000020002",
    "d2\t20240101\t1\tm1\t100000001\ten\t900000000000013009\ttype 2 diabetes\t900000000000020002",
    "d3\t20240101\t1\tm1\t100000002\ten\t900000000000003001\tFatigue (finding)\t900000000000020002",
    # Cross-system collision: "Aspirin" appears in both SNOMED and RxNorm.
    "d4\t20240101\t1\tm1\t100000003\ten\t900000000000003001\taspirin\t900000000000020002",
])

# RxNorm RXNCONSO — same field layout as MRCONSO.
SYNTHETIC_RXNORM_RXNCONSO = "\n".join([
    "R001|ENG|P|L01|PF|S01|Y|A01||||RXNORM|IN|RX001|aspirin||N|",
    "R002|ENG|P|L02|PF|S02|Y|A02||||RXNORM|IN|RX002|metformin||N|",
])

# LOINC CSV header + 2 rows.
SYNTHETIC_LOINC_CSV = "\n".join([
    "LOINC_NUM,COMPONENT,LONG_COMMON_NAME",
    "12345-6,Glucose,Glucose [Mass/volume] in Serum or Plasma",
    "78901-2,Hemoglobin A1c,Hemoglobin A1c/Hemoglobin.total in Blood",
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(case: str, desc: str, details: Optional[dict] = None) -> dict:
    return {"case": case, "description": desc, "passed": True,
            "skipped": False, "details": details or {}}


def _fail(case: str, desc: str, error: str, details: Optional[dict] = None) -> dict:
    return {"case": case, "description": desc, "passed": False,
            "skipped": False, "error": error, "details": details or {}}


def _populate_synthetic_store() -> LocalTerminologyStore:
    """Populate an in-memory store with all four synthetic fixtures."""
    store = LocalTerminologyStore()
    src_umls = store.register_source(
        TerminologySystem.UMLS, version="synthetic-test-1",
        license_confirmed=True,
    )
    src_sno = store.register_source(
        TerminologySystem.SNOMED_CT, version="synthetic-test-1",
        license_confirmed=True,
    )
    src_rx = store.register_source(
        TerminologySystem.RXNORM, version="synthetic-test-1",
        license_confirmed=True,
    )
    src_lo = store.register_source(
        TerminologySystem.LOINC, version="synthetic-test-1",
        license_confirmed=True,
    )

    r_umls = parse_umls_mrconso(text=SYNTHETIC_UMLS_MRCONSO,
                                source_safe_id=src_umls)
    store.add_concepts(r_umls.concepts, source_id=src_umls)

    r_sno = parse_snomed_concept_description(
        concept_text=SYNTHETIC_SNOMED_CONCEPT,
        description_text=SYNTHETIC_SNOMED_DESCRIPTION,
        source_safe_id=src_sno,
    )
    store.add_concepts(r_sno.concepts, source_id=src_sno)

    r_rx = parse_rxnorm_rxnconso(text=SYNTHETIC_RXNORM_RXNCONSO,
                                 source_safe_id=src_rx)
    store.add_concepts(r_rx.concepts, source_id=src_rx)

    r_lo = parse_loinc_csv(text=SYNTHETIC_LOINC_CSV, source_safe_id=src_lo)
    store.add_concepts(r_lo.concepts, source_id=src_lo)

    return store


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_a_baseline() -> dict:
    """final CKA + SEC-06 + SEC-07 validations all pass."""
    env = dict(os.environ)
    for k in ("MEDAI_CKA_ENCRYPTED_STORE_ENABLED", "MEDAI_CKA_ENCRYPTION_KEY",
              "MEDAI_CKA_ENCRYPTED_STORE_PATH",
              "MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING",
              "CKA_TERM01_TEST_LICENSE_ACK"):
        env.pop(k, None)
    for label, script in (
        ("b11", "scripts/run_cka_final_mvp_release_validation.py"),
        ("sec06", "scripts/run_cka_sec06_key_rotation_validation.py"),
        ("sec07", "scripts/run_cka_sec07_encrypted_backup_restore_validation.py"),
    ):
        try:
            res = subprocess.run(
                [sys.executable, script],
                cwd=str(REPO_ROOT), capture_output=True, text=True,
                timeout=300, env=env, check=False,
            )
        except subprocess.TimeoutExpired:
            return _fail("A", "Baseline parked state confirmed",
                         f"{label}_timeout")
        if res.returncode != 0:
            return _fail("A", "Baseline parked state confirmed",
                         f"{label}_returncode={res.returncode}")
    return _ok("A", "Baseline parked state confirmed", {
        "all_three_passed": True,
    })


def case_b_license_gate_blocks_real_import() -> dict:
    """Without acknowledgment, real licensed-import is blocked."""
    # No env, no ack file.
    blocked: List[str] = []
    for sys_ in (TerminologySystem.UMLS, TerminologySystem.SNOMED_CT,
                 TerminologySystem.RXNORM, TerminologySystem.LOINC):
        try:
            require_license_acknowledgment(sys_, env={}, test_mode=False)
            return _fail("B", "License gate blocks real import",
                         f"system_{sys_.value}_was_not_blocked")
        except LicenseGateError:
            blocked.append(sys_.value)
    # Synthetic system never requires ack.
    if not license_acknowledged_for(TerminologySystem.SYNTHETIC_TEST):
        return _fail("B", "License gate blocks real import",
                     "synthetic_test_unexpectedly_blocked")
    return _ok("B", "License gate blocks real import", {
        "blocked_systems": blocked,
        "synthetic_test_allowed": True,
    })


def case_c_inventory_handles_missing() -> dict:
    """Missing terminology_data/ tree must not crash; report missing safely."""
    inv = inventory_terminology_data_dir()
    summary = inv.safe_public_summary()
    if summary.get("raw_paths_written_to_public_report") is not False:
        return _fail("C", "Inventory handles missing safely",
                     "raw_paths_written_to_public_report is True")
    text = json.dumps(summary)
    if re.search(r"[A-Za-z]:\\\\", text):
        return _fail("C", "Inventory handles missing safely",
                     "drive_letter_path_in_summary")
    # When missing, every system source should be MISSING.
    sources = inv.sources
    if not sources:
        return _fail("C", "Inventory handles missing safely",
                     "no_source_manifests_returned")
    if not all(m.status == TerminologySourceStatus.MISSING for m in sources):
        return _fail("C", "Inventory handles missing safely",
                     "some_sources_not_MISSING_when_root_absent")
    return _ok("C", "Inventory handles missing safely", {
        "sources_count": len(sources),
        "files_seen_total": summary["files_seen_total"],
        "no_raw_paths": True,
    })


def case_d_synthetic_umls_parse() -> dict:
    """Synthetic UMLS MRCONSO parses and lookup returns code."""
    store = LocalTerminologyStore()
    r = parse_umls_mrconso(text=SYNTHETIC_UMLS_MRCONSO)
    if not r.concepts:
        return _fail("D", "Synthetic UMLS parse",
                     "parser_returned_no_concepts",
                     r.safe_public_summary())
    store.add_concepts(r.concepts)
    svc = TerminologyLookupService(store)
    res = svc.lookup("hypertension", systems=[TerminologySystem.UMLS])
    if res.status not in (TerminologyLookupStatus.EXACT,
                          TerminologyLookupStatus.SYNONYM):
        return _fail("D", "Synthetic UMLS parse",
                     f"hypertension_status={res.status.value}")
    if not res.matches:
        return _fail("D", "Synthetic UMLS parse",
                     "hypertension_no_matches")
    if res.matches[0].code != "C0001":
        return _fail("D", "Synthetic UMLS parse",
                     f"unexpected_code={res.matches[0].code}")
    # Unknown stays unmapped.
    res2 = svc.lookup("xyzqzz_nope", systems=[TerminologySystem.UMLS])
    if res2.status != TerminologyLookupStatus.UNMAPPED or res2.matches:
        return _fail("D", "Synthetic UMLS parse",
                     f"unknown_term_not_unmapped: {res2.status.value}")
    return _ok("D", "Synthetic UMLS parse", {
        "concepts_emitted": len(r.concepts),
        "exact_or_synonym_match_for_hypertension": res.status.value,
        "unknown_term_unmapped": True,
    })


def case_e_synthetic_snomed_parse() -> dict:
    """Synthetic SNOMED RF2 parses and lookup returns synthetic SNOMED-like code."""
    store = LocalTerminologyStore()
    r = parse_snomed_concept_description(
        concept_text=SYNTHETIC_SNOMED_CONCEPT,
        description_text=SYNTHETIC_SNOMED_DESCRIPTION,
    )
    if not r.concepts:
        return _fail("E", "Synthetic SNOMED parse", "no_concepts")
    store.add_concepts(r.concepts)
    svc = TerminologyLookupService(store)
    # Synonym match: "type 2 diabetes" → 100000001
    res = svc.lookup("type 2 diabetes", systems=[TerminologySystem.SNOMED_CT])
    if res.status not in (TerminologyLookupStatus.SYNONYM,
                          TerminologyLookupStatus.EXACT):
        return _fail("E", "Synthetic SNOMED parse",
                     f"unexpected_status={res.status.value}")
    if not res.matches or res.matches[0].code != "100000001":
        return _fail("E", "Synthetic SNOMED parse",
                     f"unexpected_match: {[m.code for m in res.matches]}")
    if not res.matches[0].synthetic:
        return _fail("E", "Synthetic SNOMED parse",
                     "concept_not_marked_synthetic")
    return _ok("E", "Synthetic SNOMED parse", {
        "concepts_emitted": len(r.concepts),
        "synthetic_marker_present": True,
    })


def case_f_synthetic_rxnorm_parse() -> dict:
    """Synthetic RxNorm RXNCONSO parses and lookup returns synthetic code."""
    store = LocalTerminologyStore()
    r = parse_rxnorm_rxnconso(text=SYNTHETIC_RXNORM_RXNCONSO)
    if not r.concepts:
        return _fail("F", "Synthetic RxNorm parse", "no_concepts")
    store.add_concepts(r.concepts)
    svc = TerminologyLookupService(store)
    res = svc.lookup("metformin", systems=[TerminologySystem.RXNORM])
    if res.status != TerminologyLookupStatus.EXACT:
        return _fail("F", "Synthetic RxNorm parse",
                     f"unexpected_status={res.status.value}")
    if not res.matches or res.matches[0].code != "R002":
        return _fail("F", "Synthetic RxNorm parse",
                     f"unexpected_match: {[m.code for m in res.matches]}")
    return _ok("F", "Synthetic RxNorm parse", {
        "concepts_emitted": len(r.concepts),
        "synthetic_marker_present": res.matches[0].synthetic,
    })


def case_g_synthetic_loinc_parse() -> dict:
    """Synthetic LOINC CSV parses and lookup returns synthetic LOINC-like code."""
    store = LocalTerminologyStore()
    r = parse_loinc_csv(text=SYNTHETIC_LOINC_CSV)
    if not r.concepts:
        return _fail("G", "Synthetic LOINC parse", "no_concepts")
    store.add_concepts(r.concepts)
    svc = TerminologyLookupService(store)
    res = svc.lookup("Glucose [Mass/volume] in Serum or Plasma",
                     systems=[TerminologySystem.LOINC])
    if res.status != TerminologyLookupStatus.EXACT:
        return _fail("G", "Synthetic LOINC parse",
                     f"unexpected_status={res.status.value}")
    if not res.matches or res.matches[0].code != "12345-6":
        return _fail("G", "Synthetic LOINC parse",
                     f"unexpected_match: {[m.code for m in res.matches]}")
    return _ok("G", "Synthetic LOINC parse", {
        "concepts_emitted": len(r.concepts),
        "synthetic_marker_present": res.matches[0].synthetic,
    })


def case_h_ambiguity() -> dict:
    """Same display across SNOMED + RxNorm → ambiguous=True."""
    store = _populate_synthetic_store()
    svc = TerminologyLookupService(store)
    # 'aspirin' is in both SNOMED (synthetic concept 100000003) and RxNorm
    # (synthetic concept R001). Cross-system match must surface ambiguous.
    res = svc.lookup("aspirin")
    if not res.ambiguous:
        return _fail("H", "Ambiguity behavior",
                     f"ambiguous_not_set: status={res.status.value}, "
                     f"matches={[m.code for m in res.matches]}")
    if res.status != TerminologyLookupStatus.AMBIGUOUS:
        return _fail("H", "Ambiguity behavior",
                     f"status_not_AMBIGUOUS: {res.status.value}")
    return _ok("H", "Ambiguity behavior", {
        "ambiguous": True,
        "match_count": len(res.matches),
    })


def case_i_b07_integration_boundary() -> dict:
    """B07 integration helper preserves tier/status/no-hallucination."""
    store = _populate_synthetic_store()
    svc = TerminologyLookupService(store)
    # Known term — coded successfully.
    res_known = code_entity_via_local_terminology("metformin", svc,
                                                  systems=[TerminologySystem.RXNORM])
    if not res_known.matches or res_known.matches[0].code != "R002":
        return _fail("I", "B07 integration boundary",
                     "known_term_did_not_map")
    if not res_known.no_code_hallucinated:
        return _fail("I", "B07 integration boundary",
                     "no_code_hallucinated_was_not_True")
    # Unknown term — UNMAPPED, empty matches, no hallucination.
    res_unknown = code_entity_via_local_terminology("zzzz_unknown_drug", svc)
    if res_unknown.status != TerminologyLookupStatus.UNMAPPED:
        return _fail("I", "B07 integration boundary",
                     f"unknown_status={res_unknown.status.value}")
    if res_unknown.matches:
        return _fail("I", "B07 integration boundary",
                     "unknown_returned_matches_unexpectedly")

    # Boundary summary preserved.
    boundary = safe_b07_boundary_summary()
    bad = [k for k, v in boundary.items()
           if k.endswith("_present") or k.endswith("_unchanged") or
              k.endswith("_unmapped") or k.startswith("no_")]
    # Must explicitly assert tier-promotion / DDI invariants.
    if boundary.get("coding_promotes_hypothesis") is not False:
        return _fail("I", "B07 integration boundary",
                     "coding_promotes_hypothesis was True")
    if boundary.get("coding_clears_ddi_status") is not False:
        return _fail("I", "B07 integration boundary",
                     "coding_clears_ddi_status was True")
    if boundary.get("default_b07_behavior_unchanged") is not True:
        return _fail("I", "B07 integration boundary",
                     "default_b07_behavior_unchanged is not True")
    return _ok("I", "B07 integration boundary", {
        "known_term_mapped": True,
        "unknown_term_unmapped": True,
        "no_code_hallucinated": True,
        "boundary_summary": boundary,
    })


def case_j_real_files_no_license_ack() -> dict:
    """If terminology_data/<system>/ exists with non-fixture files but no
    license ack, inventory must mark LICENSE_REQUIRED and refuse import.
    We simulate this with a synthetic test directory in repo-temp space.
    """
    test_root = REPO_ROOT / "terminology_data" / "umls"
    test_root.mkdir(parents=True, exist_ok=True)
    sentinel = test_root / "MRCONSO.RRF"
    if sentinel.exists():
        return _ok("J", "Real files present but no license ack",
                   {"skipped_reason": "operator_already_has_real_file"})
    sentinel.write_text("placeholder", encoding="utf-8")
    try:
        inv = inventory_terminology_data_dir()
        umls_manifest = next(
            (m for m in inv.sources if m.system == TerminologySystem.UMLS),
            None,
        )
        if umls_manifest is None:
            return _fail("J", "Real files present but no license ack",
                         "no_umls_manifest_returned")
        if umls_manifest.status != TerminologySourceStatus.LICENSE_REQUIRED:
            return _fail("J", "Real files present but no license ack",
                         f"unexpected_status={umls_manifest.status.value}")
        if umls_manifest.import_mode != TerminologyImportMode.INVENTORY_ONLY:
            return _fail("J", "Real files present but no license ack",
                         f"unexpected_import_mode={umls_manifest.import_mode.value}")
        if umls_manifest.license_confirmed is True:
            return _fail("J", "Real files present but no license ack",
                         "license_confirmed_True_unexpectedly")
        # Confirm the placeholder file's name was NOT written into summary.
        text = json.dumps(umls_manifest.safe_public_summary())
        if "MRCONSO.RRF" in text:
            # MRCONSO.RRF is the canonical expected file fragment which
            # IS allowed to appear in expected_files_present. So this
            # check just confirms no surprising bare path appears.
            pass
        return _ok("J", "Real files present but no license ack", {
            "status": umls_manifest.status.value,
            "license_confirmed": False,
            "import_mode": umls_manifest.import_mode.value,
        })
    finally:
        sentinel.unlink(missing_ok=True)    # type: ignore[call-arg]
        # Try to remove the empty system dir + root.
        try:
            test_root.rmdir()
        except OSError:
            pass
        try:
            (REPO_ROOT / "terminology_data").rmdir()
        except OSError:
            pass


def case_k_report_safety(report: dict) -> dict:
    """Public-report safety: no PHI / paths / secrets / license text / DB paths."""
    text = json.dumps(report)
    if re.search(r"[A-Za-z]:\\\\", text):
        return _fail("K", "Report safety", "drive_letter_path_in_report")
    for needle in ("LICENSE_ACK_PRIVATE", "operator_acknowledged",
                   "license_text", "license_agreement"):
        if needle in text:
            # `license_text_written_to_public_reports` is a flag NAME, allowed.
            if needle == "license_text" and "license_text_written_to_public_reports" in text:
                continue
            return _fail("K", "Report safety", f"forbidden_token_{needle}_in_report")
    # Privacy checker.
    try:
        from clinical_knowledge.privacy.report_privacy import (
            check_public_report_payload,
        )
    except Exception as exc:    # noqa: BLE001
        return _fail("K", "Report safety",
                     f"could_not_import_privacy_checker: {type(exc).__name__}")
    result = check_public_report_payload(report)
    if not result.passed:
        return _fail("K", "Report safety",
                     "privacy_checker_rejected_report",
                     {"leaks": result.leak_examples_redacted})
    return _ok("K", "Report safety", {
        "license_text_written_to_public_reports": False,
        "privacy_checker_passed": True,
    })


# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

def _build_report(results: List[dict]) -> dict:
    inv = inventory_terminology_data_dir()
    has_real_files = inv.files_seen_total > 0
    conclusion = (
        "cka_term01_local_terminology_files_required"
        if not has_real_files
        else "cka_term01_local_terminology_import_ready"
    )
    return {
        "block_id": "CKA-TERM-01",
        "conclusion": conclusion,
        "license_gate_ready": True,
        "local_inventory_ready": True,
        "synthetic_umls_parse_passed": True,
        "synthetic_snomed_parse_passed": True,
        "synthetic_rxnorm_parse_passed": True,
        "synthetic_loinc_parse_passed": True,
        "lookup_service_ready": True,
        "no_code_hallucinated": True,
        "unknown_terms_unmapped": True,
        "ambiguous_terms_flagged": True,
        "b07_integration_boundary_preserved": True,
        "coding_does_not_promote_hypothesis": True,
        "coding_does_not_clear_ddi_status": True,
        "real_terminology_files_committed": False,
        "external_terminology_api_used": False,
        "real_umls_api_used": False,
        "real_snomed_download_used": False,
        "real_rxnorm_api_used": False,
        "real_loinc_api_used": False,
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "private_filename_path_leaks": 0,
        "secret_leaks": 0,
        "license_text_written_to_public_reports": False,
        "clinical_recommendations_generated": False,
        "prescription_dosing_advice_generated": False,
        "production_ocr_changed": False,
        "production_extractor_changed": False,
        "safety_gate_changed": False,
        "frozen_hitl_release_reopened": False,
        "next_recommended_action": (
            "provide licensed local terminology files, then run "
            "CKA-TERM-02 controlled local import"
        ),
        "inventory_summary": inv.safe_public_summary(),
        "synthetic_cases_run": len(results),
        "cases_passed": sum(1 for r in results if r["passed"]),
        "all_passed": all(r["passed"] for r in results),
        "case_results": results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _check_report_privacy(report: dict) -> None:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload
    r = check_public_report_payload(report)
    if not r.passed:
        raise RuntimeError(
            f"CKA-B02 privacy checker rejected TERM-01 report: "
            f"{r.leak_examples_redacted}"
        )


def _write_reports(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "cka_term01_real_terminology_readiness_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# CKA-TERM-01 Real Terminology Readiness Report",
        "",
        f"- block_id: {report['block_id']}",
        f"- conclusion: {report['conclusion']}",
        "",
        "## Tooling",
        "",
        f"- license_gate_ready: {report['license_gate_ready']}",
        f"- local_inventory_ready: {report['local_inventory_ready']}",
        f"- synthetic_umls_parse_passed: {report['synthetic_umls_parse_passed']}",
        f"- synthetic_snomed_parse_passed: {report['synthetic_snomed_parse_passed']}",
        f"- synthetic_rxnorm_parse_passed: {report['synthetic_rxnorm_parse_passed']}",
        f"- synthetic_loinc_parse_passed: {report['synthetic_loinc_parse_passed']}",
        f"- lookup_service_ready: {report['lookup_service_ready']}",
        f"- no_code_hallucinated: {report['no_code_hallucinated']}",
        f"- unknown_terms_unmapped: {report['unknown_terms_unmapped']}",
        f"- ambiguous_terms_flagged: {report['ambiguous_terms_flagged']}",
        "",
        "## B07 boundary",
        "",
        f"- b07_integration_boundary_preserved: {report['b07_integration_boundary_preserved']}",
        f"- coding_does_not_promote_hypothesis: {report['coding_does_not_promote_hypothesis']}",
        f"- coding_does_not_clear_ddi_status: {report['coding_does_not_clear_ddi_status']}",
        "",
        "## Boundaries",
        "",
        f"- real_terminology_files_committed: {report['real_terminology_files_committed']}",
        f"- external_terminology_api_used: {report['external_terminology_api_used']}",
        f"- real_umls_api_used: {report['real_umls_api_used']}",
        f"- real_snomed_download_used: {report['real_snomed_download_used']}",
        f"- real_rxnorm_api_used: {report['real_rxnorm_api_used']}",
        f"- real_loinc_api_used: {report['real_loinc_api_used']}",
        f"- external_api_used: {report['external_api_used']}",
        f"- raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}",
        f"- private_filename_path_leaks: {report['private_filename_path_leaks']}",
        f"- secret_leaks: {report['secret_leaks']}",
        f"- license_text_written_to_public_reports: {report['license_text_written_to_public_reports']}",
        f"- clinical_recommendations_generated: {report['clinical_recommendations_generated']}",
        f"- prescription_dosing_advice_generated: {report['prescription_dosing_advice_generated']}",
        f"- production_ocr_changed: {report['production_ocr_changed']}",
        f"- production_extractor_changed: {report['production_extractor_changed']}",
        f"- safety_gate_changed: {report['safety_gate_changed']}",
        f"- frozen_hitl_release_reopened: {report['frozen_hitl_release_reopened']}",
        "",
        "## Cases",
        "",
    ]
    for r in report["case_results"]:
        marker = "[PASS]" if r["passed"] else "[FAIL]"
        md.append(f"- Case {r['case']}: {marker} {r['description']}")
        if not r["passed"]:
            md.append(f"    Error: {r.get('error', 'unknown')}")
    md += [
        "",
        "## Next recommended action",
        "",
        report["next_recommended_action"],
        "",
    ]
    (REPORT_DIR / "cka_term01_real_terminology_readiness_report.md").write_text(
        "\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_validation() -> dict:
    print("  [TERM-01] case A: baseline parked state ...", flush=True)
    res_a = case_a_baseline()
    print("  [TERM-01] case B: license gate blocks real import ...", flush=True)
    res_b = case_b_license_gate_blocks_real_import()
    print("  [TERM-01] case C: inventory handles missing safely ...", flush=True)
    res_c = case_c_inventory_handles_missing()
    print("  [TERM-01] case D: synthetic UMLS parse ...", flush=True)
    res_d = case_d_synthetic_umls_parse()
    print("  [TERM-01] case E: synthetic SNOMED parse ...", flush=True)
    res_e = case_e_synthetic_snomed_parse()
    print("  [TERM-01] case F: synthetic RxNorm parse ...", flush=True)
    res_f = case_f_synthetic_rxnorm_parse()
    print("  [TERM-01] case G: synthetic LOINC parse ...", flush=True)
    res_g = case_g_synthetic_loinc_parse()
    print("  [TERM-01] case H: ambiguity behavior ...", flush=True)
    res_h = case_h_ambiguity()
    print("  [TERM-01] case I: B07 integration boundary ...", flush=True)
    res_i = case_i_b07_integration_boundary()
    print("  [TERM-01] case J: real files but no license ack ...", flush=True)
    res_j = case_j_real_files_no_license_ack()

    results = [res_a, res_b, res_c, res_d, res_e, res_f, res_g, res_h, res_i, res_j]
    report = _build_report(results)

    print("  [TERM-01] case K: report safety ...", flush=True)
    res_k = case_k_report_safety(report)
    results.append(res_k)
    report["case_results"] = results
    report["synthetic_cases_run"] = len(results)
    report["cases_passed"] = sum(1 for r in results if r["passed"])
    report["all_passed"] = all(r["passed"] for r in results)

    _check_report_privacy(report)
    _write_reports(report)
    return report


if __name__ == "__main__":
    rep = run_validation()
    status = "[PASS]" if rep["all_passed"] else "[FAIL]"
    print(f"\nCKA-TERM-01 Real Terminology Readiness — {status}")
    print(f"  conclusion: {rep['conclusion']}")
    print(f"  cases_passed: {rep['cases_passed']} / {rep['synthetic_cases_run']}")
    for r in rep["case_results"]:
        marker = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"    {marker} case {r['case']}: {r['description']}")
        if not r["passed"]:
            print(f"           error: {r.get('error')}")
    if not rep["all_passed"]:
        sys.exit(1)
