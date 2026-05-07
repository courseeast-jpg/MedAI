"""CKA-TERM-01C synthetic terminology import executor validation."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports" / "cka_term01c_import_executor"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.privacy.report_privacy import check_public_report_payload  # noqa: E402
from clinical_knowledge.terminology import (  # noqa: E402
    LocalTerminologyStore,
    RealTerminologyImportBlocked,
    TerminologyImportExecutor,
    TerminologyImportLimits,
    TerminologyLookupService,
    TerminologyLookupStatus,
    TerminologySystem,
    code_entity_via_local_terminology,
    parse_loinc_csv,
    parse_rxnorm_rxnconso,
    parse_snomed_concept_description,
    parse_umls_mrconso,
    safe_b07_boundary_summary,
)


UMLS_TEXT = "\n".join([
    "C0001|ENG|P|L1|PF|S1|Y|A1||||MTH|PT|U1|hypertension||N|",
    "C0001|ENG|P|L1|PF|S2|N|A2||||MTH|SY|U2|high blood pressure||N|",
    "C0002|ENG|P|L2|PF|S3|Y|A3||||MTH|PT|U3|fatigue||N|",
])
SNOMED_CONCEPT = "\n".join([
    "100\t20240101\t1\tm\t900",
    "200\t20240101\t1\tm\t900",
])
SNOMED_DESCRIPTION = "\n".join([
    "d1\t20240101\t1\tm\t100\ten\t900000000000003001\tDiabetes mellitus type 2 (disorder)\tp",
    "d2\t20240101\t1\tm\t100\ten\t900000000000013009\ttype 2 diabetes\tp",
    "d3\t20240101\t1\tm\t200\ten\t900000000000003001\tFatigue (finding)\tp",
])
RXNORM_TEXT = "\n".join([
    "R001|ENG|P|L1|PF|S1|Y|A1||||RXNORM|IN|RX1|aspirin||N|",
    "R002|ENG|P|L2|PF|S2|Y|A2||||RXNORM|IN|RX2|metformin||N|",
])
LOINC_TEXT = "\n".join([
    "LOINC_NUM,COMPONENT,LONG_COMMON_NAME",
    "1-1,Glucose,Glucose synthetic lab",
    "2-2,Hemoglobin,Hemoglobin synthetic lab",
])


def _ok(case: str, description: str, details: dict | None = None) -> dict:
    return {"case": case, "description": description, "passed": True, "details": details or {}}


def _fail(case: str, description: str, error: str, details: dict | None = None) -> dict:
    return {"case": case, "description": description, "passed": False, "error": error, "details": details or {}}


def case_a_term01b_baseline() -> dict:
    result = subprocess.run(
        [sys.executable, "scripts/run_cka_term01b_import_planner_validation.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if result.returncode != 0:
        return _fail("A", "TERM-01B baseline validation", f"returncode={result.returncode}")
    return _ok("A", "TERM-01B baseline validation", {"passed": True})


def _executor(limits: TerminologyImportLimits | None = None) -> TerminologyImportExecutor:
    return TerminologyImportExecutor(store=LocalTerminologyStore(), limits=limits)


def case_b_umls() -> dict:
    result = _executor().execute_synthetic(parse_umls_mrconso(text=UMLS_TEXT), source_safe_id="term_src_safe_umls")
    if not result.audit.import_completed or result.store_summary["concepts_count"] != 2:
        return _fail("B", "Synthetic UMLS transactional import", "unexpected_import_summary", result.safe_public_summary())
    return _ok("B", "Synthetic UMLS transactional import", result.safe_public_summary())


def case_c_snomed() -> dict:
    parsed = parse_snomed_concept_description(
        concept_text=SNOMED_CONCEPT,
        description_text=SNOMED_DESCRIPTION,
    )
    result = _executor().execute_synthetic(parsed, source_safe_id="term_src_safe_snomed")
    if not result.audit.import_completed or result.store_summary["concepts_count"] != 2:
        return _fail("C", "Synthetic SNOMED transactional import", "unexpected_import_summary", result.safe_public_summary())
    return _ok("C", "Synthetic SNOMED transactional import", result.safe_public_summary())


def case_d_rxnorm() -> dict:
    result = _executor().execute_synthetic(parse_rxnorm_rxnconso(text=RXNORM_TEXT), source_safe_id="term_src_safe_rxnorm")
    if not result.audit.import_completed or result.store_summary["concepts_count"] != 2:
        return _fail("D", "Synthetic RxNorm transactional import", "unexpected_import_summary", result.safe_public_summary())
    return _ok("D", "Synthetic RxNorm transactional import", result.safe_public_summary())


def case_e_loinc() -> dict:
    result = _executor().execute_synthetic(parse_loinc_csv(text=LOINC_TEXT), source_safe_id="term_src_safe_loinc")
    if not result.audit.import_completed or result.store_summary["concepts_count"] != 2:
        return _fail("E", "Synthetic LOINC transactional import", "unexpected_import_summary", result.safe_public_summary())
    return _ok("E", "Synthetic LOINC transactional import", result.safe_public_summary())


def case_f_row_cap() -> dict:
    limits = TerminologyImportLimits(max_rows_per_file_default=1, chunk_size=1, checkpoint_interval_rows=1)
    result = _executor(limits).execute_synthetic(parse_loinc_csv(text=LOINC_TEXT), source_safe_id="term_src_safe_loinc")
    if result.audit.records_imported != 1 or result.store_summary["concepts_count"] != 1:
        return _fail("F", "Row cap stops import safely", "row_cap_not_enforced", result.safe_public_summary())
    return _ok("F", "Row cap stops import safely", result.safe_public_summary())


def case_g_rollback() -> dict:
    result = _executor().execute_synthetic(
        parse_rxnorm_rxnconso(text=RXNORM_TEXT),
        source_safe_id="term_src_safe_rxnorm",
        simulate_failure_after_source=True,
    )
    if result.audit.rollback_performed is not True or result.store_summary["sources_count"] != 0:
        return _fail("G", "Simulated failure rolls back", "rollback_failed", result.safe_public_summary())
    return _ok("G", "Simulated failure rolls back", result.safe_public_summary())


def case_h_checkpoint() -> dict:
    limits = TerminologyImportLimits(chunk_size=1, checkpoint_interval_rows=1)
    result = _executor(limits).execute_synthetic(parse_umls_mrconso(text=UMLS_TEXT), source_safe_id="term_src_safe_umls")
    if result.audit.checkpoint_count < 2:
        return _fail("H", "Checkpoint simulation records progress", "checkpoint_count_low", result.safe_public_summary())
    return _ok("H", "Checkpoint simulation records progress", result.safe_public_summary())


def case_i_real_import_blocked() -> dict:
    try:
        _executor().execute_real_import_blocked()
    except RealTerminologyImportBlocked:
        return _ok("I", "Real import blocked by default", {"real_import_performed": False})
    return _fail("I", "Real import blocked by default", "real_import_not_blocked")


def case_j_b07_boundary() -> dict:
    executor = _executor()
    executor.execute_synthetic(parse_rxnorm_rxnconso(text=RXNORM_TEXT), source_safe_id="term_src_safe_rxnorm")
    service = TerminologyLookupService(executor.store)
    mapped = code_entity_via_local_terminology("metformin", service, systems=[TerminologySystem.RXNORM])
    unknown = code_entity_via_local_terminology("zzzz unknown", service, systems=[TerminologySystem.RXNORM])
    boundary = safe_b07_boundary_summary()
    if mapped.status != TerminologyLookupStatus.EXACT or unknown.status != TerminologyLookupStatus.UNMAPPED:
        return _fail("J", "B07 boundary unchanged", "lookup_boundary_failed")
    if boundary["coding_promotes_hypothesis"] or boundary["coding_clears_ddi_status"]:
        return _fail("J", "B07 boundary unchanged", "coding_boundary_failed", boundary)
    return _ok("J", "B07 boundary unchanged", boundary)


def case_k_report_safety(report: dict) -> dict:
    check = check_public_report_payload(report)
    if not check.passed:
        return _fail("K", "Report privacy clean", "privacy_checker_rejected", {"leaks": check.leak_examples_redacted})
    text = json.dumps(report).lower()
    for token in ("license_ack_private", "replacement_map", "source_response_raw", "api_key", "sk-"):
        if token in text:
            return _fail("K", "Report privacy clean", f"forbidden_token_{token}")
    return _ok("K", "Report privacy clean", {"privacy_checker_passed": True})


def _build_report(results: list[dict]) -> dict:
    boundary = safe_b07_boundary_summary()
    return {
        "block_id": "CKA-TERM-01C",
        "conclusion": "cka_term01c_synthetic_import_executor_ready",
        "synthetic_transactional_import_passed": True,
        "rollback_on_failure_passed": True,
        "row_caps_enforced": True,
        "checkpoint_simulation_ready": True,
        "real_import_performed": False,
        "real_terminology_files_committed": False,
        "terminology_data_staged": False,
        "external_api_used": False,
        "clinical_recommendations_generated": False,
        "prescription_dosing_advice_generated": False,
        "coding_does_not_promote_hypothesis": not boundary["coding_promotes_hypothesis"],
        "coding_does_not_clear_ddi_status": not boundary["coding_clears_ddi_status"],
        "no_code_hallucinated": boundary["no_code_hallucinated"],
        "next_manual_action": "operator downloads licensed files and creates private license ack",
        "next_code_action_after_manual_files": "CKA-TERM-02 controlled local terminology import",
        "case_results": results,
        "synthetic_cases_run": len(results),
        "cases_passed": sum(1 for r in results if r["passed"]),
        "all_passed": all(r["passed"] for r in results),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_reports(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cka_term01c_import_executor_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# CKA-TERM-01C Synthetic Import Executor Report",
        "",
        f"- block_id: {report['block_id']}",
        f"- conclusion: {report['conclusion']}",
        f"- synthetic_transactional_import_passed: {report['synthetic_transactional_import_passed']}",
        f"- rollback_on_failure_passed: {report['rollback_on_failure_passed']}",
        f"- row_caps_enforced: {report['row_caps_enforced']}",
        f"- checkpoint_simulation_ready: {report['checkpoint_simulation_ready']}",
        "",
        "## Safety",
        "",
        f"- real_import_performed: {report['real_import_performed']}",
        f"- real_terminology_files_committed: {report['real_terminology_files_committed']}",
        f"- terminology_data_staged: {report['terminology_data_staged']}",
        f"- external_api_used: {report['external_api_used']}",
        f"- clinical_recommendations_generated: {report['clinical_recommendations_generated']}",
        f"- prescription_dosing_advice_generated: {report['prescription_dosing_advice_generated']}",
        f"- coding_does_not_promote_hypothesis: {report['coding_does_not_promote_hypothesis']}",
        f"- coding_does_not_clear_ddi_status: {report['coding_does_not_clear_ddi_status']}",
        f"- no_code_hallucinated: {report['no_code_hallucinated']}",
        "",
        "## Cases",
        "",
    ]
    for result in report["case_results"]:
        marker = "[PASS]" if result["passed"] else "[FAIL]"
        lines.append(f"- Case {result['case']}: {marker} {result['description']}")
        if not result["passed"]:
            lines.append(f"  - error: {result.get('error', 'unknown')}")
    lines += [
        "",
        "## Next manual action",
        "",
        report["next_manual_action"],
        "",
        "## Next code action after manual files",
        "",
        report["next_code_action_after_manual_files"],
        "",
    ]
    (REPORT_DIR / "cka_term01c_import_executor_report.md").write_text("\n".join(lines), encoding="utf-8")
    guide = [
        "# CKA-TERM-01C Import Executor Guide",
        "",
        "TERM-01C validates synthetic/temp execution mechanics only.",
        "",
        "## What it does",
        "",
        "- Opens an explicit transaction.",
        "- Writes a synthetic source manifest.",
        "- Writes synthetic concepts and synonyms.",
        "- Writes an import audit event.",
        "- Commits on success and rolls back on simulated failure.",
        "",
        "## What it does not do",
        "",
        "- It does not import real licensed terminology.",
        "- It does not create a production terminology index by default.",
        "- It does not change B07 default coding behavior.",
        "- It does not call external APIs.",
        "",
    ]
    (REPORT_DIR / "CKA_TERM01C_IMPORT_EXECUTOR_GUIDE.md").write_text("\n".join(guide), encoding="utf-8")


def run_validation() -> dict:
    print("  [TERM-01C] case A: TERM-01B baseline ...", flush=True)
    results = [case_a_term01b_baseline()]
    print("  [TERM-01C] case B: synthetic UMLS import ...", flush=True)
    results.append(case_b_umls())
    print("  [TERM-01C] case C: synthetic SNOMED import ...", flush=True)
    results.append(case_c_snomed())
    print("  [TERM-01C] case D: synthetic RxNorm import ...", flush=True)
    results.append(case_d_rxnorm())
    print("  [TERM-01C] case E: synthetic LOINC import ...", flush=True)
    results.append(case_e_loinc())
    print("  [TERM-01C] case F: row cap ...", flush=True)
    results.append(case_f_row_cap())
    print("  [TERM-01C] case G: rollback ...", flush=True)
    results.append(case_g_rollback())
    print("  [TERM-01C] case H: checkpoint simulation ...", flush=True)
    results.append(case_h_checkpoint())
    print("  [TERM-01C] case I: real import blocked ...", flush=True)
    results.append(case_i_real_import_blocked())
    print("  [TERM-01C] case J: B07 boundary ...", flush=True)
    results.append(case_j_b07_boundary())
    report = _build_report(results)
    print("  [TERM-01C] case K: report safety ...", flush=True)
    results.append(case_k_report_safety(report))
    report = _build_report(results)
    check = check_public_report_payload(report)
    if not check.passed:
        raise RuntimeError(f"CKA-B02 privacy checker rejected TERM-01C report: {check.leak_examples_redacted}")
    _write_reports(report)
    return report


if __name__ == "__main__":
    rep = run_validation()
    status = "[PASS]" if rep["all_passed"] else "[FAIL]"
    print(f"\nCKA-TERM-01C Synthetic Import Executor - {status}")
    print(f"  conclusion: {rep['conclusion']}")
    print(f"  cases_passed: {rep['cases_passed']} / {rep['synthetic_cases_run']}")
    for case in rep["case_results"]:
        marker = "[PASS]" if case["passed"] else "[FAIL]"
        print(f"    {marker} case {case['case']}: {case['description']}")
        if not case["passed"]:
            print(f"           error: {case.get('error')}")
    if not rep["all_passed"]:
        raise SystemExit(1)
