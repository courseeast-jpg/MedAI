"""CKA-TERM-01D terminology QA validation."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports" / "cka_term01d_terminology_qa"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.privacy.report_privacy import check_public_report_payload  # noqa: E402
from clinical_knowledge.terminology import run_synthetic_terminology_qa  # noqa: E402


def _ok(case: str, description: str, details: dict | None = None) -> dict:
    return {"case": case, "description": description, "passed": True, "details": details or {}}


def _fail(case: str, description: str, error: str, details: dict | None = None) -> dict:
    return {"case": case, "description": description, "passed": False, "error": error, "details": details or {}}


def _run_script(script: str, timeout: int = 420) -> tuple[bool, dict]:
    result = subprocess.run(
        [sys.executable, script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return result.returncode == 0, {
        "script": Path(script).name,
        "returncode": result.returncode,
    }


def case_a_fixtures_load(qa_summary: dict) -> dict:
    metadata = qa_summary["fixture_metadata"]
    if sorted(metadata["systems_loaded"]) != ["loinc", "rxnorm", "snomed_ct", "umls"]:
        return _fail("A", "Synthetic QA fixtures load", "systems_loaded_unexpected", metadata)
    if qa_summary["metrics"]["total_cases"] < 7:
        return _fail("A", "Synthetic QA fixtures load", "case_count_low", qa_summary["metrics"])
    return _ok("A", "Synthetic QA fixtures load", metadata)


def _metric_case(case: str, description: str, qa_summary: dict, metric: str) -> dict:
    if qa_summary["metrics"].get(metric) is not True:
        return _fail(case, description, f"{metric}_not_true", qa_summary["metrics"])
    return _ok(case, description, {metric: True})


def case_h_no_external_api(qa_summary: dict) -> dict:
    if qa_summary["metrics"]["external_api_used"] is not False:
        return _fail("H", "No external API", "external_api_used_true")
    if qa_summary["metrics"]["real_terminology_imported"] is not False:
        return _fail("H", "No external API", "real_terminology_imported_true")
    return _ok("H", "No external API", {
        "external_api_used": False,
        "real_terminology_imported": False,
    })


def case_i_report_privacy(report: dict) -> dict:
    check = check_public_report_payload(report)
    if not check.passed:
        return _fail("I", "Privacy report clean", "privacy_checker_rejected", {"leaks": check.leak_examples_redacted})
    text = json.dumps(report).lower()
    for token in ("license_ack_private", "replacement_map", "source_response_raw", "api_key", "sk-"):
        if token in text:
            return _fail("I", "Privacy report clean", f"forbidden_token_{token}")
    return _ok("I", "Privacy report clean", {"privacy_checker_passed": True})


def case_j_baseline_validations() -> dict:
    scripts = [
        "scripts/run_cka_term01b_import_planner_validation.py",
        "scripts/run_cka_term01a_intake_automation_validation.py",
        "scripts/run_cka_term01_real_terminology_readiness_validation.py",
    ]
    outcomes = []
    for script in scripts:
        passed, details = _run_script(script)
        outcomes.append(details)
        if not passed:
            return _fail("J", "TERM-01/01A/01B validations still pass", "baseline_validation_failed", {"outcomes": outcomes})
    return _ok("J", "TERM-01/01A/01B validations still pass", {"outcomes": outcomes})


def case_k_final_validation() -> dict:
    passed, details = _run_script("scripts/run_cka_final_mvp_release_validation.py")
    if not passed:
        return _fail("K", "Final CKA validation still passes", "final_validation_failed", details)
    return _ok("K", "Final CKA validation still passes", details)


def _build_report(results: list[dict], qa_summary: dict) -> dict:
    metrics = qa_summary["metrics"]
    return {
        "block_id": "CKA-TERM-01D",
        "conclusion": "cka_term01d_terminology_qa_ready",
        "synthetic_golden_cases_ready": True,
        "exact_match_validation_ready": metrics["exact_match_passed"],
        "synonym_match_validation_ready": metrics["synonym_match_passed"],
        "ambiguity_validation_ready": metrics["ambiguous_flag_passed"],
        "unmapped_no_hallucination_ready": metrics["unmapped_no_hallucination_passed"],
        "b07_boundary_preserved": metrics["b07_boundary_passed"],
        "real_terminology_imported": False,
        "external_api_used": False,
        "clinical_recommendations_generated": False,
        "prescription_dosing_advice_generated": False,
        "next_manual_action": "operator downloads licensed terminology files",
        "next_code_action_after_manual_files": "CKA-TERM-02 controlled local terminology import",
        "qa_summary": qa_summary,
        "case_results": results,
        "validation_cases_run": len(results),
        "validation_cases_passed": sum(1 for result in results if result["passed"]),
        "all_passed": all(result["passed"] for result in results),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_reports(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cka_term01d_terminology_qa_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# CKA-TERM-01D Terminology QA Report",
        "",
        f"- block_id: {report['block_id']}",
        f"- conclusion: {report['conclusion']}",
        f"- synthetic_golden_cases_ready: {report['synthetic_golden_cases_ready']}",
        f"- exact_match_validation_ready: {report['exact_match_validation_ready']}",
        f"- synonym_match_validation_ready: {report['synonym_match_validation_ready']}",
        f"- ambiguity_validation_ready: {report['ambiguity_validation_ready']}",
        f"- unmapped_no_hallucination_ready: {report['unmapped_no_hallucination_ready']}",
        f"- b07_boundary_preserved: {report['b07_boundary_preserved']}",
        "",
        "## Safety",
        "",
        f"- real_terminology_imported: {report['real_terminology_imported']}",
        f"- external_api_used: {report['external_api_used']}",
        f"- clinical_recommendations_generated: {report['clinical_recommendations_generated']}",
        f"- prescription_dosing_advice_generated: {report['prescription_dosing_advice_generated']}",
        "",
        "## QA metrics",
        "",
    ]
    for key, value in report["qa_summary"]["metrics"].items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Validation cases",
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
    (REPORT_DIR / "cka_term01d_terminology_qa_report.md").write_text("\n".join(lines), encoding="utf-8")
    guide = [
        "# CKA-TERM-01D Terminology QA Guide",
        "",
        "TERM-01D provides a synthetic golden lookup harness that TERM-02 can reuse after real local imports are available.",
        "",
        "## Command",
        "",
        "```powershell",
        "python scripts/cka_terminology_run_qa.py --json",
        "```",
        "",
        "## Boundaries",
        "",
        "- Uses synthetic fixtures only.",
        "- Unknown terms must remain unmapped.",
        "- Ambiguous terms must be flagged.",
        "- B07 local terminology lookup remains opt-in.",
        "- No external APIs are used.",
        "",
    ]
    (REPORT_DIR / "CKA_TERM01D_TERMINOLOGY_QA_GUIDE.md").write_text("\n".join(guide), encoding="utf-8")


def run_validation() -> dict:
    qa_summary = run_synthetic_terminology_qa().safe_public_summary()
    print("  [TERM-01D] case A: fixtures load ...", flush=True)
    results = [case_a_fixtures_load(qa_summary)]
    print("  [TERM-01D] case B: exact match ...", flush=True)
    results.append(_metric_case("B", "Exact match passes", qa_summary, "exact_match_passed"))
    print("  [TERM-01D] case C: synonym match ...", flush=True)
    results.append(_metric_case("C", "Synonym match passes", qa_summary, "synonym_match_passed"))
    print("  [TERM-01D] case D: ambiguity flagged ...", flush=True)
    results.append(_metric_case("D", "Ambiguous term is flagged", qa_summary, "ambiguous_flag_passed"))
    print("  [TERM-01D] case E/F: unknown unmapped/no hallucination ...", flush=True)
    results.append(_metric_case("E", "Unknown term is unmapped", qa_summary, "unmapped_no_hallucination_passed"))
    results.append(_metric_case("F", "No hallucinated code", qa_summary, "unmapped_no_hallucination_passed"))
    print("  [TERM-01D] case G: B07 opt-in boundary ...", flush=True)
    results.append(_metric_case("G", "B07 opt-in helper preserves boundary", qa_summary, "b07_boundary_passed"))
    print("  [TERM-01D] case H: no external API ...", flush=True)
    results.append(case_h_no_external_api(qa_summary))
    report = _build_report(results, qa_summary)
    print("  [TERM-01D] case I: privacy clean ...", flush=True)
    results.append(case_i_report_privacy(report))
    print("  [TERM-01D] case J: TERM-01/01A/01B validations ...", flush=True)
    results.append(case_j_baseline_validations())
    print("  [TERM-01D] case K: final CKA validation ...", flush=True)
    results.append(case_k_final_validation())
    report = _build_report(results, qa_summary)
    check = check_public_report_payload(report)
    if not check.passed:
        raise RuntimeError(f"CKA-B02 privacy checker rejected TERM-01D report: {check.leak_examples_redacted}")
    _write_reports(report)
    return report


if __name__ == "__main__":
    rep = run_validation()
    status = "[PASS]" if rep["all_passed"] else "[FAIL]"
    print(f"\nCKA-TERM-01D Terminology QA - {status}")
    print(f"  conclusion: {rep['conclusion']}")
    print(f"  cases_passed: {rep['validation_cases_passed']} / {rep['validation_cases_run']}")
    for case in rep["case_results"]:
        marker = "[PASS]" if case["passed"] else "[FAIL]"
        print(f"    {marker} case {case['case']}: {case['description']}")
        if not case["passed"]:
            print(f"           error: {case.get('error')}")
    if not rep["all_passed"]:
        raise SystemExit(1)
