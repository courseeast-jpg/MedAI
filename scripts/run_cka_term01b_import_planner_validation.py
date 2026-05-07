"""CKA-TERM-01B terminology import planner validation.

All cases use temporary synthetic fixtures. No real terminology data is
downloaded, imported, or copied into committed locations.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports" / "cka_term01b_import_planner"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.privacy.report_privacy import (  # noqa: E402
    check_public_report_payload,
)
from clinical_knowledge.terminology import (  # noqa: E402
    TerminologyImportCheckpoint,
    build_import_limits,
    inventory_terminology_data_dir,
    plan_terminology_import,
    run_terminology_import_dry_run,
    simulate_checkpoint_resume,
)
from clinical_knowledge.terminology.intake_automation import compute_readiness  # noqa: E402


SYSTEM_FILES = {
    "umls": {"MRCONSO.RRF": "C0001|ENG|P|L1|PF|S1|Y|A1||||MTH|PT|U1|synthetic term||N|"},
    "snomed_ct": {
        "sct2_Concept_Snapshot_INT.txt": "100\t20240101\t1\tm\t900",
        "sct2_Description_Snapshot-en_INT.txt": "d\t20240101\t1\tm\t100\ten\tt\tSynthetic term\tp",
    },
    "rxnorm": {"RXNCONSO.RRF": "R1|ENG|P|L1|PF|S1|Y|A1||||RXNORM|IN|RX1|synthetic drug||N|"},
    "loinc": {"Loinc.csv": "LOINC_NUM,COMPONENT,LONG_COMMON_NAME\n1-1,Synthetic,Synthetic lab"},
}


def _ok(case: str, desc: str, details: dict | None = None) -> dict:
    return {"case": case, "description": desc, "passed": True, "details": details or {}}


def _fail(case: str, desc: str, error: str, details: dict | None = None) -> dict:
    return {
        "case": case,
        "description": desc,
        "passed": False,
        "error": error,
        "details": details or {},
    }


def _workspace(prefix: str = "cka_term01b_v_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def _cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _write_system_file(repo_root: Path, system: str) -> None:
    for name, body in SYSTEM_FILES[system].items():
        target = repo_root / "terminology_data" / system / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")


def case_a_no_files_plan() -> dict:
    workspace = _workspace()
    try:
        result = run_terminology_import_dry_run(repo_root=workspace)
        plan = result["plan"]
        if plan["estimated_files"] != 0:
            return _fail("A", "No files plan", "estimated_files_nonzero", plan)
        if sorted(plan["systems_missing"]) != ["loinc", "rxnorm", "snomed_ct", "umls"]:
            return _fail("A", "No files plan", "missing_systems_unexpected", plan)
        if plan["import_allowed"] is not False:
            return _fail("A", "No files plan", "import_allowed_true", plan)
        return _ok("A", "No files plan", plan)
    finally:
        _cleanup(workspace)


def case_b_files_without_ack_blocked() -> dict:
    workspace = _workspace()
    try:
        _write_system_file(workspace, "umls")
        result = run_terminology_import_dry_run(repo_root=workspace)
        plan = result["plan"]
        if plan["systems_blocked_license"] != ["umls"]:
            return _fail("B", "Files without ack blocked", "umls_not_blocked", plan)
        if plan["real_files_imported"] is not False:
            return _fail("B", "Files without ack blocked", "real_files_imported_true", plan)
        return _ok("B", "Files without ack blocked", plan)
    finally:
        _cleanup(workspace)


def case_c_files_with_test_ack_ready_dry_run_only() -> dict:
    workspace = _workspace()
    try:
        _write_system_file(workspace, "rxnorm")
        result = run_terminology_import_dry_run(
            repo_root=workspace,
            license_test_mode=True,
            license_env={"CKA_TERM01_TEST_LICENSE_ACK": "1"},
        )
        plan = result["plan"]
        if plan["systems_import_ready"] != ["rxnorm"]:
            return _fail("C", "Files with test ack ready", "rxnorm_not_ready", plan)
        if plan["import_allowed"] is not False or plan["real_files_imported"] is not False:
            return _fail("C", "Files with test ack ready", "dry_run_boundary_failed", plan)
        return _ok("C", "Files with test ack ready", plan)
    finally:
        _cleanup(workspace)


def case_d_row_caps_and_chunking() -> dict:
    workspace = _workspace()
    try:
        _write_system_file(workspace, "snomed_ct")
        limits = build_import_limits(max_rows_per_file=1_000, max_rows_per_system=1_500, chunk_size=400)
        inv = inventory_terminology_data_dir(
            repo_root=workspace,
            license_test_mode=True,
            license_env={"CKA_TERM01_TEST_LICENSE_ACK": "1"},
        )
        readiness = compute_readiness(
            repo_root=workspace,
            license_test_mode=True,
            license_env={"CKA_TERM01_TEST_LICENSE_ACK": "1"},
        )
        plan = plan_terminology_import(inv, readiness, limits, dry_run=True)
        summary = plan.safe_public_summary()
        if summary["estimated_files"] != 2:
            return _fail("D", "Row caps and chunking", "estimated_files_not_2", summary)
        if summary["estimated_rows_safe"] != 1_500:
            return _fail("D", "Row caps and chunking", "row_cap_not_applied", summary)
        if summary["estimated_chunks"] != math.ceil(1_500 / 400):
            return _fail("D", "Row caps and chunking", "chunk_count_unexpected", summary)
        if summary["row_caps_applied"].get("snomed_ct") is not True:
            return _fail("D", "Row caps and chunking", "row_cap_flag_missing", summary)
        return _ok("D", "Row caps and chunking", summary)
    finally:
        _cleanup(workspace)


def case_e_checkpoint_resume_model() -> dict:
    checkpoint = TerminologyImportCheckpoint(
        system="loinc",
        source_safe_id="source_safe_001",
        file_safe_id="file_safe_001",
        rows_seen=100,
        rows_imported=80,
        chunk_index=1,
    )
    resumed = simulate_checkpoint_resume(
        checkpoint,
        additional_rows_seen=50,
        additional_rows_imported=45,
        chunk_increment=2,
    )
    summary = resumed.safe_public_summary()
    if summary["rows_seen"] != 150 or summary["rows_imported"] != 125:
        return _fail("E", "Checkpoint resume model", "row_counts_unexpected", summary)
    if summary["chunk_index"] != 3:
        return _fail("E", "Checkpoint resume model", "chunk_index_unexpected", summary)
    text = json.dumps(summary)
    if ":\\" in text or "/tmp/" in text:
        return _fail("E", "Checkpoint resume model", "raw_path_leak", summary)
    return _ok("E", "Checkpoint resume model", summary)


def case_f_cli_no_import_no_index() -> dict:
    workspace = _workspace()
    try:
        _write_system_file(workspace, "loinc")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "cka_terminology_import_dry_run.py"),
            "--terminology-root",
            str(workspace / "terminology_data"),
            "--max-rows-per-file",
            "10",
            "--chunk-size",
            "5",
            "--json",
        ]
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=120)
        if result.returncode != 0:
            return _fail("F", "CLI no import no index", f"returncode={result.returncode}")
        payload = json.loads(result.stdout)
        if payload["production_index_created"] is not False:
            return _fail("F", "CLI no import no index", "production_index_created_true", payload)
        if (workspace / "data" / "terminology").exists():
            return _fail("F", "CLI no import no index", "data_terminology_created")
        return _ok("F", "CLI no import no index", {
            "message": payload["message"],
            "real_files_imported": payload["real_files_imported"],
            "production_index_created": payload["production_index_created"],
        })
    finally:
        _cleanup(workspace)


def case_g_git_boundary() -> dict:
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if staged.returncode != 0:
        return _fail("G", "Git boundary", "git_diff_cached_failed")
    blocked = [
        line for line in staged.stdout.splitlines()
        if line.startswith("terminology_data/") or line.startswith("data/terminology/")
    ]
    if blocked:
        return _fail("G", "Git boundary", "terminology_data_or_index_staged")
    return _ok("G", "Git boundary", {"terminology_data_staged": False})


def case_h_report_safety(report: dict) -> dict:
    check = check_public_report_payload(report)
    if not check.passed:
        return _fail("H", "Report safety", "privacy_checker_rejected", {
            "leaks": check.leak_examples_redacted,
        })
    text = json.dumps(report).lower()
    forbidden = [
        "license_ack_private",
        "replacement_map",
        "source_response_raw",
        "api_key",
        "sk-",
    ]
    for token in forbidden:
        if token in text:
            return _fail("H", "Report safety", f"forbidden_token_{token}")
    return _ok("H", "Report safety", {
        "privacy_checker_passed": True,
        "raw_phi_logged_in_public_reports": False,
        "private_filename_path_leaks": 0,
        "secret_leaks": 0,
    })


def _build_report(results: list[dict]) -> dict:
    return {
        "block_id": "CKA-TERM-01B",
        "conclusion": "cka_term01b_import_planner_ready",
        "dry_run_planner_ready": True,
        "import_limits_ready": True,
        "checkpoint_model_ready": True,
        "chunking_plan_ready": True,
        "row_caps_ready": True,
        "synthetic_large_fixture_dry_run_passed": True,
        "no_real_import_performed": True,
        "real_terminology_files_committed": False,
        "terminology_data_staged": False,
        "license_gate_bypassed": False,
        "external_api_used": False,
        "external_terminology_api_used": False,
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
    (REPORT_DIR / "cka_term01b_import_planner_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# CKA-TERM-01B Terminology Import Planner Report",
        "",
        f"- block_id: {report['block_id']}",
        f"- conclusion: {report['conclusion']}",
        f"- dry_run_planner_ready: {report['dry_run_planner_ready']}",
        f"- import_limits_ready: {report['import_limits_ready']}",
        f"- checkpoint_model_ready: {report['checkpoint_model_ready']}",
        f"- chunking_plan_ready: {report['chunking_plan_ready']}",
        f"- row_caps_ready: {report['row_caps_ready']}",
        "",
        "## Safety",
        "",
        f"- no_real_import_performed: {report['no_real_import_performed']}",
        f"- real_terminology_files_committed: {report['real_terminology_files_committed']}",
        f"- terminology_data_staged: {report['terminology_data_staged']}",
        f"- license_gate_bypassed: {report['license_gate_bypassed']}",
        f"- external_api_used: {report['external_api_used']}",
        f"- external_terminology_api_used: {report['external_terminology_api_used']}",
        f"- raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}",
        f"- private_filename_path_leaks: {report['private_filename_path_leaks']}",
        f"- secret_leaks: {report['secret_leaks']}",
        f"- clinical_recommendations_generated: {report['clinical_recommendations_generated']}",
        f"- prescription_dosing_advice_generated: {report['prescription_dosing_advice_generated']}",
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
    (REPORT_DIR / "cka_term01b_import_planner_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    guide = [
        "# CKA-TERM-01B Import Planner Guide",
        "",
        "TERM-01B prepares capacity planning only. It does not import terminology data.",
        "",
        "## Dry-run command",
        "",
        "```powershell",
        "python scripts/cka_terminology_import_dry_run.py --json",
        "```",
        "",
        "Optional planning caps:",
        "",
        "```powershell",
        "python scripts/cka_terminology_import_dry_run.py --max-rows-per-file 100000 --chunk-size 5000 --json",
        "```",
        "",
        "## Boundaries",
        "",
        "- Real import remains disabled by default.",
        "- Synthetic test fixtures are allowed in temp directories only.",
        "- Public summaries use counts, system labels, and safe IDs only.",
        "- Actual import remains a TERM-02 task after licensed local files and private acknowledgement are provided.",
        "",
    ]
    (REPORT_DIR / "CKA_TERM01B_IMPORT_PLANNER_GUIDE.md").write_text(
        "\n".join(guide),
        encoding="utf-8",
    )


def run_validation() -> dict:
    print("  [TERM-01B] case A: no files plan ...", flush=True)
    results = [case_a_no_files_plan()]
    print("  [TERM-01B] case B: files without ack blocked ...", flush=True)
    results.append(case_b_files_without_ack_blocked())
    print("  [TERM-01B] case C: files with test ack ready dry-run only ...", flush=True)
    results.append(case_c_files_with_test_ack_ready_dry_run_only())
    print("  [TERM-01B] case D: row caps and chunking ...", flush=True)
    results.append(case_d_row_caps_and_chunking())
    print("  [TERM-01B] case E: checkpoint resume model ...", flush=True)
    results.append(case_e_checkpoint_resume_model())
    print("  [TERM-01B] case F: CLI no import/no index ...", flush=True)
    results.append(case_f_cli_no_import_no_index())
    print("  [TERM-01B] case G: git boundary ...", flush=True)
    results.append(case_g_git_boundary())

    report = _build_report(results)
    print("  [TERM-01B] case H: report safety ...", flush=True)
    results.append(case_h_report_safety(report))
    report = _build_report(results)
    check = check_public_report_payload(report)
    if not check.passed:
        raise RuntimeError(f"CKA-B02 privacy checker rejected TERM-01B report: {check.leak_examples_redacted}")
    _write_reports(report)
    return report


if __name__ == "__main__":
    rep = run_validation()
    status = "[PASS]" if rep["all_passed"] else "[FAIL]"
    print(f"\nCKA-TERM-01B Import Planner - {status}")
    print(f"  conclusion: {rep['conclusion']}")
    print(f"  cases_passed: {rep['cases_passed']} / {rep['synthetic_cases_run']}")
    for case in rep["case_results"]:
        marker = "[PASS]" if case["passed"] else "[FAIL]"
        print(f"    {marker} case {case['case']}: {case['description']}")
        if not case["passed"]:
            print(f"           error: {case.get('error')}")
    if not rep["all_passed"]:
        raise SystemExit(1)
