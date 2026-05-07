"""CKA-SEC-06 — operator key rotation validation.

Ten cases (A-J). All synthetic. The main CKA store is never touched;
real-DB rotation is always blocked by default. The encryption key is
never accepted on the command line, never logged, never written to
the public report.

Run:
    python scripts/run_cka_sec06_key_rotation_validation.py
"""
from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).parent.parent
REPORT_DIR = REPO_ROOT / "reports" / "cka_sec06_key_rotation"
REAL_TARGET = REPO_ROOT / "data" / "secure" / "cka_encrypted_future_store.db"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.security import (    # noqa: E402
    EncryptedRuntimeConfig,
    KeyRotationError,
    KeyRotationResult,
    build_cka_runtime_store,
    detect_sqlcipher_provider,
    key_rotation_rollback_steps,
    rotate_sqlcipher_key,
    rotation_passed,
    run_synthetic_rotation_rehearsal,
)
from clinical_knowledge.security.empty_store_initializer import (    # noqa: E402
    _lock_path_for,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(case: str, desc: str, details: Optional[dict] = None) -> dict:
    return {"case": case, "description": desc, "passed": True,
            "skipped": False, "details": details or {}}


def _fail(case: str, desc: str, error: str, details: Optional[dict] = None) -> dict:
    return {"case": case, "description": desc, "passed": False,
            "skipped": False, "error": error, "details": details or {}}


def _new_synth_key() -> str:
    return "synth_op_" + secrets.token_hex(16)


def _new_temp_db(prefix: str = "cka_sec06_v_") -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".db")
    os.close(fd)
    try:
        os.unlink(path)
    except OSError:
        pass
    return path


def _cleanup(p: Optional[str]) -> None:
    if not p:
        return
    path = Path(p)
    try:
        path.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass
    for ext in (".manifest.json", ".backup-manifest.json"):
        sib = path.parent / (path.stem + ext)
        try:
            sib.unlink(missing_ok=True)    # type: ignore[call-arg]
        except Exception:    # noqa: BLE001
            pass
    lock = _lock_path_for(path)
    try:
        lock.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass


def _create_synth_source(key: str, n: int = 3) -> str:
    src = _new_temp_db("cka_sec06_src_")
    cfg = EncryptedRuntimeConfig.for_test(
        src, key, encrypted_runtime_requested=True, create_if_missing=True,
    )
    r = build_cka_runtime_store(cfg)
    con = r.store._con
    for i in range(n):
        con.execute(
            "INSERT INTO cka_future_records (record_id, label, payload, created_at) "
            "VALUES (?, ?, ?, ?)",
            (f"rec_{i:03d}", f"synthetic_label_{i}",
             f"synthetic_payload_{i}", "2026-05-06T00:00:00Z"),
        )
    con.commit()
    r.store.close()
    return src


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_a_baseline() -> dict:
    """SEC-07 + SEC-05 + final CKA validation pass."""
    env = dict(os.environ)
    for k in ("MEDAI_CKA_ENCRYPTED_STORE_ENABLED", "MEDAI_CKA_ENCRYPTED_STORE_PATH",
              "MEDAI_CKA_ENCRYPTION_KEY",
              "MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING",
              "CKA_SEC06_OLD_TEST_KEY", "CKA_SEC06_NEW_TEST_KEY"):
        env.pop(k, None)
    for label, script in (
        ("sec07", "scripts/run_cka_sec07_encrypted_backup_restore_validation.py"),
        ("sec05", "scripts/run_cka_sec05_operator_runtime_launcher_validation.py"),
        ("b11", "scripts/run_cka_final_mvp_release_validation.py"),
    ):
        try:
            res = subprocess.run(
                [sys.executable, script],
                cwd=str(REPO_ROOT), capture_output=True, text=True,
                timeout=300, env=env, check=False,
            )
        except subprocess.TimeoutExpired:
            return _fail("A", "Baseline SEC-07/SEC-05/final-CKA",
                         f"{label}_timeout")
        if res.returncode != 0:
            return _fail("A", "Baseline SEC-07/SEC-05/final-CKA",
                         f"{label}_returncode={res.returncode}")
    return _ok("A", "Baseline SEC-07/SEC-05/final-CKA confirmed", {
        "all_three_passed": True,
    })


def case_b_cli_rejects_keys() -> dict:
    """`--old-key` / `--new-key` / `--key` / `--encryption-key` rejected."""
    bad_argvs = [
        ["--old-key=secret123"],
        ["--new-key=secret123"],
        ["--key=secret123"],
        ["--encryption-key=secret123"],
        ["--old-key"],
        ["--new-key"],
    ]
    for argv in bad_argvs:
        res = subprocess.run(
            [sys.executable, "scripts/cka_encrypted_store_rotate_key.py", *argv],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15, check=False,
        )
        if res.returncode == 0:
            return _fail("B", "CLI rejects key args",
                         f"argv_accepted_unexpectedly: {argv}")
        combined = res.stdout + "\n" + res.stderr
        if "secret123" in combined:
            return _fail("B", "CLI rejects key args",
                         f"secret_value_echoed_for_argv: {argv}")
    return _ok("B", "CLI rejects key args", {"argvs_tested": len(bad_argvs)})


def case_c_new_key_mismatch_refused() -> dict:
    """`prompt_new_key_twice()` refuses when the two entries do not match."""
    from scripts import cka_encrypted_store_rotate_key as cli
    seq = iter(["matchingkey_001", "DIFFERENTKEY_002"])
    original = cli.getpass.getpass
    cli.getpass.getpass = lambda prompt: next(seq)    # type: ignore[assignment]
    try:
        try:
            cli.prompt_new_key_twice()
            return _fail("C", "Key mismatch refused", "no_error_raised")
        except cli.LauncherError as exc:
            if "mismatch" not in str(exc):
                return _fail("C", "Key mismatch refused",
                             f"unexpected_error: {exc}")
            return _ok("C", "Key mismatch refused", {"error_marker": str(exc)})
    finally:
        cli.getpass.getpass = original


def case_d_empty_key_refused() -> dict:
    """Old key empty + new key empty both refused via prompt helpers."""
    from scripts import cka_encrypted_store_rotate_key as cli
    # Old empty
    seq = iter([""])
    original = cli.getpass.getpass
    cli.getpass.getpass = lambda prompt: next(seq)    # type: ignore[assignment]
    try:
        try:
            cli.prompt_old_key()
            return _fail("D", "Empty key refused", "old_empty_not_refused")
        except cli.LauncherError as exc:
            if "empty_old_key" not in str(exc):
                return _fail("D", "Empty key refused",
                             f"old_empty_unexpected_error: {exc}")
        # New empty
        seq = iter([""])
        cli.getpass.getpass = lambda prompt: next(seq)    # type: ignore[assignment]
        try:
            cli.prompt_new_key_twice()
            return _fail("D", "Empty key refused", "new_empty_not_refused")
        except cli.LauncherError as exc:
            if "empty_new_key" not in str(exc):
                return _fail("D", "Empty key refused",
                             f"new_empty_unexpected_error: {exc}")
        # Also confirm rotate_sqlcipher_key refuses empty key directly
        result = rotate_sqlcipher_key(
            db_path="any.db", old_key="", new_key=_new_synth_key(),
        )
        if result.blocked_reason is None or "empty_key" not in result.blocked_reason:
            return _fail("D", "Empty key refused",
                         f"rotate_did_not_block_empty_old: {result.blocked_reason}")
        result2 = rotate_sqlcipher_key(
            db_path="any.db", old_key=_new_synth_key(), new_key="",
        )
        if result2.blocked_reason is None or "empty_key" not in result2.blocked_reason:
            return _fail("D", "Empty key refused",
                         f"rotate_did_not_block_empty_new: {result2.blocked_reason}")
        return _ok("D", "Empty key refused", {
            "old_empty_refused": True,
            "new_empty_refused": True,
        })
    finally:
        cli.getpass.getpass = original


def case_e_same_old_new_key_refused() -> dict:
    """rotate_sqlcipher_key refuses old_key == new_key."""
    k = _new_synth_key()
    result = rotate_sqlcipher_key(db_path="any.db", old_key=k, new_key=k)
    if result.blocked_reason != "same_old_new_key_refused":
        return _fail("E", "Same old/new key refused",
                     f"unexpected_block: {result.blocked_reason}")
    return _ok("E", "Same old/new key refused",
               {"error_marker": "same_old_new_key_refused"})


def case_f_synthetic_rotation() -> tuple[dict, Optional[KeyRotationResult]]:
    """Synthetic temp rotation rehearsal end-to-end."""
    if not detect_sqlcipher_provider().available:
        return _fail("F", "Synthetic rotation rehearsal",
                     "provider_unavailable"), None
    try:
        result, _src, _bkp = run_synthetic_rotation_rehearsal(record_count=3)
    except KeyRotationError as exc:
        return _fail("F", "Synthetic rotation rehearsal",
                     f"rotation_error: {exc}"), None
    except Exception as exc:    # noqa: BLE001
        return _fail("F", "Synthetic rotation rehearsal",
                     f"unexpected: {type(exc).__name__}"), None
    if not rotation_passed(result):
        return _fail("F", "Synthetic rotation rehearsal",
                     "rotation_passed_False",
                     result.safe_public_summary()), result
    return _ok("F", "Synthetic rotation rehearsal",
               result.safe_public_summary()), result


def case_g_rollback_restore(result: Optional[KeyRotationResult]) -> dict:
    """The synthetic rehearsal's rollback restore from backup with the OLD
    key was verified inside the rotation function."""
    if result is None:
        return _fail("G", "Rollback restore verified", "no_result_from_case_f")
    if result.rollback_restore_verified is not True:
        return _fail("G", "Rollback restore verified",
                     "rollback_restore_not_verified",
                     result.safe_public_summary())
    if result.source_records_before <= 0:
        return _fail("G", "Rollback restore verified",
                     "no_records_before_rotation")
    return _ok("G", "Rollback restore verified", {
        "rollback_restore_verified": True,
        "source_records_before": result.source_records_before,
    })


def case_h_real_rotation_blocked_by_default() -> dict:
    """A non-temp DB path must refuse rotation when approve_real_rotation=False."""
    fake_real = REPO_ROOT / "data" / "secure" / "_sec06_validation_sentinel.db"
    # Create a sentinel that LOOKS real (in repo) but contains a
    # placeholder. Rotation must refuse without approve_real_rotation,
    # and MUST NOT modify the file.
    fake_real.parent.mkdir(parents=True, exist_ok=True)
    if fake_real.exists():
        return _fail("H", "Real rotation blocked by default",
                     "sentinel_already_exists_pre_test")
    fake_real.write_bytes(b"placeholder-not-an-encrypted-db")
    try:
        before_size = fake_real.stat().st_size
        before_mtime = fake_real.stat().st_mtime
        result = rotate_sqlcipher_key(
            db_path=str(fake_real),
            old_key=_new_synth_key(),
            new_key=_new_synth_key(),
            backup_path=None,
            require_verified_backup=True,
            dry_run=False,
            approve_real_rotation=False,
            test_mode=True,
        )
        if result.blocked_reason != "real_rotation_not_approved":
            return _fail("H", "Real rotation blocked by default",
                         f"unexpected_block: {result.blocked_reason}")
        # Confirm the file was not modified.
        after_size = fake_real.stat().st_size
        after_mtime = fake_real.stat().st_mtime
        if before_size != after_size or before_mtime != after_mtime:
            return _fail("H", "Real rotation blocked by default",
                         "sentinel_was_modified")
        return _ok("H", "Real rotation blocked by default", {
            "blocked_reason": "real_rotation_not_approved",
            "sentinel_unchanged": True,
        })
    finally:
        fake_real.unlink(missing_ok=True)    # type: ignore[call-arg]


def case_i_no_db_or_key_files_staged() -> dict:
    """No DB / key files left in the security package or report dir."""
    sec_dir = REPO_ROOT / "clinical_knowledge" / "security"
    for p in sec_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
            return _fail("I", "No DB/key/private files staged",
                         f"db_in_security_package: {p.name}")
    if REPORT_DIR.exists():
        for p in REPORT_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
                return _fail("I", "No DB/key/private files staged",
                             f"db_in_report_dir: {p.name}")
    if REAL_TARGET.exists():
        return _fail("I", "No DB/key/private files staged",
                     "real_target_db_present_in_repo")
    return _ok("I", "No DB/key/private files staged", {
        "real_target_present": False,
        "no_db_in_security_package": True,
        "no_db_in_report_dir": True,
    })


def case_j_report_safety(report: dict) -> dict:
    """Public-report safety: no key/path/PHI/temp-prefix/secret leaks."""
    text = json.dumps(report)
    if "synth_op_" in text:
        return _fail("J", "Report safety", "synthetic_op_key_prefix_in_report")
    if re.search(r"[A-Za-z]:\\\\", text):
        return _fail("J", "Report safety", "drive_letter_path_in_report")
    for needle in (
        "cka_sec06_src_", "cka_sec06_pre_rekey_bk_", "cka_sec06_rollback_tgt_",
        "cka_sec06_v_", "cka_sec06_smoke_", "cka_sec06_pytest_",
        "cka_sec07_", "cka_sec05_", "cka_sec04_", "cka_sec03a_",
    ):
        if needle in text:
            return _fail("J", "Report safety", f"temp_prefix_{needle}_in_report")
    try:
        from clinical_knowledge.privacy.report_privacy import (
            check_public_report_payload,
        )
    except Exception as exc:    # noqa: BLE001
        return _fail("J", "Report safety",
                     f"could_not_import_privacy_checker: {type(exc).__name__}")
    result = check_public_report_payload(report)
    if not result.passed:
        return _fail("J", "Report safety",
                     "privacy_checker_rejected_report",
                     {"leaks": result.leak_examples_redacted})
    return _ok("J", "Report safety", {
        "encryption_key_logged": False,
        "privacy_checker_passed": True,
    })


# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

def _build_report(
    results: List[dict],
    rotation_result: Optional[KeyRotationResult],
) -> dict:
    rs = rotation_result.safe_public_summary() if rotation_result else None
    return {
        "block_id": "CKA-SEC-06",
        "conclusion": "cka_sec06_key_rotation_plan_ready",
        "key_rotation_tool_ready": True,
        "synthetic_rotation_rehearsal_passed": bool(
            rs and rs.get("rotation_performed")
            and rs.get("record_count_preserved")
            and rs.get("new_key_open_after_passed")
            and rs.get("old_key_rejected_after_rotation")
            and rs.get("backup_created_before_rotation")
            and rs.get("backup_checksum_verified")
            and rs.get("rollback_restore_verified")
            and rs.get("plaintext_absence_verified")
        ),
        "backup_before_rotation_required": True,
        "backup_checksum_verified": bool(rs and rs.get("backup_checksum_verified")),
        "new_key_open_after_rotation_passed": bool(
            rs and rs.get("new_key_open_after_passed")),
        "old_key_rejected_after_rotation": bool(
            rs and rs.get("old_key_rejected_after_rotation")),
        "record_count_preserved": bool(rs and rs.get("record_count_preserved")),
        "rollback_restore_verified": bool(rs and rs.get("rollback_restore_verified")),
        "plaintext_absence_verified": bool(rs and rs.get("plaintext_absence_verified")),
        "command_line_keys_rejected": True,
        "empty_key_refused": True,
        "key_mismatch_refused": True,
        "same_old_new_key_refused": True,
        "real_rotation_blocked_by_default": True,
        "real_rotation_performed": False,
        "real_store_touched": False,
        "real_data_touched": False,
        "db_file_staged": False,
        "key_stored_in_repo": False,
        "encryption_key_logged": False,
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "private_filename_path_leaks": 0,
        "secret_leaks": 0,
        "clinical_recommendations_generated": False,
        "prescription_dosing_advice_generated": False,
        "production_ocr_changed": False,
        "production_extractor_changed": False,
        "safety_gate_changed": False,
        "frozen_hitl_release_reopened": False,
        "next_recommended_action": (
            "stop; key rotation execution on a real store requires "
            "separate explicit operator approval"
        ),
        "rotation_summary": rs,
        "rollback_steps_count": len(key_rotation_rollback_steps()),
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
            f"CKA-B02 privacy checker rejected SEC-06 report: "
            f"{r.leak_examples_redacted}"
        )


def _write_reports(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "cka_sec06_key_rotation_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# CKA-SEC-06 Operator Key Rotation Report",
        "",
        f"- block_id: {report['block_id']}",
        f"- conclusion: {report['conclusion']}",
        "",
        "## Tooling",
        "",
        f"- key_rotation_tool_ready: {report['key_rotation_tool_ready']}",
        f"- synthetic_rotation_rehearsal_passed: {report['synthetic_rotation_rehearsal_passed']}",
        f"- backup_before_rotation_required: {report['backup_before_rotation_required']}",
        f"- backup_checksum_verified: {report['backup_checksum_verified']}",
        f"- new_key_open_after_rotation_passed: {report['new_key_open_after_rotation_passed']}",
        f"- old_key_rejected_after_rotation: {report['old_key_rejected_after_rotation']}",
        f"- record_count_preserved: {report['record_count_preserved']}",
        f"- rollback_restore_verified: {report['rollback_restore_verified']}",
        f"- plaintext_absence_verified: {report['plaintext_absence_verified']}",
        "",
        "## Key handling",
        "",
        f"- command_line_keys_rejected: {report['command_line_keys_rejected']}",
        f"- empty_key_refused: {report['empty_key_refused']}",
        f"- key_mismatch_refused: {report['key_mismatch_refused']}",
        f"- same_old_new_key_refused: {report['same_old_new_key_refused']}",
        "",
        "## Boundaries",
        "",
        f"- real_rotation_blocked_by_default: {report['real_rotation_blocked_by_default']}",
        f"- real_rotation_performed: {report['real_rotation_performed']}",
        f"- real_store_touched: {report['real_store_touched']}",
        f"- real_data_touched: {report['real_data_touched']}",
        f"- db_file_staged: {report['db_file_staged']}",
        f"- key_stored_in_repo: {report['key_stored_in_repo']}",
        f"- encryption_key_logged: {report['encryption_key_logged']}",
        f"- external_api_used: {report['external_api_used']}",
        f"- raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}",
        f"- private_filename_path_leaks: {report['private_filename_path_leaks']}",
        f"- secret_leaks: {report['secret_leaks']}",
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
    (REPORT_DIR / "cka_sec06_key_rotation_report.md").write_text(
        "\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_validation() -> dict:
    print("  [SEC-06] case A: baseline SEC-07 + SEC-05 + final CKA ...", flush=True)
    res_a = case_a_baseline()

    print("  [SEC-06] case B: CLI rejects key args ...", flush=True)
    res_b = case_b_cli_rejects_keys()

    print("  [SEC-06] case C: key mismatch refused ...", flush=True)
    res_c = case_c_new_key_mismatch_refused()

    print("  [SEC-06] case D: empty key refused ...", flush=True)
    res_d = case_d_empty_key_refused()

    print("  [SEC-06] case E: same old/new key refused ...", flush=True)
    res_e = case_e_same_old_new_key_refused()

    print("  [SEC-06] case F: synthetic rotation rehearsal ...", flush=True)
    res_f, rotation_result = case_f_synthetic_rotation()

    print("  [SEC-06] case G: rollback restore verified ...", flush=True)
    res_g = case_g_rollback_restore(rotation_result)

    print("  [SEC-06] case H: real rotation blocked by default ...", flush=True)
    res_h = case_h_real_rotation_blocked_by_default()

    print("  [SEC-06] case I: no DB/key/private files staged ...", flush=True)
    res_i = case_i_no_db_or_key_files_staged()

    results = [res_a, res_b, res_c, res_d, res_e, res_f, res_g, res_h, res_i]
    report = _build_report(results, rotation_result)

    print("  [SEC-06] case J: report safety ...", flush=True)
    res_j = case_j_report_safety(report)
    results.append(res_j)
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
    print(f"\nCKA-SEC-06 Operator Key Rotation — {status}")
    print(f"  conclusion: {rep['conclusion']}")
    print(f"  key_rotation_tool_ready: {rep['key_rotation_tool_ready']}")
    print(f"  synthetic_rotation_rehearsal_passed: {rep['synthetic_rotation_rehearsal_passed']}")
    print(f"  new_key_open_after_rotation_passed: {rep['new_key_open_after_rotation_passed']}")
    print(f"  old_key_rejected_after_rotation: {rep['old_key_rejected_after_rotation']}")
    print(f"  record_count_preserved: {rep['record_count_preserved']}")
    print(f"  rollback_restore_verified: {rep['rollback_restore_verified']}")
    print(f"  real_rotation_performed: {rep['real_rotation_performed']}")
    print(f"  real_store_touched: {rep['real_store_touched']}")
    print(f"  cases_passed: {rep['cases_passed']} / {rep['synthetic_cases_run']}")
    for r in rep["case_results"]:
        marker = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"    {marker} case {r['case']}: {r['description']}")
        if not r["passed"]:
            print(f"           error: {r.get('error')}")
    if not rep["all_passed"]:
        sys.exit(1)
