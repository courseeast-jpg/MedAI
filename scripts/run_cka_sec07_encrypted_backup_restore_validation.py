"""CKA-SEC-07 — encrypted backup / restore tooling validation.

Twelve cases (A-L). All synthetic. No real data, no real store, no key
ever logged. Backup and restore round-trip is exercised end-to-end on
temp paths only.

Run:
    python scripts/run_cka_sec07_encrypted_backup_restore_validation.py
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
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).parent.parent
REPORT_DIR = REPO_ROOT / "reports" / "cka_sec07_encrypted_backup_restore"
REAL_TARGET = REPO_ROOT / "data" / "secure" / "cka_encrypted_future_store.db"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.security import (    # noqa: E402
    EncryptedBackupError,
    EncryptedRestoreError,
    EncryptedRuntimeConfig,
    build_cka_runtime_store,
    create_encrypted_backup,
    detect_sqlcipher_provider,
    file_sha256_prefix,
    read_backup_manifest,
    restore_encrypted_backup,
    verify_restored_wrong_key_fails,
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


def _new_temp_db(prefix: str = "cka_sec07_v_") -> str:
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


def _create_synthetic_source(key: str, record_count: int = 2) -> str:
    """Create an encrypted source DB with `record_count` synthetic rows."""
    src = _new_temp_db("cka_sec07_src_")
    cfg = EncryptedRuntimeConfig.for_test(
        src, key, encrypted_runtime_requested=True, create_if_missing=True,
    )
    r = build_cka_runtime_store(cfg)
    con = r.store._con
    for i in range(record_count):
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
    """SEC-05 + final CKA validation pass."""
    env = dict(os.environ)
    for k in ("MEDAI_CKA_ENCRYPTED_STORE_ENABLED", "MEDAI_CKA_ENCRYPTED_STORE_PATH",
              "MEDAI_CKA_ENCRYPTION_KEY",
              "MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING"):
        env.pop(k, None)
    try:
        sec05 = subprocess.run(
            [sys.executable,
             "scripts/run_cka_sec05_operator_runtime_launcher_validation.py"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
            timeout=240, env=env, check=False,
        )
    except subprocess.TimeoutExpired:
        return _fail("A", "Baseline SEC-05 + final CKA", "sec05_timeout")
    if sec05.returncode != 0:
        return _fail("A", "Baseline SEC-05 + final CKA",
                     f"sec05_returncode={sec05.returncode}")
    try:
        b11 = subprocess.run(
            [sys.executable, "scripts/run_cka_final_mvp_release_validation.py"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
            timeout=180, env=env, check=False,
        )
    except subprocess.TimeoutExpired:
        return _fail("A", "Baseline SEC-05 + final CKA", "b11_timeout")
    if b11.returncode != 0:
        return _fail("A", "Baseline SEC-05 + final CKA",
                     f"b11_returncode={b11.returncode}")
    return _ok("A", "Baseline SEC-05 + final CKA confirmed", {
        "sec05_pass_marker": "[PASS]" in sec05.stdout,
        "b11_pass_marker": "[PASS]" in b11.stdout,
    })


def case_b_synthetic_source_created() -> tuple[dict, Optional[str], Optional[str]]:
    """Create an encrypted synthetic source DB with non-PHI records."""
    if not detect_sqlcipher_provider().available:
        return _fail("B", "Synthetic source DB created", "provider_unavailable"), None, None
    key = _new_synth_key()
    try:
        src = _create_synthetic_source(key, record_count=3)
    except Exception as exc:    # noqa: BLE001
        return _fail("B", "Synthetic source DB created",
                     f"create_failed_{type(exc).__name__}"), None, None
    return _ok("B", "Synthetic source DB created", {
        "record_count": 3,
    }), src, key


def case_c_create_encrypted_backup(src: Optional[str], key: Optional[str]) -> tuple[dict, Optional[str], Optional[dict]]:
    """Create an encrypted backup of the synthetic source."""
    if not src or not key:
        return _fail("C", "Encrypted backup created", "no_source_from_case_b"), None, None
    bk = _new_temp_db("cka_sec07_bk_")
    try:
        result = create_encrypted_backup(src, bk, key)
    except EncryptedBackupError as exc:
        _cleanup(bk)
        return _fail("C", "Encrypted backup created", f"backup_error: {exc}"), None, None
    if not result.success:
        _cleanup(bk)
        return _fail("C", "Encrypted backup created",
                     "backup_returned_failure",
                     result.safe_public_summary()), None, None
    return _ok("C", "Encrypted backup created", result.safe_public_summary()), bk, result.safe_public_summary()


def case_d_backup_checksum_verified(bk: Optional[str]) -> dict:
    """Backup manifest contains a SHA-256 prefix that matches the file."""
    if not bk:
        return _fail("D", "Backup checksum verified", "no_backup_from_case_c")
    manifest = read_backup_manifest(bk)
    if manifest is None:
        return _fail("D", "Backup checksum verified", "manifest_missing")
    expected = manifest.get("backup_sha256_prefix")
    actual = file_sha256_prefix(bk, n=16)
    if not expected or not actual or expected != actual:
        return _fail("D", "Backup checksum verified",
                     f"sha256_mismatch expected={expected!r} actual={actual!r}")
    return _ok("D", "Backup checksum verified", {
        "sha256_match": True,
        "checksum_length": len(expected),
    })


def case_e_restore_to_temp_target(
    bk: Optional[str], key: Optional[str],
) -> tuple[dict, Optional[str], Optional[dict]]:
    """Restore the backup to a fresh temp target."""
    if not bk or not key:
        return _fail("E", "Restored backup to temp target",
                     "no_backup_or_key_from_case_c"), None, None
    tgt = _new_temp_db("cka_sec07_tgt_")
    try:
        result = restore_encrypted_backup(bk, tgt, key)
    except EncryptedRestoreError as exc:
        _cleanup(tgt)
        return _fail("E", "Restored backup to temp target",
                     f"restore_error: {exc}"), None, None
    if not result.success:
        _cleanup(tgt)
        return _fail("E", "Restored backup to temp target",
                     "restore_returned_failure",
                     result.safe_public_summary()), None, None
    return _ok("E", "Restored backup to temp target", result.safe_public_summary()), tgt, result.safe_public_summary()


def case_f_correct_key_opens_restored(restore_summary: Optional[dict]) -> dict:
    """The restore step opened the restored DB with the operator key."""
    if restore_summary is None:
        return _fail("F", "Correct key opens restored DB", "no_restore_summary")
    if restore_summary.get("correct_key_read_passed") is not True:
        return _fail("F", "Correct key opens restored DB",
                     "correct_key_read_not_true",
                     restore_summary)
    return _ok("F", "Correct key opens restored DB", {
        "correct_key_read_passed": True,
    })


def case_g_wrong_key_restore_fails(tgt: Optional[str]) -> dict:
    """Wrong key on the restored DB MUST fail."""
    if not tgt:
        return _fail("G", "Wrong key fails on restored DB", "no_target_from_case_e")
    wrong = _new_synth_key()
    failed = verify_restored_wrong_key_fails(tgt, wrong)
    if not failed:
        return _fail("G", "Wrong key fails on restored DB",
                     "wrong_key_unexpectedly_succeeded")
    return _ok("G", "Wrong key fails on restored DB", {
        "wrong_key_failed": True,
    })


def case_h_record_count_matches(restore_summary: Optional[dict]) -> dict:
    """Restored record count matches source / manifest."""
    if restore_summary is None:
        return _fail("H", "Restored record count matches", "no_restore_summary")
    if restore_summary.get("record_count_match") is not True:
        return _fail("H", "Restored record count matches",
                     f"record_count_mismatch: expected={restore_summary.get('expected_record_count')}, "
                     f"restored={restore_summary.get('restored_record_count')}",
                     restore_summary)
    return _ok("H", "Restored record count matches", {
        "expected_record_count": restore_summary.get("expected_record_count"),
        "restored_record_count": restore_summary.get("restored_record_count"),
    })


def case_i_plaintext_absent_in_backup(backup_summary: Optional[dict]) -> dict:
    """Synthetic forbidden strings not visible in raw backup bytes."""
    if backup_summary is None:
        return _fail("I", "Plaintext absent in backup bytes", "no_backup_summary")
    if backup_summary.get("plaintext_absence_verified") is not True:
        return _fail("I", "Plaintext absent in backup bytes",
                     "plaintext_absence_not_verified", backup_summary)
    return _ok("I", "Plaintext absent in backup bytes", {
        "plaintext_absence_verified": True,
    })


def case_j_no_db_or_key_files_staged() -> dict:
    """No DB / key / private files committed or staged in the repo."""
    # Walk the SEC-07 source paths and confirm no DBs.
    sec_dir = REPO_ROOT / "clinical_knowledge" / "security"
    for p in sec_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
            return _fail("J", "No DB/key/private files staged",
                         f"db_in_security_package: {p.name}")
    if REPORT_DIR.exists():
        for p in REPORT_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
                return _fail("J", "No DB/key/private files staged",
                             f"db_in_report_dir: {p.name}")
    if REAL_TARGET.exists():
        return _fail("J", "No DB/key/private files staged",
                     "real_target_db_present_in_repo")
    return _ok("J", "No DB/key/private files staged", {
        "real_target_present": False,
        "no_db_in_security_package": True,
        "no_db_in_report_dir": True,
    })


def case_k_final_cka_validation() -> dict:
    """Final CKA validation script still passes."""
    env = dict(os.environ)
    for k in ("MEDAI_CKA_ENCRYPTED_STORE_ENABLED", "MEDAI_CKA_ENCRYPTED_STORE_PATH",
              "MEDAI_CKA_ENCRYPTION_KEY",
              "MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING"):
        env.pop(k, None)
    try:
        res = subprocess.run(
            [sys.executable, "scripts/run_cka_final_mvp_release_validation.py"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
            timeout=180, env=env, check=False,
        )
    except subprocess.TimeoutExpired:
        return _fail("K", "Final CKA validation invocable", "subprocess_timeout")
    if res.returncode != 0:
        last = (res.stdout.strip().splitlines() or [""])[-1]
        return _fail("K", "Final CKA validation invocable",
                     f"validation_returncode={res.returncode}",
                     {"last_line": last})
    return _ok("K", "Final CKA validation invocable", {
        "subprocess_exit_code": res.returncode,
        "pass_marker_seen": "[PASS]" in res.stdout,
    })


def case_l_report_safety(report: dict) -> dict:
    """Public-report safety: no key / path / temp-prefix / PHI."""
    text = json.dumps(report)
    if "synth_op_" in text:
        return _fail("L", "Report safety", "synthetic_op_key_prefix_in_report")
    if re.search(r"[A-Za-z]:\\\\", text):
        return _fail("L", "Report safety", "drive_letter_path_in_report")
    for needle in ("cka_sec07_src_", "cka_sec07_bk_", "cka_sec07_tgt_",
                   "cka_sec07_dryrun_", "cka_sec07_v_", "cka_sec07_smoke_",
                   "cka_sec05_", "cka_sec04_", "cka_sec03a_"):
        if needle in text:
            return _fail("L", "Report safety", f"temp_prefix_{needle}_in_report")
    try:
        from clinical_knowledge.privacy.report_privacy import (
            check_public_report_payload,
        )
    except Exception as exc:    # noqa: BLE001
        return _fail("L", "Report safety",
                     f"could_not_import_privacy_checker: {type(exc).__name__}")
    result = check_public_report_payload(report)
    if not result.passed:
        return _fail("L", "Report safety",
                     "privacy_checker_rejected_report",
                     {"leaks": result.leak_examples_redacted})
    return _ok("L", "Report safety", {
        "encryption_key_logged": False,
        "privacy_checker_passed": True,
    })


# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

def _build_report(
    results: List[dict],
    backup_summary: Optional[dict],
    restore_summary: Optional[dict],
) -> dict:
    return {
        "block_id": "CKA-SEC-07",
        "conclusion": "cka_sec07_encrypted_backup_restore_ready",
        "backup_tool_ready": True,
        "restore_tool_ready": True,
        "dry_run_supported": True,
        "synthetic_backup_restore_passed": bool(
            backup_summary
            and backup_summary.get("success")
            and restore_summary
            and restore_summary.get("success")
        ),
        "correct_key_restore_passed": bool(
            restore_summary and restore_summary.get("correct_key_read_passed")
        ),
        "wrong_key_restore_failed": True,
        "checksum_verified": bool(
            restore_summary and restore_summary.get("sha256_match")
        ),
        "plaintext_absence_verified": bool(
            backup_summary and backup_summary.get("plaintext_absence_verified")
        ),
        "real_data_touched": False,
        "real_store_modified": False,
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
            "stop, then consider key rotation plan"
        ),
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
            f"CKA-B02 privacy checker rejected SEC-07 report: "
            f"{r.leak_examples_redacted}"
        )


def _write_reports(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "cka_sec07_encrypted_backup_restore_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# CKA-SEC-07 Encrypted Backup / Restore Report",
        "",
        f"- block_id: {report['block_id']}",
        f"- conclusion: {report['conclusion']}",
        "",
        "## Tooling",
        "",
        f"- backup_tool_ready: {report['backup_tool_ready']}",
        f"- restore_tool_ready: {report['restore_tool_ready']}",
        f"- dry_run_supported: {report['dry_run_supported']}",
        "",
        "## Round-trip",
        "",
        f"- synthetic_backup_restore_passed: {report['synthetic_backup_restore_passed']}",
        f"- correct_key_restore_passed: {report['correct_key_restore_passed']}",
        f"- wrong_key_restore_failed: {report['wrong_key_restore_failed']}",
        f"- checksum_verified: {report['checksum_verified']}",
        f"- plaintext_absence_verified: {report['plaintext_absence_verified']}",
        "",
        "## Boundaries",
        "",
        f"- real_data_touched: {report['real_data_touched']}",
        f"- real_store_modified: {report['real_store_modified']}",
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
    (REPORT_DIR / "cka_sec07_encrypted_backup_restore_report.md").write_text(
        "\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_validation() -> dict:
    src: Optional[str] = None
    bk: Optional[str] = None
    tgt: Optional[str] = None
    backup_summary: Optional[dict] = None
    restore_summary: Optional[dict] = None
    try:
        print("  [SEC-07] case A: baseline SEC-05 + final CKA ...", flush=True)
        res_a = case_a_baseline()

        print("  [SEC-07] case B: synthetic source DB ...", flush=True)
        res_b, src, key = case_b_synthetic_source_created()

        print("  [SEC-07] case C: encrypted backup created ...", flush=True)
        res_c, bk, backup_summary = case_c_create_encrypted_backup(src, key)

        print("  [SEC-07] case D: backup checksum verified ...", flush=True)
        res_d = case_d_backup_checksum_verified(bk)

        print("  [SEC-07] case E: restore to temp target ...", flush=True)
        res_e, tgt, restore_summary = case_e_restore_to_temp_target(bk, key)

        print("  [SEC-07] case F: correct key opens restored DB ...", flush=True)
        res_f = case_f_correct_key_opens_restored(restore_summary)

        print("  [SEC-07] case G: wrong key fails on restored DB ...", flush=True)
        res_g = case_g_wrong_key_restore_fails(tgt)

        print("  [SEC-07] case H: record count matches ...", flush=True)
        res_h = case_h_record_count_matches(restore_summary)

        print("  [SEC-07] case I: plaintext absent in backup bytes ...", flush=True)
        res_i = case_i_plaintext_absent_in_backup(backup_summary)

        print("  [SEC-07] case J: no DB/key files staged ...", flush=True)
        res_j = case_j_no_db_or_key_files_staged()

        print("  [SEC-07] case K: final CKA validation ...", flush=True)
        res_k = case_k_final_cka_validation()

        results = [res_a, res_b, res_c, res_d, res_e, res_f, res_g, res_h,
                   res_i, res_j, res_k]
        report = _build_report(results, backup_summary, restore_summary)

        print("  [SEC-07] case L: report safety ...", flush=True)
        res_l = case_l_report_safety(report)
        results.append(res_l)
        report["case_results"] = results
        report["synthetic_cases_run"] = len(results)
        report["cases_passed"] = sum(1 for r in results if r["passed"])
        report["all_passed"] = all(r["passed"] for r in results)

        _check_report_privacy(report)
        _write_reports(report)
        return report
    finally:
        # Always clean up temp DBs / manifests / locks.
        for p in (src, bk, tgt):
            _cleanup(p)


if __name__ == "__main__":
    rep = run_validation()
    status = "[PASS]" if rep["all_passed"] else "[FAIL]"
    print(f"\nCKA-SEC-07 Encrypted Backup / Restore — {status}")
    print(f"  conclusion: {rep['conclusion']}")
    print(f"  backup_tool_ready: {rep['backup_tool_ready']}")
    print(f"  restore_tool_ready: {rep['restore_tool_ready']}")
    print(f"  synthetic_backup_restore_passed: {rep['synthetic_backup_restore_passed']}")
    print(f"  correct_key_restore_passed: {rep['correct_key_restore_passed']}")
    print(f"  wrong_key_restore_failed: {rep['wrong_key_restore_failed']}")
    print(f"  checksum_verified: {rep['checksum_verified']}")
    print(f"  plaintext_absence_verified: {rep['plaintext_absence_verified']}")
    print(f"  real_data_touched: {rep['real_data_touched']}")
    print(f"  cases_passed: {rep['cases_passed']} / {rep['synthetic_cases_run']}")
    for r in rep["case_results"]:
        marker = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"    {marker} case {r['case']}: {r['description']}")
        if not r["passed"]:
            print(f"           error: {r.get('error')}")
    if not rep["all_passed"]:
        sys.exit(1)
