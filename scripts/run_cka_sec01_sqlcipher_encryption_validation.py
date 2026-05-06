"""CKA-SEC-01 — SQLCipher encryption readiness validation.

Eight cases (A-H). All cases use synthetic test data only.
- No real PHI.
- No real connectors.
- No real data migration.
- The encryption key is never logged or written to a public report.

Run:
    python scripts/run_cka_sec01_sqlcipher_encryption_validation.py
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
REPORT_DIR = REPO_ROOT / "reports" / "cka_sec01_sqlcipher_encryption"

# Keep the repo importable when this script runs as a path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.security import (    # noqa: E402  (after sys.path insert)
    EncryptedCKAStore,
    EncryptedStoreError,
    detect_sqlcipher_provider,
    verify_plaintext_absent,
    verify_wrong_key_fails,
)
from clinical_knowledge.security.encryption_checks import (    # noqa: E402
    SYNTHETIC_FORBIDDEN_STRINGS,
)


SKIP_REASON_PROVIDER_UNAVAILABLE = "skipped_provider_unavailable"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(case: str, desc: str, details: Optional[dict] = None) -> dict:
    return {"case": case, "description": desc, "passed": True,
            "skipped": False, "details": details or {}}


def _skip(case: str, desc: str, reason: str, details: Optional[dict] = None) -> dict:
    return {"case": case, "description": desc, "passed": True,
            "skipped": True, "skip_reason": reason, "details": details or {}}


def _fail(case: str, desc: str, error: str, details: Optional[dict] = None) -> dict:
    return {"case": case, "description": desc, "passed": False,
            "skipped": False, "error": error, "details": details or {}}


def _new_synthetic_key() -> str:
    """Generate a synthetic test key — NEVER logged."""
    return "synthkey_" + secrets.token_hex(16)


def _temp_db_path() -> str:
    fd, path = tempfile.mkstemp(prefix="cka_sec01_synth_", suffix=".db")
    os.close(fd)
    # Remove the empty file so SQLCipher creates a fresh, encrypted one.
    try:
        os.unlink(path)
    except OSError:
        pass
    return path


def _safe_unlink(path: Optional[str]) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_a_provider_detection() -> dict:
    """Provider detection returns a safe status without crashing."""
    status = detect_sqlcipher_provider()
    summary = status.safe_public_summary()
    # Ensure summary contains no path-shaped values.
    for k, v in summary.items():
        if isinstance(v, str) and ("\\" in v or "/" in v):
            return _fail("A", "Provider detection",
                         f"path-like value in summary[{k!r}]")
    return _ok("A", "Provider detection", {
        "provider_available": summary["available"],
        "provider_name": summary["provider_name"],
        "cipher_version_available": bool(summary["cipher_version"]),
        "notes": summary["notes"],
    })


def case_b_empty_key_refused() -> dict:
    """Adapter refuses an empty encryption key."""
    try:
        EncryptedCKAStore(":memory:", "")
        return _fail("B", "Empty key refused", "no error raised")
    except EncryptedStoreError as exc:
        if str(exc) != "empty_encryption_key_refused":
            return _fail("B", "Empty key refused",
                         f"unexpected error: {exc}")
        return _ok("B", "Empty key refused", {"error_marker": str(exc)})


def case_c_synthetic_encrypted_db_create(
    provider_available: bool,
) -> Tuple[dict, Optional[str], Optional[str]]:
    """Create encrypted synthetic DB, insert, read with correct key."""
    if not provider_available:
        return _skip("C", "Synthetic encrypted DB create",
                     SKIP_REASON_PROVIDER_UNAVAILABLE), None, None

    db = _temp_db_path()
    key = _new_synthetic_key()
    try:
        with EncryptedCKAStore(db, key) as store:
            store.insert_synthetic_record(
                "rec_001",
                "label_alpha",
                "SYNTHETIC_PRIVATE_NAME_ALPHA payload SYNTHETIC_MRN_0001",
            )
            store.insert_synthetic_record(
                "rec_002",
                "label_beta",
                "SYNTHETIC_MEDICAL_NOTE_ALPHA second payload",
            )
            rows = store.fetch_all_synthetic()
        if len(rows) != 2:
            return (
                _fail("C", "Synthetic encrypted DB create",
                      f"expected 2 rows, got {len(rows)}"),
                db, key,
            )
        return (
            _ok("C", "Synthetic encrypted DB create", {
                "rows_inserted": 2,
                "correct_key_read_passed": True,
            }),
            db, key,
        )
    except EncryptedStoreError as exc:
        return _fail("C", "Synthetic encrypted DB create", str(exc)), db, key
    except Exception as exc:    # noqa: BLE001
        return _fail("C", "Synthetic encrypted DB create",
                     f"{type(exc).__name__}"), db, key


def case_d_wrong_key_fails(
    provider_available: bool,
    db_path: Optional[str],
    correct_key: Optional[str],
) -> dict:
    """Wrong key cannot read the synthetic encrypted DB."""
    if not provider_available:
        return _skip("D", "Wrong key fails", SKIP_REASON_PROVIDER_UNAVAILABLE)
    if not db_path:
        return _fail("D", "Wrong key fails", "no encrypted db_path from case C")
    wrong_key = _new_synthetic_key()
    if correct_key is not None and wrong_key == correct_key:
        wrong_key += "_x"   # cannot be equal
    failed = verify_wrong_key_fails(db_path, wrong_key)
    if not failed:
        return _fail("D", "Wrong key fails", "wrong key DECRYPTED the db")
    return _ok("D", "Wrong key fails", {"wrong_key_read_failed": True})


def case_e_plaintext_absent(
    provider_available: bool,
    db_path: Optional[str],
) -> dict:
    """Synthetic forbidden strings must not appear in raw db bytes."""
    if not provider_available:
        return _skip("E", "Plaintext absent", SKIP_REASON_PROVIDER_UNAVAILABLE)
    if not db_path:
        return _fail("E", "Plaintext absent", "no encrypted db_path from case C")
    ok = verify_plaintext_absent(db_path, SYNTHETIC_FORBIDDEN_STRINGS)
    if not ok:
        return _fail("E", "Plaintext absent",
                     "forbidden synthetic string visible in raw db bytes")
    return _ok("E", "Plaintext absent", {"plaintext_absence_verified": True})


def case_f_main_store_untouched() -> dict:
    """Confirm we did NOT migrate the main CKA store or replace MKBStore."""
    # The main MKBStore class is intact.
    from clinical_knowledge.store import MKBStore
    if not isinstance(MKBStore, type):
        return _fail("F", "Main store untouched", "MKBStore not a class anymore")
    # The encrypted adapter is parallel — it is NOT MKBStore.
    if MKBStore is EncryptedCKAStore:
        return _fail("F", "Main store untouched",
                     "MKBStore was replaced by EncryptedCKAStore")
    # Confirm no SEC-01 module rebinds production CKA modules.
    import clinical_knowledge.store as core_store
    import clinical_knowledge.security as sec
    if getattr(core_store, "MKBStore") is not MKBStore:
        return _fail("F", "Main store untouched",
                     "core_store.MKBStore identity mutated")
    return _ok("F", "Main store untouched", {
        "main_store_migration_performed": False,
        "real_data_migrated": False,
        "sqlcipher_encryption_active_for_main_store": False,
    })


def case_g_final_cka_validation_invocable() -> dict:
    """Final CKA validation script can be invoked and still passes.

    Runs as a subprocess with CKA_B11_SKIP_NESTED_PYTEST=1 so the inner
    full-suite pytest is short-circuited (it already runs in the outer
    suite). Timeout-bounded.
    """
    env = dict(os.environ)
    env["CKA_B11_SKIP_NESTED_PYTEST"] = "1"
    try:
        res = subprocess.run(
            [sys.executable, "scripts/run_cka_final_mvp_release_validation.py"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True,
            timeout=120, env=env, check=False,
        )
    except subprocess.TimeoutExpired:
        return _fail("G", "Final CKA validation invocable",
                     "subprocess timed out (>120s)")

    if res.returncode != 0:
        # Capture only the last summary line (no paths/keys leak).
        last = (res.stdout.strip().splitlines() or [""])[-1]
        return _fail("G", "Final CKA validation invocable",
                     f"validation returncode={res.returncode}",
                     {"last_line": last})
    # Search for the PASS marker without copying full stdout into the report.
    pass_seen = "[PASS]" in res.stdout
    return _ok("G", "Final CKA validation invocable", {
        "subprocess_exit_code": res.returncode,
        "pass_marker_seen": pass_seen,
    })


def case_h_report_safety(report: dict) -> dict:
    """Public-report safety: no key, no PHI, no path, no secret."""
    text = json.dumps(report)
    # Reject anything that looks like the synthetic test key prefix
    # (keys are NEVER written to the report; this is defense-in-depth).
    if "synthkey_" in text:
        return _fail("H", "Report safety", "synthetic key string leaked into report")
    # Reject anything that looks like a Windows drive-letter path.
    if re.search(r"[A-Za-z]:\\\\", text):
        return _fail("H", "Report safety", "drive-letter path leaked into report")
    # Reject Unix tempdir hint
    for needle in ("/tmp/cka_sec01_synth_", "Temp\\\\cka_sec01_synth_"):
        if needle in text:
            return _fail("H", "Report safety", "temp db path leaked into report")
    # Run the CKA-B02 public report privacy checker.
    try:
        from clinical_knowledge.privacy.report_privacy import check_public_report_payload
    except Exception as exc:    # noqa: BLE001
        return _fail("H", "Report safety",
                     f"could not import privacy checker: {type(exc).__name__}")
    result = check_public_report_payload(report)
    if not result.passed:
        return _fail("H", "Report safety",
                     "privacy checker rejected report",
                     {"leaks": result.leak_examples_redacted})
    return _ok("H", "Report safety", {
        "encryption_key_logged": False,
        "privacy_checker_passed": True,
    })


# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------

def _build_report(
    results: List[dict],
    provider_summary: dict,
    synthetic_db_created: bool,
    correct_key_read_passed: Optional[bool],
    wrong_key_read_failed: Optional[bool],
    plaintext_absence_verified: Optional[bool],
) -> dict:
    provider_available = bool(provider_summary.get("available"))
    cipher_version_available = bool(provider_summary.get("cipher_version"))

    if (
        provider_available
        and synthetic_db_created
        and correct_key_read_passed is True
        and wrong_key_read_failed is True
        and plaintext_absence_verified is True
    ):
        conclusion = "cka_sec01_sqlcipher_encrypted_store_ready"
    else:
        conclusion = "cka_sec01_sqlcipher_provider_required"

    all_passed = all(r["passed"] for r in results)

    report = {
        "block_id": "CKA-SEC-01",
        "conclusion": conclusion,
        "provider_name": provider_summary.get("provider_name"),
        "provider_available": provider_available,
        "cipher_version_available": cipher_version_available,
        "synthetic_encrypted_store_created": synthetic_db_created,
        "correct_key_read_passed": bool(correct_key_read_passed) if correct_key_read_passed is not None else False,
        "wrong_key_read_failed": bool(wrong_key_read_failed) if wrong_key_read_failed is not None else False,
        "plaintext_absence_verified": bool(plaintext_absence_verified) if plaintext_absence_verified is not None else False,
        "sqlcipher_encryption_active_for_main_store": False,
        "main_store_migration_performed": False,
        "real_data_migrated": False,
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "private_filename_path_leaks": 0,
        "secret_leaks": 0,
        "encryption_key_logged": False,
        "clinical_recommendations_generated": False,
        "prescription_dosing_advice_generated": False,
        "production_ocr_changed": False,
        "production_extractor_changed": False,
        "safety_gate_changed": False,
        "frozen_hitl_release_reopened": False,
        "next_recommended_block": (
            "CKA-SEC-02 Main Store Migration Plan, only after provider is "
            "available and SEC-01 encrypted synthetic validation passes"
        ),
        "synthetic_cases_run": len(results),
        "cases_passed": sum(1 for r in results if r["passed"] and not r.get("skipped")),
        "cases_skipped": sum(1 for r in results if r.get("skipped")),
        "all_passed": all_passed,
        "case_results": results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return report


def _check_report_privacy(report: dict) -> None:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload
    result = check_public_report_payload(report)
    if not result.passed:
        raise RuntimeError(
            f"CKA-B02 privacy checker rejected SEC-01 report: "
            f"{result.leak_examples_redacted}"
        )


def _write_reports(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "cka_sec01_sqlcipher_encryption_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# CKA-SEC-01 SQLCipher Encryption Readiness Report",
        "",
        f"- block_id: {report['block_id']}",
        f"- conclusion: {report['conclusion']}",
        f"- provider_available: {report['provider_available']}",
        f"- provider_name: {report['provider_name']}",
        f"- cipher_version_available: {report['cipher_version_available']}",
        f"- synthetic_encrypted_store_created: {report['synthetic_encrypted_store_created']}",
        f"- correct_key_read_passed: {report['correct_key_read_passed']}",
        f"- wrong_key_read_failed: {report['wrong_key_read_failed']}",
        f"- plaintext_absence_verified: {report['plaintext_absence_verified']}",
        "",
        "## Main store boundary",
        "",
        f"- sqlcipher_encryption_active_for_main_store: {report['sqlcipher_encryption_active_for_main_store']}",
        f"- main_store_migration_performed: {report['main_store_migration_performed']}",
        f"- real_data_migrated: {report['real_data_migrated']}",
        "",
        "## Safety / privacy",
        "",
        f"- external_api_used: {report['external_api_used']}",
        f"- raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}",
        f"- private_filename_path_leaks: {report['private_filename_path_leaks']}",
        f"- secret_leaks: {report['secret_leaks']}",
        f"- encryption_key_logged: {report['encryption_key_logged']}",
        f"- clinical_recommendations_generated: {report['clinical_recommendations_generated']}",
        f"- prescription_dosing_advice_generated: {report['prescription_dosing_advice_generated']}",
        f"- production_ocr_changed: {report['production_ocr_changed']}",
        f"- production_extractor_changed: {report['production_extractor_changed']}",
        f"- safety_gate_changed: {report['safety_gate_changed']}",
        f"- frozen_hitl_release_reopened: {report['frozen_hitl_release_reopened']}",
        "",
        "## Case results",
        "",
    ]
    for r in report["case_results"]:
        if r.get("skipped"):
            md.append(f"- Case {r['case']}: [SKIP] {r['description']} ({r.get('skip_reason')})")
        elif r["passed"]:
            md.append(f"- Case {r['case']}: [PASS] {r['description']}")
        else:
            md.append(f"- Case {r['case']}: [FAIL] {r['description']}")
            md.append(f"    Error: {r.get('error', 'unknown')}")
    md += [
        "",
        "## Next recommended block",
        "",
        report["next_recommended_block"],
        "",
    ]
    md_path = REPORT_DIR / "cka_sec01_sqlcipher_encryption_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_validation() -> dict:
    results: List[dict] = []
    print("  [SEC-01] case A: provider detection ...", flush=True)
    res_a = case_a_provider_detection()
    results.append(res_a)
    provider_status = detect_sqlcipher_provider()
    provider_available = bool(provider_status.available)
    provider_summary = provider_status.safe_public_summary()
    print(f"    provider_available={provider_available}", flush=True)

    print("  [SEC-01] case B: empty key refused ...", flush=True)
    results.append(case_b_empty_key_refused())

    print("  [SEC-01] case C: synthetic encrypted DB create ...", flush=True)
    res_c, c_db, c_key = case_c_synthetic_encrypted_db_create(provider_available)
    results.append(res_c)

    print("  [SEC-01] case D: wrong key fails ...", flush=True)
    res_d = case_d_wrong_key_fails(provider_available, c_db, c_key)
    results.append(res_d)

    print("  [SEC-01] case E: plaintext absent ...", flush=True)
    res_e = case_e_plaintext_absent(provider_available, c_db)
    results.append(res_e)

    # Cleanup synthetic temp DB BEFORE reporting (no path leaks).
    _safe_unlink(c_db)
    c_db = None
    c_key = None

    print("  [SEC-01] case F: main store untouched ...", flush=True)
    results.append(case_f_main_store_untouched())

    print("  [SEC-01] case G: final CKA validation invocable ...", flush=True)
    results.append(case_g_final_cka_validation_invocable())

    synthetic_db_created = (provider_available and res_c.get("passed") and not res_c.get("skipped", False))
    correct_key_read_passed = (
        provider_available and res_c.get("passed")
        and not res_c.get("skipped", False)
        and bool(res_c.get("details", {}).get("correct_key_read_passed", False))
    )
    wrong_key_read_failed = (
        provider_available and res_d.get("passed")
        and not res_d.get("skipped", False)
        and bool(res_d.get("details", {}).get("wrong_key_read_failed", False))
    )
    plaintext_absence_verified = (
        provider_available and res_e.get("passed")
        and not res_e.get("skipped", False)
        and bool(res_e.get("details", {}).get("plaintext_absence_verified", False))
    )

    report = _build_report(
        results=results,
        provider_summary=provider_summary,
        synthetic_db_created=synthetic_db_created,
        correct_key_read_passed=correct_key_read_passed,
        wrong_key_read_failed=wrong_key_read_failed,
        plaintext_absence_verified=plaintext_absence_verified,
    )

    print("  [SEC-01] case H: report safety ...", flush=True)
    res_h = case_h_report_safety(report)
    results.append(res_h)
    # Re-build with H included.
    report["case_results"] = results
    report["synthetic_cases_run"] = len(results)
    report["cases_passed"] = sum(1 for r in results if r["passed"] and not r.get("skipped"))
    report["cases_skipped"] = sum(1 for r in results if r.get("skipped"))
    report["all_passed"] = all(r["passed"] for r in results)

    _check_report_privacy(report)
    _write_reports(report)
    return report


if __name__ == "__main__":
    rep = run_validation()
    status = "[PASS]" if rep["all_passed"] else "[FAIL]"
    print(f"\nCKA-SEC-01 SQLCipher Encryption Readiness — {status}")
    print(f"  conclusion: {rep['conclusion']}")
    print(f"  provider_available: {rep['provider_available']}")
    print(f"  provider_name: {rep['provider_name']}")
    print(f"  cipher_version_available: {rep['cipher_version_available']}")
    print(f"  synthetic_encrypted_store_created: {rep['synthetic_encrypted_store_created']}")
    print(f"  correct_key_read_passed: {rep['correct_key_read_passed']}")
    print(f"  wrong_key_read_failed: {rep['wrong_key_read_failed']}")
    print(f"  plaintext_absence_verified: {rep['plaintext_absence_verified']}")
    print(f"  sqlcipher_encryption_active_for_main_store: {rep['sqlcipher_encryption_active_for_main_store']}")
    print(f"  cases_passed: {rep['cases_passed']} / {rep['synthetic_cases_run']}"
          f" ({rep['cases_skipped']} skipped)")
    for r in rep["case_results"]:
        if r.get("skipped"):
            marker = "[SKIP]"
        elif r["passed"]:
            marker = "[PASS]"
        else:
            marker = "[FAIL]"
        print(f"    {marker} case {r['case']}: {r['description']}")
        if not r["passed"]:
            print(f"           error: {r.get('error')}")
    if not rep["all_passed"]:
        sys.exit(1)
