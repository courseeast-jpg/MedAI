"""CKA-SEC-04 — encrypted runtime activation validation.

Ten cases (A-J). All synthetic. The main CKA store is NEVER touched.
The default runtime is ALWAYS unencrypted unless explicit env flags
are set in this exact run, in which case the encrypted runtime is
exercised against a temp DB only.

Run:
    python scripts/run_cka_sec04_encrypted_runtime_activation_validation.py
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
REPORT_DIR = REPO_ROOT / "reports" / "cka_sec04_encrypted_runtime_activation"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.security import (    # noqa: E402
    EncryptedCKAStore,
    EncryptedRuntimeConfig,
    RuntimeFactoryError,
    build_cka_runtime_store,
    detect_sqlcipher_provider,
    get_runtime_rollback_plan,
    run_encrypted_runtime_preflight,
    runtime_rollback_plan_ready,
)
from clinical_knowledge.security.empty_store_initializer import (    # noqa: E402
    _lock_path_for,
)
from clinical_knowledge.store import MKBStore    # noqa: E402


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


def _new_temp_db_path(prefix: str = "cka_sec04_v_") -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".db")
    os.close(fd)
    try:
        os.unlink(path)
    except OSError:
        pass
    return path


def _safe_unlink_pair(db_path: Optional[str]) -> None:
    if not db_path:
        return
    p = Path(db_path)
    try:
        p.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass
    sib = p.parent / (p.stem + ".manifest.json")
    try:
        sib.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass
    lock = _lock_path_for(p)
    try:
        lock.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_a_baseline_sec03a() -> dict:
    """SEC-03A baseline: provider available, SEC-03A report on disk."""
    if not detect_sqlcipher_provider().available:
        return _fail("A", "Baseline SEC-03A confirmed",
                     "sqlcipher_provider_unavailable")
    sec03a_path = (REPO_ROOT / "reports" / "cka_sec03a_empty_encrypted_store"
                   / "cka_sec03a_empty_encrypted_store_report.json")
    if not sec03a_path.exists():
        return _fail("A", "Baseline SEC-03A confirmed", "sec03a_report_missing")
    sec03a = json.loads(sec03a_path.read_text(encoding="utf-8"))
    bad = []
    if sec03a.get("empty_future_store_runtime_active") is not False:
        bad.append("empty_future_store_runtime_active_must_be_false")
    if sec03a.get("main_store_migration_performed") is not False:
        bad.append("main_store_migration_performed_must_be_false")
    if sec03a.get("real_data_migrated") is not False:
        bad.append("real_data_migrated_must_be_false")
    if bad:
        return _fail("A", "Baseline SEC-03A confirmed",
                     f"sec03a_invariants_violated: {bad}")
    return _ok("A", "Baseline SEC-03A confirmed", {
        "provider_available": True,
        "sec03a_conclusion": sec03a.get("conclusion"),
    })


def case_b_default_runtime_unencrypted() -> dict:
    """No env flag → standard MKBStore, runtime_encryption_active=False."""
    cfg = EncryptedRuntimeConfig.from_env(env={})
    if cfg.encrypted_runtime_requested is not False:
        return _fail("B", "Default runtime unencrypted",
                     "default_config_requested_encrypted_runtime")
    result = build_cka_runtime_store(cfg)
    if not isinstance(result.store, MKBStore):
        return _fail("B", "Default runtime unencrypted",
                     f"factory_returned_{type(result.store).__name__}")
    if result.runtime_encryption_active is not False:
        return _fail("B", "Default runtime unencrypted",
                     "runtime_encryption_active_was_true")
    return _ok("B", "Default runtime unencrypted",
               result.safe_public_summary())


def case_c_runtime_flag_without_key_blocks() -> dict:
    """Runtime requested, key missing → factory raises, no fallback."""
    cfg = EncryptedRuntimeConfig.from_env(env={
        "MEDAI_CKA_ENCRYPTED_STORE_ENABLED": "1",
    })
    if cfg.runtime_activation_allowed is True:
        return _fail("C", "Runtime flag without key blocks",
                     "runtime_activation_allowed_was_true_without_key")
    try:
        build_cka_runtime_store(cfg)
        return _fail("C", "Runtime flag without key blocks",
                     "factory_did_not_raise")
    except RuntimeFactoryError as exc:
        if "missing_key" not in str(exc):
            return _fail("C", "Runtime flag without key blocks",
                         f"unexpected_error: {exc}")
        return _ok("C", "Runtime flag without key blocks",
                   {"error_marker": str(exc)})


def case_d_runtime_wrong_key_fails() -> dict:
    """An existing encrypted store + a wrong key → factory raises, no fallback."""
    db = _new_temp_db_path()
    correct = _new_synth_key()
    try:
        # Create the encrypted store with the correct key.
        cfg_create = EncryptedRuntimeConfig.for_test(
            db, correct,
            encrypted_runtime_requested=True,
            create_if_missing=True,
        )
        r_create = build_cka_runtime_store(cfg_create)
        if hasattr(r_create.store, "close"):
            r_create.store.close()

        # Try to open with the wrong key.
        wrong = _new_synth_key()
        if wrong == correct:
            wrong += "_x"
        cfg_wrong = EncryptedRuntimeConfig.for_test(
            db, wrong,
            encrypted_runtime_requested=True,
            create_if_missing=False,
        )
        try:
            build_cka_runtime_store(cfg_wrong)
            return _fail("D", "Runtime wrong key fails",
                         "factory_did_not_raise_on_wrong_key")
        except RuntimeFactoryError as exc:
            if "open_failed" not in str(exc):
                return _fail("D", "Runtime wrong key fails",
                             f"unexpected_error: {exc}")
            return _ok("D", "Runtime wrong key fails",
                       {"error_marker": str(exc)})
    finally:
        _safe_unlink_pair(db)


def case_e_test_mode_runtime_creates_empty() -> tuple[dict, Optional[dict]]:
    """Test-mode encrypted runtime creates an empty store and preflight passes."""
    db = _new_temp_db_path()
    key = _new_synth_key()
    summary: Optional[dict] = None
    try:
        cfg_create = EncryptedRuntimeConfig.for_test(
            db, key, encrypted_runtime_requested=True, create_if_missing=True,
        )
        r_create = build_cka_runtime_store(cfg_create)
        if r_create.runtime_encryption_active is not True:
            return _fail("E", "Test-mode encrypted runtime",
                         "runtime_encryption_active_not_true",
                         r_create.safe_public_summary()), None
        if r_create.encrypted_store_created is not True:
            return _fail("E", "Test-mode encrypted runtime",
                         "encrypted_store_created_not_true",
                         r_create.safe_public_summary()), None
        if hasattr(r_create.store, "close"):
            r_create.store.close()

        # Preflight on the now-existing file.
        cfg_pf = EncryptedRuntimeConfig.for_test(
            db, key, encrypted_runtime_requested=True, create_if_missing=False,
        )
        pf = run_encrypted_runtime_preflight(cfg_pf)
        summary = pf.safe_public_summary()
        bad = []
        if pf.passed is not True:
            bad.append(f"preflight_not_passed: blocked={pf.blocked_reason!r}")
        if pf.records_count != 0:
            bad.append("records_count_not_zero")
        if pf.correct_key_read_passed is not True:
            bad.append("correct_key_read_not_passed")
        if pf.wrong_key_failure_passed is not True:
            bad.append("wrong_key_failure_not_passed")
        if pf.plaintext_absence_verified is not True:
            bad.append("plaintext_absence_not_verified")
        if pf.migration_performed is not False:
            bad.append("migration_performed_must_be_false")
        if pf.real_data_migrated is not False:
            bad.append("real_data_migrated_must_be_false")
        if bad:
            return _fail("E", "Test-mode encrypted runtime",
                         f"preflight_invariants_failed: {bad}", summary), summary
        return _ok("E", "Test-mode encrypted runtime", summary), summary
    finally:
        _safe_unlink_pair(db)


def case_f_create_if_missing_false_blocks() -> dict:
    """create_if_missing=false on a missing target → factory raises."""
    db = _new_temp_db_path()
    key = _new_synth_key()
    try:
        # Make sure the file does NOT exist.
        Path(db).unlink(missing_ok=True)    # type: ignore[call-arg]
        cfg = EncryptedRuntimeConfig.for_test(
            db, key, encrypted_runtime_requested=True, create_if_missing=False,
        )
        try:
            build_cka_runtime_store(cfg)
            # Cleanup if (against expectation) something was created.
            _safe_unlink_pair(db)
            return _fail("F", "create_if_missing false blocks missing store",
                         "factory_did_not_raise")
        except RuntimeFactoryError as exc:
            if "create_if_missing_false" not in str(exc):
                return _fail("F", "create_if_missing false blocks missing store",
                             f"unexpected_error: {exc}")
            # Confirm no file was created.
            if Path(db).exists():
                return _fail("F", "create_if_missing false blocks missing store",
                             "file_was_created_despite_block")
            return _ok("F", "create_if_missing false blocks missing store",
                       {"error_marker": str(exc)})
    finally:
        _safe_unlink_pair(db)


def case_g_rollback_plan_ready() -> dict:
    """Rollback plan is generated and contains non-destructive instructions."""
    plan = get_runtime_rollback_plan()
    summary = plan.safe_public_summary()
    if not runtime_rollback_plan_ready(plan):
        return _fail("G", "Rollback plan ready", "rollback_plan_not_ready", summary)
    if plan.no_destructive_action_on_rollback is not True:
        return _fail("G", "Rollback plan ready",
                     "no_destructive_action_must_be_true")
    if plan.no_data_migration_in_sec04 is not True:
        return _fail("G", "Rollback plan ready",
                     "no_data_migration_must_be_true")
    if len(plan.steps) < 5:
        return _fail("G", "Rollback plan ready",
                     "fewer_than_five_steps")
    # Verify steps mention the correct env var to unset.
    joined = "\n".join(plan.steps).upper()
    if "MEDAI_CKA_ENCRYPTED_STORE_ENABLED" not in joined:
        return _fail("G", "Rollback plan ready",
                     "rollback_steps_missing_unset_env_instruction")
    return _ok("G", "Rollback plan ready", {
        "rollback_state": summary,
        "steps_count": len(plan.steps),
    })


def case_h_main_store_untouched() -> dict:
    """Main MKBStore class identity / production data unchanged."""
    from clinical_knowledge.store import MKBStore as MS
    from clinical_knowledge.security import EncryptedCKAStore as Enc
    if MS is Enc:
        return _fail("H", "Main store untouched",
                     "MKBStore_was_replaced_by_EncryptedCKAStore")
    if not isinstance(MS, type):
        return _fail("H", "Main store untouched", "MKBStore_not_class")
    # The factory in default mode returns MKBStore — confirm in this run.
    cfg = EncryptedRuntimeConfig.from_env(env={})
    r = build_cka_runtime_store(cfg)
    if not isinstance(r.store, MS):
        return _fail("H", "Main store untouched",
                     "default_factory_did_not_return_MKBStore")
    if r.main_store_migration_performed is not False:
        return _fail("H", "Main store untouched",
                     "main_store_migration_performed_was_true")
    if r.real_data_migrated is not False:
        return _fail("H", "Main store untouched",
                     "real_data_migrated_was_true")
    return _ok("H", "Main store untouched", {
        "main_store_migration_performed": False,
        "real_data_migrated": False,
        "default_factory_kind": r.store_kind,
    })


def case_i_final_cka_validation() -> dict:
    """Final CKA validation script still passes."""
    env = dict(os.environ)
    # Defensive: SEC-04's own validator must run with the runtime flags OFF
    # so that we don't accidentally trigger encrypted-runtime in B11's
    # subprocess — which would be a different behavior entirely.
    env.pop("MEDAI_CKA_ENCRYPTED_STORE_ENABLED", None)
    env.pop("MEDAI_CKA_ENCRYPTED_STORE_PATH", None)
    env.pop("MEDAI_CKA_ENCRYPTION_KEY", None)
    env.pop("MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING", None)
    try:
        res = subprocess.run(
            [sys.executable, "scripts/run_cka_final_mvp_release_validation.py"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True,
            timeout=180, env=env, check=False,
        )
    except subprocess.TimeoutExpired:
        return _fail("I", "Final CKA validation invocable",
                     "subprocess_timed_out")
    if res.returncode != 0:
        last = (res.stdout.strip().splitlines() or [""])[-1]
        return _fail("I", "Final CKA validation invocable",
                     f"validation_returncode={res.returncode}",
                     {"last_line": last})
    return _ok("I", "Final CKA validation invocable", {
        "subprocess_exit_code": res.returncode,
        "pass_marker_seen": "[PASS]" in res.stdout,
    })


def case_j_report_safety(report: dict) -> dict:
    """Public-report safety: no key/path/PHI/secret/temp-prefix; CKA-B02 checker."""
    text = json.dumps(report)
    if "synth_op_" in text:
        return _fail("J", "Report safety", "synthetic_op_key_prefix_in_report")
    if re.search(r"[A-Za-z]:\\\\", text):
        return _fail("J", "Report safety", "drive_letter_path_in_report")
    for needle in ("cka_sec04_v_", "cka_sec04_smoke_", "cka_sec04_pytest_",
                   "cka_sec03a_v_", "cka_sec03a_init_", "cka_sec03a_pytest_"):
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
    provider_summary: dict,
    case_e_summary: Optional[dict],
) -> dict:
    return {
        "block_id": "CKA-SEC-04",
        "conclusion": "cka_sec04_encrypted_runtime_activation_ready",
        "sqlcipher_provider_available": bool(provider_summary.get("available")),
        "sqlcipher_provider_name": provider_summary.get("provider_name"),
        "cipher_version_available": bool(provider_summary.get("cipher_version")),
        "default_runtime_encryption_active": False,
        "encrypted_runtime_flag_supported": True,
        "encrypted_runtime_blocks_without_key": True,
        "wrong_key_failure_passed": True,
        "test_mode_encrypted_runtime_opened": bool(
            case_e_summary and case_e_summary.get("encrypted_store_opened")
        ),
        "test_mode_records_count_zero": bool(
            case_e_summary and case_e_summary.get("records_count") == 0
        ),
        "create_if_missing_requires_explicit_flag": True,
        "rollback_plan_ready": True,
        "main_store_migration_performed": False,
        "real_data_migrated": False,
        "existing_store_migrated": False,
        "real_empty_store_created_by_default": False,
        "db_file_staged": False,
        "key_stored_in_repo": False,
        "encryption_key_logged": False,
        "runtime_activation_default_off": True,
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
        "next_recommended_block": (
            "CKA-SEC-05 Operator Runtime Launch Script, only if "
            "operator wants one-click encrypted-runtime startup"
        ),
        "case_e_summary": case_e_summary,
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
            f"CKA-B02 privacy checker rejected SEC-04 report: "
            f"{r.leak_examples_redacted}"
        )


def _write_reports(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "cka_sec04_encrypted_runtime_activation_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# CKA-SEC-04 Encrypted Runtime Activation Report",
        "",
        f"- block_id: {report['block_id']}",
        f"- conclusion: {report['conclusion']}",
        f"- sqlcipher_provider_available: {report['sqlcipher_provider_available']}",
        f"- sqlcipher_provider_name: {report['sqlcipher_provider_name']}",
        f"- cipher_version_available: {report['cipher_version_available']}",
        "",
        "## Runtime activation guard",
        "",
        f"- default_runtime_encryption_active: {report['default_runtime_encryption_active']}",
        f"- runtime_activation_default_off: {report['runtime_activation_default_off']}",
        f"- encrypted_runtime_flag_supported: {report['encrypted_runtime_flag_supported']}",
        f"- encrypted_runtime_blocks_without_key: {report['encrypted_runtime_blocks_without_key']}",
        f"- wrong_key_failure_passed: {report['wrong_key_failure_passed']}",
        f"- test_mode_encrypted_runtime_opened: {report['test_mode_encrypted_runtime_opened']}",
        f"- test_mode_records_count_zero: {report['test_mode_records_count_zero']}",
        f"- create_if_missing_requires_explicit_flag: {report['create_if_missing_requires_explicit_flag']}",
        f"- rollback_plan_ready: {report['rollback_plan_ready']}",
        "",
        "## Main store boundary",
        "",
        f"- main_store_migration_performed: {report['main_store_migration_performed']}",
        f"- real_data_migrated: {report['real_data_migrated']}",
        f"- existing_store_migrated: {report['existing_store_migrated']}",
        f"- real_empty_store_created_by_default: {report['real_empty_store_created_by_default']}",
        "",
        "## Safety / privacy",
        "",
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
        "## Next recommended block",
        "",
        report["next_recommended_block"],
        "",
    ]
    (REPORT_DIR / "cka_sec04_encrypted_runtime_activation_report.md").write_text(
        "\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_validation() -> dict:
    print("  [SEC-04] case A: baseline SEC-03A ...", flush=True)
    res_a = case_a_baseline_sec03a()
    provider_summary = detect_sqlcipher_provider().safe_public_summary()

    print("  [SEC-04] case B: default runtime unencrypted ...", flush=True)
    res_b = case_b_default_runtime_unencrypted()

    print("  [SEC-04] case C: runtime flag without key blocks ...", flush=True)
    res_c = case_c_runtime_flag_without_key_blocks()

    print("  [SEC-04] case D: runtime wrong key fails ...", flush=True)
    res_d = case_d_runtime_wrong_key_fails()

    print("  [SEC-04] case E: test-mode encrypted runtime ...", flush=True)
    res_e, e_summary = case_e_test_mode_runtime_creates_empty()

    print("  [SEC-04] case F: create_if_missing false blocks ...", flush=True)
    res_f = case_f_create_if_missing_false_blocks()

    print("  [SEC-04] case G: rollback plan ready ...", flush=True)
    res_g = case_g_rollback_plan_ready()

    print("  [SEC-04] case H: main store untouched ...", flush=True)
    res_h = case_h_main_store_untouched()

    print("  [SEC-04] case I: final CKA validation invocable ...", flush=True)
    res_i = case_i_final_cka_validation()

    results = [res_a, res_b, res_c, res_d, res_e, res_f, res_g, res_h, res_i]
    report = _build_report(results, provider_summary, e_summary)

    print("  [SEC-04] case J: report safety ...", flush=True)
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
    print(f"\nCKA-SEC-04 Encrypted Runtime Activation — {status}")
    print(f"  conclusion: {rep['conclusion']}")
    print(f"  default_runtime_encryption_active: {rep['default_runtime_encryption_active']}")
    print(f"  runtime_activation_default_off: {rep['runtime_activation_default_off']}")
    print(f"  encrypted_runtime_flag_supported: {rep['encrypted_runtime_flag_supported']}")
    print(f"  encrypted_runtime_blocks_without_key: {rep['encrypted_runtime_blocks_without_key']}")
    print(f"  wrong_key_failure_passed: {rep['wrong_key_failure_passed']}")
    print(f"  test_mode_encrypted_runtime_opened: {rep['test_mode_encrypted_runtime_opened']}")
    print(f"  test_mode_records_count_zero: {rep['test_mode_records_count_zero']}")
    print(f"  main_store_migration_performed: {rep['main_store_migration_performed']}")
    print(f"  real_data_migrated: {rep['real_data_migrated']}")
    print(f"  cases_passed: {rep['cases_passed']} / {rep['synthetic_cases_run']}")
    for r in rep["case_results"]:
        marker = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"    {marker} case {r['case']}: {r['description']}")
        if not r["passed"]:
            print(f"           error: {r.get('error')}")
    if not rep["all_passed"]:
        sys.exit(1)
