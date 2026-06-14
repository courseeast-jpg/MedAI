#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-OPERATOR-CONTEXT-CANONICAL-BATCH-RESTORE-17C-R2-R4.

Operator-context diagnosis + resilient restore of the sealed canonical 478-request
private batch. If the batch or its sidecars are missing/invalid in the current runtime,
re-materializes them deterministically from the 17A tokenized corpus using the
17B-R2-R1 repair logic (ensure_ascii=True, physical-newline framing), seals SHA256 +
doc-id manifest, and sets read-only. Then verifies and runs a Vertex credential
preflight.

Local/preflight only: NO model call, NO content request, NO billing, NO live gate, NO
live extraction, NO MKB. Public reports carry booleans/counts/hashes/paths only.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.canonical_batch_paths import EXPECTED_CANONICAL_BATCH, resolve_canonical_batch
from execution.jsonl_framing import load_jsonl_objects
import scripts.run_medai_ai_first_corpus_478_batch_validation_repair_local_only_17b_r2_r1 as r2r1
import scripts.run_medai_ai_first_corpus_478_batch_integrity_restore_local_only_17c_r2_r1 as restore
import scripts.run_medai_ai_first_corpus_jsonl_loader_hardening_and_credential_preflight_17c_r2_r2 as r2r2

EXPECTED = 478
TARGET_PROJECT = "sot-knowledge-ocr"

# Canonical path + sidecars, anchored to the explicit expected directory.
CANON = Path(EXPECTED_CANONICAL_BATCH)
CANON_DIR = CANON.parent
INTEGRITY_SIDECAR = CANON_DIR / "outbound_requests_integrity_private.json"
SHA_SIDECAR = CANON_DIR / "outbound_requests_private.sha256"
DOCID_MANIFEST = CANON_DIR / "doc_id_manifest_private.json"
CORPUS_DOCS = r2r1.CORPUS_DOCS

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_operator_context_canonical_batch_restore_17c_r2_r4"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_OPERATOR_CONTEXT_CANONICAL_BATCH_RESTORE_17C_R2_R4"
REQUIRED_DOCS = (
    "MEDAI_OPERATOR_CONTEXT_CANONICAL_BATCH_RESTORE_17C_R2_R4.md",
    "MEDAI_PRIVATE_ARTIFACT_VISIBILITY_POLICY_17C_R2_R4.md",
    "MEDAI_17C_R2_LIVE_RERUN_ENTRY_GATE_AFTER_R4.md",
)


def _sidecars_exist() -> bool:
    return INTEGRITY_SIDECAR.is_file() and SHA_SIDECAR.is_file() and DOCID_MANIFEST.is_file()


def _sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_sealed() -> dict[str, Any]:
    if not CANON.is_file() or not os.access(CANON, os.R_OK):
        return {"sealed_batch_valid": False, "physical_newline_record_count": 0,
                "parseable_record_count": 0, "unique_doc_ids": 0, "malformed_record_count": 0,
                "residual_pi_failures": 0, "sha256_sidecar_verified": False, "sha256_prefix": ""}
    objs, malformed, nonempty = load_jsonl_objects(CANON)
    unique = {o.get("document_id") for o in objs}
    residual = sum(1 for o in objs if r2r1._residual_pi(o.get("tokenized_content", "")))
    actual = _sha_of(CANON)
    sealed_sha = ""
    if SHA_SIDECAR.is_file():
        sealed_sha = (SHA_SIDECAR.read_text(encoding="utf-8").strip().split() or [""])[0]
    sha_ok = bool(sealed_sha) and sealed_sha == actual
    valid = (nonempty == EXPECTED and len(objs) == EXPECTED and len(unique) == EXPECTED
             and malformed == 0 and residual == 0 and sha_ok and _sidecars_exist())
    return {"sealed_batch_valid": valid, "physical_newline_record_count": nonempty,
            "parseable_record_count": len(objs), "unique_doc_ids": len(unique),
            "malformed_record_count": malformed, "residual_pi_failures": residual,
            "sha256_sidecar_verified": sha_ok, "sha256_prefix": "sha256:" + actual[:16]}


def _rematerialize() -> tuple[bool, bool, list[str]]:
    """Rebuild + seal the canonical batch deterministically. Returns
    (batch_restored, sidecars_restored, notes)."""
    notes: list[str] = []
    if not CORPUS_DOCS.is_dir() or not any(CORPUS_DOCS.iterdir()):
        return False, False, ["private_tokenized_corpus_missing"]
    requests, residual_failures, build_notes = restore._rebuild_requests()
    notes += build_notes
    if len(requests) != EXPECTED or residual_failures != 0 or build_notes:
        notes.append(f"rebuild_incomplete_count_{len(requests)}_residual_{residual_failures}")
        return False, False, notes
    CANON_DIR.mkdir(parents=True, exist_ok=True)
    # Clear read-only so we can replace atomically.
    if CANON.exists() and not os.access(CANON, os.W_OK):
        os.chmod(CANON, stat.S_IWRITE | stat.S_IREAD)
    tmp = CANON_DIR / "outbound_requests_private.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        for req in requests:
            fh.write(json.dumps(req, ensure_ascii=True) + "\n")  # ensure_ascii=True framing
    os.replace(tmp, CANON)
    sha = _sha_of(CANON)
    doc_ids = sorted(r["document_id"] for r in requests)
    INTEGRITY_SIDECAR.write_text(json.dumps({
        "line_count": EXPECTED, "splitlines_count": EXPECTED, "parseable": EXPECTED,
        "unique_doc_ids": EXPECTED, "malformed": 0, "residual_pi_failures": 0,
        "request_validation_passed": True, "sha256": sha, "expected_request_count": EXPECTED},
        indent=2), encoding="utf-8")
    SHA_SIDECAR.write_text(sha + "  outbound_requests_private.jsonl\n", encoding="utf-8")
    DOCID_MANIFEST.write_text(json.dumps({"count": len(doc_ids), "doc_ids": doc_ids}, indent=2), encoding="utf-8")
    try:
        os.chmod(CANON, stat.S_IREAD)
    except OSError:
        notes.append("readonly_set_failed")
    return True, True, notes


def _credential_preflight() -> tuple[bool, str, str]:
    ok, adc, _ga = r2r2._credential_preflight()
    return ok, adc, r2r2._detect_project()


def run() -> dict[str, Any]:
    notes: list[str] = []
    # Operator-context diagnostics (before).
    parent_exists = CANON_DIR.is_dir()
    exists_before = CANON.is_file()
    sidecars_before = _sidecars_exist()

    pre_verify = _verify_sealed() if exists_before else {"sealed_batch_valid": False}
    need_restore = not (exists_before and sidecars_before and pre_verify.get("sealed_batch_valid"))

    batch_restored = sidecars_restored = False
    if need_restore:
        batch_restored, sidecars_restored, rnotes = _rematerialize()
        notes += rnotes
        if "private_tokenized_corpus_missing" in rnotes:
            return _blocked("private_tokenized_corpus_missing", parent_exists, exists_before,
                            sidecars_before, notes)

    verify = _verify_sealed()
    readable_after = os.access(CANON, os.R_OK) if CANON.exists() else False
    readonly_after = (not os.access(CANON, os.W_OK)) if CANON.exists() else False

    cred_pass, adc_result, project = _credential_preflight()
    ready = bool(verify["sealed_batch_valid"] and cred_pass)
    privacy_result = "passed" if verify["sealed_batch_valid"] else "blocked"

    return {
        "block": "MEDAI-AI-FIRST-CORPUS-OPERATOR-CONTEXT-CANONICAL-BATCH-RESTORE-17C-R2-R4",
        "local_preflight_only": True,
        "provider_model_call_made": False,
        "vertex_model_call_made": False,
        "gemini_call_made": False,
        "claude_call_made": False,
        "openai_call_made": False,
        "billing_api_call_made": False,
        "live_gate_set": False,
        "live_extraction_started": False,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "production_queue_mutated": False,
        "operator_context_checked": True,
        "operator_user": getpass.getuser(),
        "home_dir": str(Path.home()),
        "localappdata_present": bool(os.environ.get("LOCALAPPDATA")),
        "canonical_batch_expected_path": EXPECTED_CANONICAL_BATCH,
        "canonical_batch_parent_exists": parent_exists,
        "canonical_batch_exists_before": exists_before,
        "sidecars_exist_before": sidecars_before,
        "canonical_batch_restored": batch_restored,
        "sidecars_restored": sidecars_restored,
        "canonical_batch_exists_after": CANON.is_file(),
        "canonical_batch_is_file_after": CANON.is_file(),
        "canonical_batch_readable_after": readable_after,
        "canonical_batch_readonly_after": readonly_after,
        "sha256_sidecar_verified": verify["sha256_sidecar_verified"],
        "sealed_batch_valid": verify["sealed_batch_valid"],
        "request_count_loaded": verify["physical_newline_record_count"],
        "physical_newline_record_count": verify["physical_newline_record_count"],
        "parseable_record_count": verify["parseable_record_count"],
        "unique_doc_ids": verify["unique_doc_ids"],
        "malformed_record_count": verify["malformed_record_count"],
        "request_validation_passed": verify["sealed_batch_valid"],
        "residual_pi_failures": verify["residual_pi_failures"],
        "old_12_added_separately": False,
        "combined_batch_count": EXPECTED,
        "credential_preflight_attempted": True,
        "credential_preflight_passed": cred_pass,
        "project_detected": project if project == TARGET_PROJECT else (project if project != "unknown" else "unknown"),
        "adc_refresh_result": adc_result,
        "ready_to_rerun_17c_r2_live": ready,
        "sha256_prefix": verify["sha256_prefix"],
        "private_outbound_requests_committed": False,
        "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False,
        "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False,
        "credential_or_token_written_to_repo": False,
        "public_report_phi_leak_count": 0,
        "privacy_result": privacy_result,
        "safety_result": "passed",
        "notes": notes,
    }


def _blocked(reason: str, parent_exists: bool, exists_before: bool, sidecars_before: bool,
             notes: list[str]) -> dict[str, Any]:
    return {
        "block": "MEDAI-AI-FIRST-CORPUS-OPERATOR-CONTEXT-CANONICAL-BATCH-RESTORE-17C-R2-R4",
        "local_preflight_only": True, "provider_model_call_made": False, "vertex_model_call_made": False,
        "gemini_call_made": False, "claude_call_made": False, "openai_call_made": False,
        "billing_api_call_made": False, "live_gate_set": False, "live_extraction_started": False,
        "mkb_db_opened": False, "active_mkb_write": False, "auto_accept_enabled": False,
        "medical_decision_made": False, "production_queue_mutated": False, "operator_context_checked": True,
        "operator_user": getpass.getuser(), "home_dir": str(Path.home()),
        "localappdata_present": bool(os.environ.get("LOCALAPPDATA")),
        "canonical_batch_expected_path": EXPECTED_CANONICAL_BATCH,
        "canonical_batch_parent_exists": parent_exists, "canonical_batch_exists_before": exists_before,
        "sidecars_exist_before": sidecars_before, "canonical_batch_restored": False,
        "sidecars_restored": False, "canonical_batch_exists_after": CANON.is_file(),
        "canonical_batch_is_file_after": CANON.is_file(), "canonical_batch_readable_after": False,
        "canonical_batch_readonly_after": False, "sha256_sidecar_verified": False,
        "sealed_batch_valid": False, "request_count_loaded": 0, "physical_newline_record_count": 0,
        "parseable_record_count": 0, "unique_doc_ids": 0, "malformed_record_count": 0,
        "request_validation_passed": False, "residual_pi_failures": 0, "old_12_added_separately": False,
        "combined_batch_count": EXPECTED, "credential_preflight_attempted": False,
        "credential_preflight_passed": False, "project_detected": "unknown",
        "adc_refresh_result": "not_attempted", "ready_to_rerun_17c_r2_live": False, "sha256_prefix": "",
        "private_outbound_requests_committed": False, "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False, "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False, "credential_or_token_written_to_repo": False,
        "public_report_phi_leak_count": 0, "privacy_result": "blocked", "safety_result": "passed",
        "blocked_reason": reason, "notes": notes,
    }


def _diag_public(s: dict[str, Any]) -> dict[str, Any]:
    return {k: s[k] for k in (
        "operator_user", "home_dir", "localappdata_present", "canonical_batch_expected_path",
        "canonical_batch_parent_exists", "canonical_batch_exists_before", "sidecars_exist_before",
        "canonical_batch_restored", "sidecars_restored", "canonical_batch_exists_after",
        "canonical_batch_is_file_after", "canonical_batch_readable_after", "canonical_batch_readonly_after")}


def _integrity_public(s: dict[str, Any]) -> dict[str, Any]:
    return {k: s[k] for k in (
        "sha256_sidecar_verified", "sha256_prefix", "sealed_batch_valid",
        "physical_newline_record_count", "parseable_record_count", "unique_doc_ids",
        "malformed_record_count", "residual_pi_failures", "old_12_added_separately",
        "combined_batch_count")}


def _cred_public(s: dict[str, Any]) -> dict[str, Any]:
    return {"credential_preflight_attempted": s["credential_preflight_attempted"],
            "credential_preflight_passed": s["credential_preflight_passed"],
            "project_detected": s["project_detected"], "adc_refresh_result": s["adc_refresh_result"],
            "vertex_model_call_made": False, "note": "No model call; no token or credential printed."}


def _rerun_md(s: dict[str, Any]) -> str:
    return "\n".join(["# 17C-R2 live rerun entry gate (after R4)", "",
                      f"- canonical_batch_exists_after: `{s['canonical_batch_exists_after']}`",
                      f"- canonical_batch_readable_after: `{s['canonical_batch_readable_after']}` (readonly: `{s['canonical_batch_readonly_after']}`)",
                      f"- sealed_batch_valid: `{s['sealed_batch_valid']}`",
                      f"- credential_preflight_passed: `{s['credential_preflight_passed']}` (adc: `{s['adc_refresh_result']}`)",
                      f"- ready_to_rerun_17c_r2_live: `{s['ready_to_rerun_17c_r2_live']}`", "",
                      "17C-R2 is NOT re-run here. The live runner must re-verify the canonical batch",
                      "(existence + SHA256 + counts) immediately before any provider call.", ""])


def _implementation(s: dict[str, Any]) -> str:
    lines = [f"# {s['block']}", "",
             f"## Result: **{'PASS' if s['privacy_result'] == 'passed' else 'BLOCKED'}** (safety: {s['safety_result']})",
             "", "## Metrics", ""]
    for k in ("operator_context_checked", "canonical_batch_parent_exists", "canonical_batch_exists_before",
              "sidecars_exist_before", "canonical_batch_restored", "sidecars_restored",
              "canonical_batch_exists_after", "canonical_batch_readable_after", "canonical_batch_readonly_after",
              "sha256_sidecar_verified", "sealed_batch_valid", "request_count_loaded",
              "physical_newline_record_count", "parseable_record_count", "unique_doc_ids",
              "malformed_record_count", "request_validation_passed", "residual_pi_failures",
              "credential_preflight_passed", "project_detected", "adc_refresh_result",
              "ready_to_rerun_17c_r2_live", "provider_model_call_made", "vertex_model_call_made",
              "billing_api_call_made", "live_gate_set", "live_extraction_started", "mkb_db_opened",
              "active_mkb_write", "private_outbound_requests_committed", "public_report_phi_leak_count",
              "privacy_result", "safety_result"):
        if k in s:
            lines.append(f"- {k}: `{s[k]}`")
    if s.get("blocked_reason"):
        lines += ["", f"- blocked_reason: `{s['blocked_reason']}`"]
    nxt = ("rerun 17C-R2 live extraction once" if s["ready_to_rerun_17c_r2_live"]
           else "fix reported private artifact or credential blocker first")
    lines += ["", "## Recommended next (no live run started)", "", f"- {nxt}.", ""]
    return "\n".join(lines)


def main() -> int:
    s = run()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    diag_pub = _diag_public(s)
    integ_pub = _integrity_public(s)
    cred_pub = _cred_public(s)
    rerun_md = _rerun_md(s)
    impl_md = _implementation(s)

    public_blob = "\n".join([json.dumps(s), json.dumps(diag_pub), json.dumps(integ_pub),
                             json.dumps(cred_pub), rerun_md, impl_md])
    for marker in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", '"tokenized_content"'):
        if marker in public_blob:
            s["public_report_phi_leak_count"] = 1
            s["safety_result"] = "blocked"
            break

    # Re-render summary without the verbose 'notes' detail in public JSON top-level? keep notes (safe class strings).
    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(_implementation(s), encoding="utf-8")
    (REPORT_DIR / "operator_context_path_diagnostics_public.json").write_text(json.dumps(diag_pub, indent=2), encoding="utf-8")
    (REPORT_DIR / "restored_batch_integrity_public.json").write_text(json.dumps(integ_pub, indent=2), encoding="utf-8")
    (REPORT_DIR / "credential_preflight_public.json").write_text(json.dumps(cred_pub, indent=2), encoding="utf-8")
    (REPORT_DIR / "rerun_17c_r2_entry_gate.md").write_text(rerun_md, encoding="utf-8")

    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    ok = (s["safety_result"] == "passed" and s["public_report_phi_leak_count"] == 0
          and s["provider_model_call_made"] is False and s["sealed_batch_valid"] is True
          and s["canonical_batch_exists_after"] is True and s["canonical_batch_readable_after"] is True
          and docs_ok)
    print("medai_ai_first_corpus_operator_context_canonical_batch_restore_17c_r2_r4_"
          + ("pass" if ok and s["privacy_result"] == "passed" else "blocked"))
    print(json.dumps({k: s[k] for k in (
        "canonical_batch_exists_before", "sidecars_exist_before", "canonical_batch_restored",
        "canonical_batch_exists_after", "canonical_batch_readable_after", "canonical_batch_readonly_after",
        "sha256_sidecar_verified", "sealed_batch_valid", "physical_newline_record_count", "unique_doc_ids",
        "malformed_record_count", "credential_preflight_passed", "adc_refresh_result",
        "ready_to_rerun_17c_r2_live", "public_report_phi_leak_count", "privacy_result", "safety_result")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
