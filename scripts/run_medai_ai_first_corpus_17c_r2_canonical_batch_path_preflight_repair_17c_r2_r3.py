#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-17C-R2-CANONICAL-BATCH-PATH-PREFLIGHT-REPAIR-17C-R2-R3.

Verifies the canonical-batch path resolution fix (robust resolver), the sealed 478
batch (physical-newline framing + SHA256), and the Vertex credential preflight.
Local/preflight only: NO model call, NO content request, NO billing, NO live gate, NO
live extraction, NO MKB.

Public reports carry path strings, counts, hashes, and status only — never request
bodies, token maps, raw OCR, PI values, credentials, or tokens.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.canonical_batch_paths import EXPECTED_CANONICAL_BATCH, resolve_canonical_batch
from execution.jsonl_framing import load_jsonl_objects
import scripts.run_medai_ai_first_corpus_478_batch_validation_repair_local_only_17b_r2_r1 as r2r1
import scripts.run_medai_ai_first_corpus_jsonl_loader_hardening_and_credential_preflight_17c_r2_r2 as r2r2

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_canonical_batch_path_preflight_repair_17c_r2_r3"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_CANONICAL_BATCH_PATH_PREFLIGHT_REPAIR_17C_R2_R3"
REQUIRED_DOCS = (
    "MEDAI_17C_R2_CANONICAL_BATCH_PATH_PREFLIGHT_REPAIR_17C_R2_R3.md",
    "MEDAI_CANONICAL_PRIVATE_BATCH_RESOLUTION_POLICY_17C_R2_R3.md",
    "MEDAI_17C_R2_LIVE_RERUN_ENTRY_GATE_AFTER_R3.md",
)
EXPECTED = 478
TARGET_PROJECT = "sot-knowledge-ocr"


def run() -> dict[str, Any]:
    path, resolved_exists = resolve_canonical_batch()
    is_file = path.is_file()
    readable = os.access(path, os.R_OK) if path.exists() else False
    readonly = (not os.access(path, os.W_OK)) if path.exists() else False
    parent_exists = path.parent.is_dir()

    sidecar_dir = path.parent
    integrity_sidecar = (sidecar_dir / "outbound_requests_integrity_private.json").is_file()
    sha_sidecar_path = sidecar_dir / "outbound_requests_private.sha256"
    sha_sidecar = sha_sidecar_path.is_file()
    docid_manifest = (sidecar_dir / "doc_id_manifest_private.json").is_file()

    # Sealed batch verification (physical-newline framing).
    sha_ok = False
    nonempty = parseable = unique = malformed = residual = 0
    if is_file and readable:
        objs, malformed, nonempty = load_jsonl_objects(path)
        parseable = len(objs)
        unique = len({o.get("document_id") for o in objs})
        residual = sum(1 for o in objs if r2r1._residual_pi(o.get("tokenized_content", "")))
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        sealed_sha = ""
        if sha_sidecar:
            sealed_sha = (sha_sidecar_path.read_text(encoding="utf-8").strip().split() or [""])[0]
        sha_ok = bool(sealed_sha) and sealed_sha == actual_sha
    else:
        actual_sha = ""

    sealed_batch_valid = (is_file and readable and nonempty == EXPECTED and parseable == EXPECTED
                          and unique == EXPECTED and malformed == 0 and residual == 0 and sha_ok)

    # Credential preflight (reuse 17C-R2-R2 helper; no model call, no token printed).
    cred_pass, adc_result, _ga = r2r2._credential_preflight()
    project = r2r2._detect_project()

    missing_reason = "" if (is_file and readable) else (
        "localappdata_unset_or_path_unresolved" if not resolved_exists else
        ("not_a_file" if not is_file else "not_readable"))

    ready = bool(sealed_batch_valid and cred_pass)
    privacy_result = "passed" if sealed_batch_valid else "blocked"

    return {
        "block": "MEDAI-AI-FIRST-CORPUS-17C-R2-CANONICAL-BATCH-PATH-PREFLIGHT-REPAIR-17C-R2-R3",
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
        "canonical_batch_expected_path": EXPECTED_CANONICAL_BATCH,
        "canonical_batch_resolved_path": str(path),
        "canonical_batch_exists": path.exists(),
        "canonical_batch_is_file": is_file,
        "canonical_batch_readable": readable,
        "canonical_batch_readonly": readonly,
        "canonical_batch_parent_exists": parent_exists,
        "canonical_batch_missing_reason": missing_reason,
        "integrity_sidecar_exists": integrity_sidecar,
        "sha256_sidecar_exists": sha_sidecar,
        "doc_id_manifest_exists": docid_manifest,
        "sha256_sidecar_verified": sha_ok,
        "sealed_batch_valid": sealed_batch_valid,
        "request_count_loaded": nonempty,
        "physical_newline_record_count": nonempty,
        "parseable_record_count": parseable,
        "unique_doc_ids": unique,
        "malformed_record_count": malformed,
        "request_validation_passed": sealed_batch_valid,
        "residual_pi_failures": residual,
        "credential_preflight_attempted": True,
        "credential_preflight_passed": cred_pass,
        "project_detected": project if project == TARGET_PROJECT else (project if project != "unknown" else "unknown"),
        "adc_refresh_result": adc_result,
        "ready_to_rerun_17c_r2_live": ready,
        "sha256_prefix": ("sha256:" + actual_sha[:16]) if actual_sha else "",
        "private_outbound_requests_committed": False,
        "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False,
        "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False,
        "credential_or_token_written_to_repo": False,
        "public_report_phi_leak_count": 0,
        "privacy_result": privacy_result,
        "safety_result": "passed",
    }


def _path_public(s: dict[str, Any]) -> dict[str, Any]:
    return {k: s[k] for k in (
        "canonical_batch_expected_path", "canonical_batch_resolved_path", "canonical_batch_exists",
        "canonical_batch_is_file", "canonical_batch_readable", "canonical_batch_readonly",
        "canonical_batch_parent_exists", "canonical_batch_missing_reason",
        "integrity_sidecar_exists", "sha256_sidecar_exists", "doc_id_manifest_exists")}


def _integrity_public(s: dict[str, Any]) -> dict[str, Any]:
    return {k: s[k] for k in (
        "sha256_sidecar_verified", "sha256_prefix", "sealed_batch_valid",
        "physical_newline_record_count", "parseable_record_count", "unique_doc_ids",
        "malformed_record_count", "residual_pi_failures")}


def _cred_public(s: dict[str, Any]) -> dict[str, Any]:
    return {"credential_preflight_attempted": s["credential_preflight_attempted"],
            "credential_preflight_passed": s["credential_preflight_passed"],
            "project_detected": s["project_detected"], "adc_refresh_result": s["adc_refresh_result"],
            "vertex_model_call_made": False, "note": "No model call; no token or credential printed."}


def _rerun_gate_md(s: dict[str, Any]) -> str:
    return "\n".join(["# 17C-R2 live rerun entry gate (after R3)", "",
                      f"- canonical_batch_resolved_path: `{s['canonical_batch_resolved_path']}`",
                      f"- canonical_batch_is_file / readable: `{s['canonical_batch_is_file']}` / `{s['canonical_batch_readable']}`",
                      f"- sealed_batch_valid: `{s['sealed_batch_valid']}`",
                      f"- credential_preflight_passed: `{s['credential_preflight_passed']}` (adc: `{s['adc_refresh_result']}`)",
                      f"- ready_to_rerun_17c_r2_live: `{s['ready_to_rerun_17c_r2_live']}`", "",
                      "17C-R2 is NOT re-run here. Re-run only when the canonical batch resolves, the",
                      "sealed batch is valid, and the credential preflight passes — keeping the run",
                      "chunked/capped with stop-on-first-failure and the live gate scoped to each chunk.", ""])


def _implementation(s: dict[str, Any]) -> str:
    lines = [f"# {s['block']}", "",
             f"## Result: **{'PASS' if s['privacy_result'] == 'passed' else 'BLOCKED'}** (safety: {s['safety_result']})",
             "", "## Root cause", "",
             "- The 17C-R2 runner resolved the canonical path only via `%LOCALAPPDATA%` expansion;",
             "  when that variable is unset, `expandvars` returns a literal nonexistent path and the",
             "  preflight falsely reported `canonical_batch_missing`. Fixed with a shared resolver",
             "  (explicit path -> %LOCALAPPDATA% -> Path.home()).",
             "", "## Metrics", ""]
    for k in ("canonical_batch_resolved_path", "canonical_batch_exists", "canonical_batch_is_file",
              "canonical_batch_readable", "canonical_batch_readonly", "canonical_batch_parent_exists",
              "canonical_batch_missing_reason", "integrity_sidecar_exists", "sha256_sidecar_exists",
              "doc_id_manifest_exists", "sha256_sidecar_verified", "sealed_batch_valid",
              "request_count_loaded", "physical_newline_record_count", "parseable_record_count",
              "unique_doc_ids", "malformed_record_count", "request_validation_passed",
              "residual_pi_failures", "credential_preflight_passed", "project_detected",
              "adc_refresh_result", "ready_to_rerun_17c_r2_live", "provider_model_call_made",
              "vertex_model_call_made", "billing_api_call_made", "live_gate_set",
              "live_extraction_started", "mkb_db_opened", "active_mkb_write",
              "private_outbound_requests_committed", "public_report_phi_leak_count",
              "privacy_result", "safety_result"):
        lines.append(f"- {k}: `{s[k]}`")
    nxt = ("rerun 17C-R2 live extraction once" if s["ready_to_rerun_17c_r2_live"]
           else "fix reported path/credential blocker first")
    lines += ["", "## Recommended next (no live run started)", "", f"- {nxt}.", ""]
    return "\n".join(lines)


def main() -> int:
    s = run()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path_pub = _path_public(s)
    integ_pub = _integrity_public(s)
    cred_pub = _cred_public(s)
    rerun_md = _rerun_gate_md(s)
    impl_md = _implementation(s)

    public_blob = "\n".join([json.dumps(s), json.dumps(path_pub), json.dumps(integ_pub),
                             json.dumps(cred_pub), rerun_md, impl_md])
    for marker in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", '"tokenized_content"'):
        if marker in public_blob:
            s["public_report_phi_leak_count"] = 1
            s["safety_result"] = "blocked"
            break

    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(_implementation(s), encoding="utf-8")
    (REPORT_DIR / "canonical_batch_path_resolution_public.json").write_text(json.dumps(path_pub, indent=2), encoding="utf-8")
    (REPORT_DIR / "sealed_batch_integrity_public.json").write_text(json.dumps(integ_pub, indent=2), encoding="utf-8")
    (REPORT_DIR / "credential_preflight_public.json").write_text(json.dumps(cred_pub, indent=2), encoding="utf-8")
    (REPORT_DIR / "rerun_17c_r2_entry_gate.md").write_text(rerun_md, encoding="utf-8")

    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    ok = (s["safety_result"] == "passed" and s["public_report_phi_leak_count"] == 0
          and s["provider_model_call_made"] is False and s["sealed_batch_valid"] is True
          and s["canonical_batch_is_file"] is True and s["canonical_batch_readable"] is True and docs_ok)
    print("medai_ai_first_corpus_17c_r2_canonical_batch_path_preflight_repair_17c_r2_r3_"
          + ("pass" if ok and s["privacy_result"] == "passed" else "blocked"))
    print(json.dumps({k: s[k] for k in (
        "canonical_batch_resolved_path", "canonical_batch_is_file", "canonical_batch_readable",
        "canonical_batch_readonly", "sha256_sidecar_verified", "sealed_batch_valid",
        "physical_newline_record_count", "unique_doc_ids", "malformed_record_count",
        "credential_preflight_passed", "adc_refresh_result", "ready_to_rerun_17c_r2_live",
        "public_report_phi_leak_count", "privacy_result", "safety_result")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
