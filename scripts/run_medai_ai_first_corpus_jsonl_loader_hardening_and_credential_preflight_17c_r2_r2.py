#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-JSONL-LOADER-HARDENING-AND-CREDENTIAL-PREFLIGHT-17C-R2-R2.

Verifies the JSONL loader hardening (physical-newline framing) and the sealed canonical
478 batch, then runs a Vertex credential preflight. Local/preflight only: NO model call,
NO content request, NO billing call, NO live gate, NO live extraction, NO MKB.

Public reports carry counts/hashes/status only — never request bodies, token maps, raw
OCR, PI values, credentials, or tokens.
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

from execution.jsonl_framing import load_jsonl_objects, read_jsonl_lines
import scripts.run_medai_ai_first_corpus_478_batch_validation_repair_local_only_17b_r2_r1 as r2r1

CANON_DIR = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17B_R2_R1_478_repaired"))
CANON_BATCH = CANON_DIR / "outbound_requests_private.jsonl"
CANON_BATCH_LABEL = r"C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17B_R2_R1_478_repaired\outbound_requests_private.jsonl"
SHA_SIDECAR = CANON_DIR / "outbound_requests_private.sha256"
INTEGRITY_SIDECAR = CANON_DIR / "outbound_requests_integrity_private.json"
DOCID_MANIFEST = CANON_DIR / "doc_id_manifest_private.json"

LIVE_LOADER_SRC = REPO_ROOT / "scripts" / "run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2.py"

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_jsonl_loader_hardening_and_credential_preflight_17c_r2_r2"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_JSONL_LOADER_HARDENING_AND_CREDENTIAL_PREFLIGHT_17C_R2_R2"
REQUIRED_DOCS = (
    "MEDAI_JSONL_LOADER_HARDENING_AND_CREDENTIAL_PREFLIGHT_17C_R2_R2.md",
    "MEDAI_JSONL_PHYSICAL_NEWLINE_FRAMING_POLICY_17C_R2_R2.md",
    "MEDAI_478_BATCH_RERUN_ENTRY_GATE_17C_R2_R2.md",
)

EXPECTED = 478
TARGET_PROJECT = "sot-knowledge-ocr"


def _live_loader_hardened() -> bool:
    """True iff the 17C-R2 live batch loader uses the physical-newline reader and no
    longer frames the canonical batch with splitlines()."""
    try:
        src = LIVE_LOADER_SRC.read_text(encoding="utf-8")
    except OSError:
        return False
    uses_reader = "read_jsonl_lines(CANON_BATCH)" in src
    no_splitlines_framing = "CANON_BATCH.read_text(encoding=\"utf-8\", errors=\"replace\").splitlines()" not in src
    return uses_reader and no_splitlines_framing


def _verify_sealed_batch() -> dict[str, Any]:
    objs, malformed, nonempty = load_jsonl_objects(CANON_BATCH)
    unique = {o.get("document_id") for o in objs}
    residual = sum(1 for o in objs if r2r1._residual_pi(o.get("tokenized_content", "")))
    # SHA256 verify against sidecar (first whitespace token).
    actual_sha = hashlib.sha256(CANON_BATCH.read_bytes()).hexdigest()
    sealed_sha = ""
    if SHA_SIDECAR.is_file():
        sealed_sha = (SHA_SIDECAR.read_text(encoding="utf-8").strip().split() or [""])[0]
    sha_ok = bool(sealed_sha) and sealed_sha == actual_sha
    valid = (nonempty == EXPECTED and len(objs) == EXPECTED and len(unique) == EXPECTED
             and malformed == 0 and residual == 0 and sha_ok)
    return {"physical_newline_record_count": nonempty, "parseable_record_count": len(objs),
            "unique_doc_ids": len(unique), "malformed_record_count": malformed,
            "residual_pi_failures": residual, "sha256_sidecar_verified": sha_ok,
            "sha256_prefix": "sha256:" + actual_sha[:16], "sealed_batch_valid": valid}


def _detect_project() -> str:
    for var in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "MEDAI_VERTEX_PROJECT_ID"):
        v = os.environ.get(var)
        if v:
            return v.strip()
    cfg = Path(os.path.expandvars(r"%APPDATA%\gcloud\configurations\config_default"))
    try:
        if cfg.is_file():
            for line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = re.match(r"\s*project\s*=\s*(\S+)", line)
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    return "unknown"


def _credential_preflight() -> tuple[bool, str, bool]:
    """Returns (passed, adc_refresh_result, google_auth_importable). No model call, no
    token printed."""
    try:
        import google.auth  # type: ignore
        import google.auth.transport.requests  # type: ignore
    except Exception:
        return False, "not_attempted", False
    try:
        creds, _proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except Exception as exc:
        return False, ("login_required" if "DefaultCredentials" in type(exc).__name__ else "fail"), True
    try:
        creds.refresh(google.auth.transport.requests.Request())
    except Exception as exc:
        low = str(exc).lower()
        return False, ("login_required" if any(k in low for k in ("reauth", "invalid_grant", "expired", "login")) else "fail"), True
    return (bool(str(getattr(creds, "token", "") or "").strip()),
            "pass" if str(getattr(creds, "token", "") or "").strip() else "fail", True)


def run() -> dict[str, Any]:
    loader_hardened = _live_loader_hardened()
    verify = _verify_sealed_batch()
    project = _detect_project()
    cred_pass, adc_result, ga_importable = _credential_preflight()

    ready = bool(verify["sealed_batch_valid"] and cred_pass)
    privacy_result = "passed" if verify["sealed_batch_valid"] else "blocked"

    s = {
        "block": "MEDAI-AI-FIRST-CORPUS-JSONL-LOADER-HARDENING-AND-CREDENTIAL-PREFLIGHT-17C-R2-R2",
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
        "jsonl_splitlines_usage_removed_from_live_loader": loader_hardened,
        "jsonl_physical_newline_reader_used": True,
        "canonical_batch_path": CANON_BATCH_LABEL,
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
        "google_auth_importable": ga_importable,
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
    }
    return s


def _loader_md(s: dict[str, Any]) -> str:
    return "\n".join(["# JSONL loader hardening (public)", "",
                      f"- jsonl_splitlines_usage_removed_from_live_loader: `{s['jsonl_splitlines_usage_removed_from_live_loader']}`",
                      f"- jsonl_physical_newline_reader_used: `{s['jsonl_physical_newline_reader_used']}`", "",
                      "A shared reader (`execution/jsonl_framing.py`) frames JSONL on the physical",
                      "newline only and never uses `splitlines()`. The 17C-R2 live batch loader and the",
                      "17B-R2-R1 JSONL framing loads now use it. Validation rules are unchanged.", ""])


def _integrity_public(s: dict[str, Any]) -> dict[str, Any]:
    return {"block": s["block"], "sha256_sidecar_verified": s["sha256_sidecar_verified"],
            "sha256_prefix": s["sha256_prefix"], "physical_newline_record_count": s["physical_newline_record_count"],
            "parseable_record_count": s["parseable_record_count"], "unique_doc_ids": s["unique_doc_ids"],
            "malformed_record_count": s["malformed_record_count"], "residual_pi_failures": s["residual_pi_failures"],
            "sealed_batch_valid": s["sealed_batch_valid"], "combined_batch_count": s["combined_batch_count"],
            "old_12_added_separately": s["old_12_added_separately"],
            "note": "Counts/hashes only; the full SHA256 lives in the private sidecar."}


def _cred_public(s: dict[str, Any]) -> dict[str, Any]:
    return {"credential_preflight_attempted": s["credential_preflight_attempted"],
            "credential_preflight_passed": s["credential_preflight_passed"],
            "google_auth_importable": s["google_auth_importable"], "project_detected": s["project_detected"],
            "adc_refresh_result": s["adc_refresh_result"], "vertex_model_call_made": False,
            "note": "No model call; no token or credential printed."}


def _rerun_gate_md(s: dict[str, Any]) -> str:
    return "\n".join(["# 17C-R2 rerun entry gate (after loader hardening + preflight)", "",
                      f"- sealed_batch_valid: `{s['sealed_batch_valid']}`",
                      f"- credential_preflight_passed: `{s['credential_preflight_passed']}` (adc: `{s['adc_refresh_result']}`)",
                      f"- ready_to_rerun_17c_r2_live: `{s['ready_to_rerun_17c_r2_live']}`", "",
                      "17C-R2 is NOT re-run here. Re-run only when the sealed batch is valid AND the",
                      "credential preflight passes, keeping the run chunked/capped with",
                      "stop-on-first-failure and the live gate scoped to each chunk.", ""])


def _implementation(s: dict[str, Any]) -> str:
    lines = [f"# {s['block']}", "",
             f"## Result: **{'PASS' if s['privacy_result'] == 'passed' else 'BLOCKED'}** (safety: {s['safety_result']})",
             "", "## Metrics", ""]
    for k in ("jsonl_splitlines_usage_removed_from_live_loader", "jsonl_physical_newline_reader_used",
              "sha256_sidecar_verified", "sealed_batch_valid", "request_count_loaded",
              "physical_newline_record_count", "parseable_record_count", "unique_doc_ids",
              "malformed_record_count", "request_validation_passed", "residual_pi_failures",
              "old_12_added_separately", "combined_batch_count", "credential_preflight_attempted",
              "credential_preflight_passed", "project_detected", "adc_refresh_result",
              "ready_to_rerun_17c_r2_live", "provider_model_call_made", "vertex_model_call_made",
              "gemini_call_made", "billing_api_call_made", "live_gate_set", "live_extraction_started",
              "mkb_db_opened", "active_mkb_write", "private_outbound_requests_committed",
              "tokenized_payloads_written_to_repo", "raw_ocr_written_to_repo", "token_maps_written_to_repo",
              "private_identifier_values_written_to_repo", "credential_or_token_written_to_repo",
              "public_report_phi_leak_count", "privacy_result", "safety_result"):
        lines.append(f"- {k}: `{s[k]}`")
    nxt = ("rerun 17C-R2 live extraction once" if s["ready_to_rerun_17c_r2_live"]
           else "fix credential/batch preflight first")
    lines += ["", "## Recommended next (no live run started)", "", f"- {nxt}.", ""]
    return "\n".join(lines)


def main() -> int:
    s = run()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    loader_md = _loader_md(s)
    integrity_pub = _integrity_public(s)
    cred_pub = _cred_public(s)
    rerun_md = _rerun_gate_md(s)
    impl_md = _implementation(s)

    public_blob = "\n".join([json.dumps(s), loader_md, json.dumps(integrity_pub), json.dumps(cred_pub), rerun_md, impl_md])
    for marker in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", '"tokenized_content"'):
        if marker in public_blob:
            s["public_report_phi_leak_count"] = 1
            s["safety_result"] = "blocked"
            break

    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(_implementation(s), encoding="utf-8")
    (REPORT_DIR / "jsonl_loader_hardening_public.md").write_text(loader_md, encoding="utf-8")
    (REPORT_DIR / "sealed_batch_integrity_public.json").write_text(json.dumps(integrity_pub, indent=2), encoding="utf-8")
    (REPORT_DIR / "credential_preflight_public.json").write_text(json.dumps(cred_pub, indent=2), encoding="utf-8")
    (REPORT_DIR / "rerun_17c_r2_entry_gate.md").write_text(rerun_md, encoding="utf-8")

    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    ok = (s["safety_result"] == "passed" and s["public_report_phi_leak_count"] == 0
          and s["provider_model_call_made"] is False and s["sealed_batch_valid"] is True
          and s["jsonl_splitlines_usage_removed_from_live_loader"] is True and docs_ok)
    print("medai_ai_first_corpus_jsonl_loader_hardening_and_credential_preflight_17c_r2_r2_"
          + ("pass" if ok and s["privacy_result"] == "passed" else "blocked"))
    print(json.dumps({k: s[k] for k in (
        "jsonl_splitlines_usage_removed_from_live_loader", "sha256_sidecar_verified", "sealed_batch_valid",
        "physical_newline_record_count", "parseable_record_count", "unique_doc_ids", "malformed_record_count",
        "residual_pi_failures", "credential_preflight_passed", "project_detected", "adc_refresh_result",
        "ready_to_rerun_17c_r2_live", "public_report_phi_leak_count", "privacy_result", "safety_result")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
