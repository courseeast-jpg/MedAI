#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-17C-R2-AUTHORIZED-COST-CAP-UPDATE-17C-R2-R5.

Records the authorized total-cost-cap update (0.25 -> 0.40, per-chunk stays 0.05) and
runs a local preflight: confirms the 17C-R2 live runner now carries the new cap, the
latest estimate is within it, the canonical 478 batch is present + sealed-valid, and the
Vertex credential preflight passes. Local/config/preflight only: NO model call, NO
content request, NO billing, NO live gate, NO live extraction, NO MKB.

Public reports carry the cap values, the estimate, booleans, and status only — never
request bodies, token maps, raw OCR, PI values, credentials, or tokens.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.canonical_batch_paths import resolve_canonical_batch
from execution.jsonl_framing import load_jsonl_objects
import scripts.run_medai_ai_first_corpus_478_batch_validation_repair_local_only_17b_r2_r1 as r2r1
import scripts.run_medai_ai_first_corpus_jsonl_loader_hardening_and_credential_preflight_17c_r2_r2 as r2r2
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live

EXPECTED = 478
TARGET_PROJECT = "sot-knowledge-ocr"
PREVIOUS_CAP_TOTAL = 0.25
AUTHORIZED_CAP_TOTAL = 0.40
LATEST_ESTIMATE = 0.325362

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_authorized_cost_cap_update_17c_r2_r5"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_AUTHORIZED_COST_CAP_UPDATE_17C_R2_R5"
REQUIRED_DOCS = (
    "MEDAI_17C_R2_AUTHORIZED_COST_CAP_UPDATE_17C_R2_R5.md",
    "MEDAI_17C_R2_COST_CAP_GUARD_POLICY_R5.md",
    "MEDAI_17C_R2_LIVE_RERUN_ENTRY_GATE_AFTER_R5.md",
)


def _verify_sealed_batch() -> tuple[bool, int]:
    path, exists = resolve_canonical_batch()
    if not exists or not path.is_file() or not os.access(path, os.R_OK):
        return False, 0
    objs, malformed, nonempty = load_jsonl_objects(path)
    unique = len({o.get("document_id") for o in objs})
    residual = sum(1 for o in objs if r2r1._residual_pi(o.get("tokenized_content", "")))
    sidecar = path.parent / "outbound_requests_private.sha256"
    sha_ok = False
    if sidecar.is_file():
        import hashlib
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        sealed = (sidecar.read_text(encoding="utf-8").strip().split() or [""])[0]
        sha_ok = bool(sealed) and sealed == actual
    valid = (nonempty == EXPECTED and len(objs) == EXPECTED and unique == EXPECTED
             and malformed == 0 and residual == 0 and sha_ok)
    return valid, nonempty


def run() -> dict[str, Any]:
    # Confirm the live runner now carries the new cap (read from the live module).
    cap_total_in_runner = float(getattr(live, "CAP_TOTAL", -1))
    cap_per_chunk_in_runner = float(getattr(live, "CAP_PER_CHUNK", -1))
    chunk_size_in_runner = int(getattr(live, "CHUNK_SIZE", -1))
    model_in_runner = getattr(live, "TARGET_MODEL", "")

    sealed_valid, loaded = _verify_sealed_batch()
    within_cap = LATEST_ESTIMATE <= AUTHORIZED_CAP_TOTAL
    cred_pass, adc_result, _ = r2r2._credential_preflight()
    project = r2r2._detect_project()

    caps_consistent = (cap_total_in_runner == AUTHORIZED_CAP_TOTAL
                       and cap_per_chunk_in_runner == 0.05 and chunk_size_in_runner == 25
                       and model_in_runner == "gemini-2.5-flash-lite")
    ready = bool(sealed_valid and cred_pass and within_cap and caps_consistent)
    privacy_result = "passed" if (sealed_valid and caps_consistent) else "blocked"

    return {
        "block": "MEDAI-AI-FIRST-CORPUS-17C-R2-AUTHORIZED-COST-CAP-UPDATE-17C-R2-R5",
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
        "target_model": model_in_runner,
        "chunk_size": chunk_size_in_runner,
        "hard_cost_cap_per_chunk_usd": cap_per_chunk_in_runner,
        "previous_hard_cost_cap_total_usd": PREVIOUS_CAP_TOTAL,
        "authorized_hard_cost_cap_total_usd": AUTHORIZED_CAP_TOTAL,
        "runner_total_cap_matches_authorized": cap_total_in_runner == AUTHORIZED_CAP_TOTAL,
        "caps_consistent_in_runner": caps_consistent,
        "latest_estimated_total_cost_usd": LATEST_ESTIMATE,
        "estimated_total_within_new_cap": within_cap,
        "request_count_loaded": loaded,
        "sealed_batch_valid": sealed_valid,
        "credential_preflight_passed": cred_pass,
        "project_detected": project if project == TARGET_PROJECT else (project if project != "unknown" else "unknown"),
        "adc_refresh_result": adc_result,
        "ready_to_rerun_17c_r2_live": ready,
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


def _cost_cap_public(s: dict[str, Any]) -> dict[str, Any]:
    return {"previous_hard_cost_cap_total_usd": s["previous_hard_cost_cap_total_usd"],
            "authorized_hard_cost_cap_total_usd": s["authorized_hard_cost_cap_total_usd"],
            "hard_cost_cap_per_chunk_usd": s["hard_cost_cap_per_chunk_usd"],
            "chunk_size": s["chunk_size"], "target_model": s["target_model"],
            "latest_estimated_total_cost_usd": s["latest_estimated_total_cost_usd"],
            "estimated_total_within_new_cap": s["estimated_total_within_new_cap"],
            "runner_total_cap_matches_authorized": s["runner_total_cap_matches_authorized"],
            "caps_consistent_in_runner": s["caps_consistent_in_runner"],
            "note": "Local estimate only; no billing API was called."}


def _rerun_md(s: dict[str, Any]) -> str:
    return "\n".join(["# 17C-R2 live rerun entry gate (after R5 cap update)", "",
                      f"- authorized_hard_cost_cap_total_usd: `{s['authorized_hard_cost_cap_total_usd']}` (was `{s['previous_hard_cost_cap_total_usd']}`)",
                      f"- hard_cost_cap_per_chunk_usd: `{s['hard_cost_cap_per_chunk_usd']}` | chunk_size: `{s['chunk_size']}`",
                      f"- latest_estimated_total_cost_usd: `{s['latest_estimated_total_cost_usd']}` | within_new_cap: `{s['estimated_total_within_new_cap']}`",
                      f"- sealed_batch_valid: `{s['sealed_batch_valid']}` | credential_preflight_passed: `{s['credential_preflight_passed']}` (adc: `{s['adc_refresh_result']}`)",
                      f"- ready_to_rerun_17c_r2_live: `{s['ready_to_rerun_17c_r2_live']}`", "",
                      "17C-R2 is NOT re-run here. Recommended order: run the R4 operator-context restore",
                      "immediately before the live run (the private batch can be deleted between blocks),",
                      "then run 17C-R2 once. The runner re-verifies the batch at preflight and enforces",
                      "the $0.05/chunk and $0.40 total caps with stop-on-first-failure.", ""])


def _safety_md(s: dict[str, Any]) -> str:
    return "\n".join(["# 17C-R2-R5 safety boundary", "",
                      "- No Gemini/Vertex/Claude/OpenAI model call; no provider content request.",
                      "- No billing API call; cost caps enforced from local estimates only.",
                      "- No live gate activation; no live extraction; no corpus request sent.",
                      "- No MKB DB open/write; no auto-accept; no medical decision; no queue mutation.",
                      "- Only the total cap constant changed (0.25 -> 0.40). Per-chunk cap (0.05),",
                      "  chunk size (25), model (gemini-2.5-flash-lite), privacy validation, response",
                      "  schema, and the prompt contract are unchanged.",
                      "- No private outbound bodies, raw OCR, token maps, PI values, or credentials",
                      "  are printed or committed.", ""])


def _implementation(s: dict[str, Any]) -> str:
    lines = [f"# {s['block']}", "",
             f"## Result: **{'PASS' if s['privacy_result'] == 'passed' else 'BLOCKED'}** (safety: {s['safety_result']})",
             "", "## Metrics", ""]
    for k in ("target_model", "chunk_size", "hard_cost_cap_per_chunk_usd",
              "previous_hard_cost_cap_total_usd", "authorized_hard_cost_cap_total_usd",
              "runner_total_cap_matches_authorized", "caps_consistent_in_runner",
              "latest_estimated_total_cost_usd", "estimated_total_within_new_cap",
              "request_count_loaded", "sealed_batch_valid", "credential_preflight_passed",
              "project_detected", "adc_refresh_result", "ready_to_rerun_17c_r2_live",
              "provider_model_call_made", "vertex_model_call_made", "billing_api_call_made",
              "live_gate_set", "live_extraction_started", "mkb_db_opened", "active_mkb_write",
              "private_outbound_requests_committed", "public_report_phi_leak_count",
              "privacy_result", "safety_result"):
        lines.append(f"- {k}: `{s[k]}`")
    nxt = ("run the R4 operator-context restore, then rerun 17C-R2 live extraction once"
           if s["ready_to_rerun_17c_r2_live"] else "fix the reported cap/batch/credential blocker first")
    lines += ["", "## Recommended next (no live run started)", "", f"- {nxt}.", ""]
    return "\n".join(lines)


def main() -> int:
    s = run()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cost_pub = _cost_cap_public(s)
    rerun_md = _rerun_md(s)
    safety_md = _safety_md(s)
    impl_md = _implementation(s)

    public_blob = "\n".join([json.dumps(s), json.dumps(cost_pub), rerun_md, safety_md, impl_md])
    for marker in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", '"tokenized_content"'):
        if marker in public_blob:
            s["public_report_phi_leak_count"] = 1
            s["safety_result"] = "blocked"
            break

    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(_implementation(s), encoding="utf-8")
    (REPORT_DIR / "cost_cap_update_public.json").write_text(json.dumps(cost_pub, indent=2), encoding="utf-8")
    (REPORT_DIR / "live_rerun_entry_gate_public.md").write_text(rerun_md, encoding="utf-8")
    (REPORT_DIR / "safety_boundary_public.md").write_text(safety_md, encoding="utf-8")

    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    ok = (s["safety_result"] == "passed" and s["public_report_phi_leak_count"] == 0
          and s["provider_model_call_made"] is False and s["caps_consistent_in_runner"] is True
          and s["estimated_total_within_new_cap"] is True and docs_ok)
    print("medai_ai_first_corpus_17c_r2_authorized_cost_cap_update_17c_r2_r5_"
          + ("pass" if ok and s["privacy_result"] == "passed" else "blocked"))
    print(json.dumps({k: s[k] for k in (
        "previous_hard_cost_cap_total_usd", "authorized_hard_cost_cap_total_usd",
        "hard_cost_cap_per_chunk_usd", "chunk_size", "latest_estimated_total_cost_usd",
        "estimated_total_within_new_cap", "caps_consistent_in_runner", "request_count_loaded",
        "sealed_batch_valid", "credential_preflight_passed", "ready_to_rerun_17c_r2_live",
        "public_report_phi_leak_count", "privacy_result", "safety_result")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
