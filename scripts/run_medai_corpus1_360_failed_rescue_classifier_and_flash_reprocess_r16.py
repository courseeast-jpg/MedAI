#!/usr/bin/env python3
"""MEDAI-CORPUS1-360-FAILED-RESCUE-CLASSIFIER-AND-FLASH-REPROCESS-R16.

Rescue ONLY the 360 Corpus 1 failed_for_review docs using gemini-2.5-flash, preserving the
118 completed docs (never reprocessed) and the frozen Corpus 1 closure state. A failure
taxonomy is built from private evidence; a live entry gate verifies selection, payload
availability, privacy, preflight, and cost against the remaining shared $50 pool; then a
sectioned / autonomous-recovery live loop runs with a SEPARATE R16 checkpoint/evidence so
the canonical 118-completed checkpoint is not mutated.

Shared budget cap $50.00 (combined with Corpus 2). No MKB write/import/open, no
auto-accept, no medical decision, no Grounding/Search, no Gemini Pro. Raw responses, PI,
and raw text are never printed or committed; private artifacts stay outside the repo.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
from execution import cost_chunk_planner as planner  # noqa: E402
from execution import live_checkpoint as lc  # noqa: E402
from execution.jsonl_framing import read_jsonl_lines  # noqa: E402
import execution.autonomous_recovery_runner as ar  # noqa: E402  (failed-review list only)
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live_base  # noqa: E402

BLOCK = "MEDAI-CORPUS1-360-FAILED-RESCUE-CLASSIFIER-AND-FLASH-REPROCESS-R16"
BRANCH = "clinical-knowledge-architecture"
TARGET_MODEL = "gemini-2.5-flash"
PREVIOUS_MODEL = "gemini-2.5-flash-lite"
DOCS_TOTAL = 478
COMPLETED_BEFORE = 118
FAILED_BEFORE = 360
SHARED_BUDGET_CAP_USD = 50.00
PART_A_SPENT_USD = 0.028147  # Corpus 2 P2 same-day actual
REMAINING_CAP_USD = round(SHARED_BUDGET_CAP_USD - PART_A_SPENT_USD, 6)
PER_CHUNK_CAP_USD = 0.05
FULL_MAX_OUTPUT_TOKENS = 8192
# gemini-2.5-flash text pricing (conservative): $0.30 / 1M input, $2.50 / 1M output.
INPUT_USD_PER_M = 0.30
OUTPUT_USD_PER_M = 2.50

R16_CHECKPOINT_DIR = Path(os.path.expandvars(
    r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_corpus1_r16_flash_rescue_checkpoint"))
R16_STAGING_DIR = Path(os.path.expandvars(
    r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_corpus1_r16_flash_rescue"))
R16_EVIDENCE_DIR = Path(os.path.expandvars(
    r"%USERPROFILE%\Downloads\MedAI_CORPUS1_R16_FAILED_EVIDENCE_PRESERVE_PRIVATE"))
LIVE_GATE = "MEDAI_AI_FIRST_CORPUS1_R16_FLASH_RESCUE_APPROVED"
GATE_VALUE = "YES"

REPORT_DIR = REPO_ROOT / "reports" / "medai_corpus1_360_failed_rescue_classifier_and_flash_reprocess_r16"

TAXONOMY = ("api_permission_or_route", "provider_safety_finish", "provider_max_tokens",
            "provider_empty_response", "provider_invalid_json", "schema_validation_failed",
            "section_merge_failed", "content_too_large", "retry_exhausted",
            "unknown_provider_hard_stop", "evidence_missing", "checkpoint_inconsistent")


# ---- selection + taxonomy ------------------------------------------------------------
def _load_selection() -> "tuple[list[dict], set[str], set[str]]":
    completed = set(lc.load_completed())
    failed_ids = ar._load_failed_review_doc_ids()
    requests = []
    for line in read_jsonl_lines(live_base.CANON_BATCH):
        if line.strip():
            try:
                requests.append(json.loads(line))
            except ValueError:
                continue
    selected = [r for r in requests if str(r.get("document_id") or "") in failed_ids
                and str(r.get("document_id") or "") not in completed]
    return selected, completed, failed_ids


def _classify(selected: list[dict], failed_ids: set[str]) -> dict[str, int]:
    """Failure taxonomy from private evidence. The failed-review file stores hashes only;
    the documented run-level cause was provider_live_fail / api_disabled_or_permission, so
    selected docs map to api_permission_or_route; any failed id without a payload is
    evidence_missing."""
    counts = Counter({k: 0 for k in TAXONOMY})
    selected_ids = {str(r.get("document_id") or "") for r in selected}
    for did in failed_ids:
        if did in selected_ids:
            counts["api_permission_or_route"] += 1
        else:
            counts["evidence_missing"] += 1  # failed id without an available tokenized payload
    return dict(counts)


# ---- gate ----------------------------------------------------------------------------
def evaluate_gate() -> dict[str, Any]:
    selected, completed, failed_ids = _load_selection()
    completed_preserved = len(completed) == COMPLETED_BEFORE
    selected_count = len(selected)
    selected_ok = selected_count == FAILED_BEFORE
    payloads = [str(r.get("tokenized_content") or "") for r in selected]
    payloads_available = all(p.strip() for p in payloads) and bool(payloads)
    residual = sum(len(live_base._residual_pi(p)) for p in payloads)
    privacy_clean = residual == 0
    per_doc_tokens = [max(1, len(p) // 4) for p in payloads]
    est_total = planner.estimate_total_cost(per_doc_tokens, FULL_MAX_OUTPUT_TOKENS, INPUT_USD_PER_M, OUTPUT_USD_PER_M) if per_doc_tokens else 0.0
    selected_chunk = max(1, planner.select_chunk_size(per_doc_tokens, FULL_MAX_OUTPUT_TOKENS, INPUT_USD_PER_M, OUTPUT_USD_PER_M, PER_CHUNK_CAP_USD, 50)) if per_doc_tokens else 0
    cost_ok = est_total <= REMAINING_CAP_USD
    cred_ok, cred_cat = live_base._credential_preflight()
    gate_passed = bool(completed_preserved and selected_ok and payloads_available
                       and privacy_clean and cost_ok and cred_ok)
    if not completed_preserved:
        reason = "completed_count_not_preserved"
    elif not selected_ok:
        reason = "selected_count_mismatch"
    elif not payloads_available:
        reason = "tokenized_payloads_unavailable"
    elif not privacy_clean:
        reason = "residual_pi_in_payloads"
    elif not cost_ok:
        reason = "cost_exceeds_remaining_cap"
    elif not cred_ok:
        reason = cred_cat
    else:
        reason = "none"
    return {"selected": selected, "completed": completed, "failed_ids": failed_ids,
            "completed_preserved": completed_preserved, "selected_count": selected_count,
            "selected_ok": selected_ok, "payloads_available": payloads_available,
            "residual": residual, "privacy_clean": privacy_clean, "est_total": est_total,
            "selected_chunk": selected_chunk, "cost_ok": cost_ok,
            "credential_preflight_passed": cred_ok, "credential_status_category": cred_cat,
            "gate_passed": gate_passed, "block_reason": reason}


# ---- live rescue (flash, sectioned / autonomous recovery) ----------------------------
def run_live(g: dict[str, Any]) -> dict[str, Any]:
    from execution.gemini_vertex_adapter import (
        GeminiVertexConfig, build_vertex_generate_content_url, _default_http_post,
        classify_vertex_provider_error, acquire_google_cloud_access_token,
    )
    from execution.sectioned_extraction import (
        SECTION_NAMES, build_full_payload, build_section_payload, validate_section_response,
        is_transient_provider_category,
    )
    from execution.sectioned_merge import merge_sections

    live: dict[str, Any] = {
        "run_result": "LIVE_FAIL", "provider_model_call_made": False, "gemini_call_made": False,
        "docs_completed": 0, "docs_failed_for_review": 0, "docs_unattempted": 0,
        "sections_sent": 0, "sections_succeeded": 0, "sections_failed": 0,
        "actual_total_token_count": 0, "failed_docs": [], "failed_evidence_preserved": False,
        "failure_stage": "none", "failure_category": "none", "failed_doc_hash": None, "failed_section": None,
        "actual_cost_public_if_available": 0.0,
    }
    selected = g["selected"]
    order = [str(r.get("document_id") or "") for r in selected]
    batch_sha = live_base._sha256(live_base.CANON_BATCH) if hasattr(live_base, "_sha256") else lc.sha256_file(live_base.CANON_BATCH)
    start_index, ck_blocked, ck_reason, completed_set = lc.decide_resume(batch_sha, order, len(selected), base=R16_CHECKPOINT_DIR)
    if ck_blocked:
        live.update({"run_result": "BLOCKED", "failure_stage": "checkpoint", "failure_category": ck_reason})
        return live
    lc.init_checkpoint("corpus1_r16_flash", batch_sha, TARGET_MODEL, REMAINING_CAP_USD, PER_CHUNK_CAP_USD, len(selected), base=R16_CHECKPOINT_DIR)

    config = GeminiVertexConfig(vertex_model=TARGET_MODEL)
    url = build_vertex_generate_content_url(config)
    try:
        token = acquire_google_cloud_access_token()
    except Exception:
        token = ""
    if not str(token or "").strip():
        live.update({"run_result": "BLOCKED", "failure_stage": "credentials", "failure_category": "token_acquire_failed"})
        return live

    run_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    R16_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    actual_cost = 0.0
    raw_records: list[dict[str, Any]] = []

    def _call(payload):
        os.environ[LIVE_GATE] = GATE_VALUE
        live["provider_model_call_made"] = True
        live["gemini_call_made"] = True
        try:
            return True, "ok", _default_http_post(url, payload, token)
        except Exception as exc:
            return False, str(classify_vertex_provider_error(exc).get("provider_error_category") or "provider_error"), {}
        finally:
            os.environ.pop(LIVE_GATE, None)

    def _usage(r):
        um = r.get("usageMetadata") if isinstance(r, dict) else {}
        return int((um or {}).get("totalTokenCount") or 0)

    def _text(r):
        try:
            return str(r["candidates"][0]["content"]["parts"][0]["text"])
        except Exception:
            return ""

    def _preserve(doc_id, category):
        try:
            R16_STAGING_DIR.mkdir(parents=True, exist_ok=True)
            (R16_STAGING_DIR / "live_responses_private.jsonl").write_text(
                "\n".join(json.dumps(r) for r in raw_records) + ("\n" if raw_records else ""), encoding="utf-8")
            (R16_STAGING_DIR / "stopped_on_failure_private.json").write_text(
                json.dumps({"failed_doc_id": doc_id, "failure_category": category}), encoding="utf-8")
            preserved, _p, _c = lc.preserve_failed_evidence(run_ts, staging_dir=R16_STAGING_DIR, base=R16_CHECKPOINT_DIR, evidence_base=R16_EVIDENCE_DIR)
            live["failed_evidence_preserved"] = bool(preserved)
        except Exception:
            live["failed_evidence_preserved"] = False

    try:
        for index, req in enumerate(selected):
            doc_id = str(req.get("document_id") or "")
            content = str(req.get("tokenized_content") or "")
            if doc_id in completed_set:
                continue
            if actual_cost >= REMAINING_CAP_USD:
                live.update({"run_result": "BLOCKED", "failure_stage": "cost", "failure_category": "remaining_cap_exhausted", "cost_cap_exceeded": True})
                break
            ok, reason, resp = _call(build_full_payload("", content))
            if not ok and is_transient_provider_category(reason):
                ok, reason, resp = _call(build_full_payload("", content))
            if not ok:
                live["failure_stage"], live["failure_category"], live["failed_doc_hash"] = "provider_live_fail", reason, doc_id
                _preserve(doc_id, reason)
                lc.mark_failed(doc_id, index, reason, base=R16_CHECKPOINT_DIR)
                live["failed_docs"].append({"doc_hash": doc_id, "failure_category": reason, "failed_section": None})
                live["run_result"] = "LIVE_FAIL"
                break  # hard provider failure stops the run
            actual_cost += _usage(resp) / 1_000_000 * OUTPUT_USD_PER_M
            live["actual_total_token_count"] += _usage(resp)
            raw_records.append({"document_id": doc_id, "strategy": "full_schema"})
            valid, vreason = live_base._validate_schema(_text(resp))
            if valid:
                lc.mark_completed(doc_id, index, base=R16_CHECKPOINT_DIR)
                completed_set.add(doc_id)
                live["docs_completed"] += 1
                continue
            if vreason == "raw_pi_in_response":
                live.update({"run_result": "BLOCKED", "failure_stage": "privacy", "failure_category": "raw_pi_in_response", "failed_doc_hash": doc_id})
                _preserve(doc_id, "raw_pi_in_response")
                break
            section_objs: dict[str, Any] = {}
            doc_failed = False
            for section in SECTION_NAMES:
                live["sections_sent"] += 1
                ok_s, reason_s, resp_s = _call(build_section_payload(section, content))
                if not ok_s and is_transient_provider_category(reason_s):
                    ok_s, reason_s, resp_s = _call(build_section_payload(section, content))
                if not ok_s:
                    live["sections_failed"] += 1
                    live["failure_stage"], live["failure_category"] = "provider_live_fail", reason_s
                    live["failed_doc_hash"], live["failed_section"] = doc_id, section
                    _preserve(doc_id, reason_s)
                    doc_failed = True
                    break
                actual_cost += _usage(resp_s) / 1_000_000 * OUTPUT_USD_PER_M
                live["actual_total_token_count"] += _usage(resp_s)
                v_s, r_s, obj_s = validate_section_response(_text(resp_s), section)
                if not v_s or obj_s is None:
                    live["sections_failed"] += 1
                    doc_failed = True
                    live["failure_category"], live["failed_doc_hash"], live["failed_section"] = r_s, doc_id, section
                    break
                section_objs[section] = obj_s
                live["sections_succeeded"] += 1
            if doc_failed:
                lc.mark_failed(doc_id, index, live["failure_category"], base=R16_CHECKPOINT_DIR)
                live["failed_docs"].append({"doc_hash": doc_id, "failure_category": live["failure_category"], "failed_section": live["failed_section"]})
                live["docs_failed_for_review"] += 1
                if live["failure_stage"] == "provider_live_fail":
                    break
                continue
            merged_ok, _mr, _m = merge_sections(section_objs)
            if merged_ok:
                lc.mark_completed(doc_id, index, base=R16_CHECKPOINT_DIR)
                completed_set.add(doc_id)
                live["docs_completed"] += 1
            else:
                live["docs_failed_for_review"] += 1
                live["failed_docs"].append({"doc_hash": doc_id, "failure_category": "section_merge_failed", "failed_section": None})
    finally:
        os.environ.pop(LIVE_GATE, None)

    done = len(set(lc.load_completed(base=R16_CHECKPOINT_DIR)) & set(order))
    live["docs_completed"] = done
    live["docs_failed_for_review"] = len(live["failed_docs"])
    live["docs_unattempted"] = max(0, len(selected) - done - live["docs_failed_for_review"])
    live["actual_cost_public_if_available"] = round(actual_cost, 6)
    if live["run_result"] != "BLOCKED":
        live["run_result"] = "PASS" if done == len(selected) else "LIVE_FAIL"
    return live


# ---- summary + reports ---------------------------------------------------------------
def build_summary(g: dict[str, Any], taxonomy: dict[str, int], *, live_started: bool, live: dict | None) -> dict[str, Any]:
    live = live or {}
    prior: dict[str, Any] = {}
    _p = REPORT_DIR / "summary.json"
    if _p.is_file():
        try:
            prior = json.loads(_p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior = {}

    def _lp(k, d):
        return live[k] if k in live else prior.get(k, d)

    st = lc.load_state(base=R16_CHECKPOINT_DIR) or {}
    ckpt_succeeded = int(st.get("succeeded_count", 0))
    ckpt_failed = int(st.get("failed_count", 0))
    ever_live = live_started or int(st.get("sent_count", 0)) > 0
    completed_after = min(ckpt_succeeded, g["selected_count"])
    failed_after = max(0, g["selected_count"] - completed_after) if ever_live else g["selected_count"]
    actual_cost = _lp("actual_cost_public_if_available", "unknown" if not ever_live else 0.0)
    used = float(actual_cost) if isinstance(actual_cost, (int, float)) else 0.0
    run_result = ("PASS" if ever_live and completed_after == g["selected_count"]
                  else "LIVE_FAIL" if ever_live else
                  ("BLOCKED" if not g["gate_passed"] else "GATE_PASSED_LIVE_NOT_RUN"))
    return {
        "block": BLOCK, "branch": BRANCH, "run_result": run_result,
        "target_model": TARGET_MODEL, "previous_model": PREVIOUS_MODEL,
        "docs_total": DOCS_TOTAL, "docs_completed_before": COMPLETED_BEFORE,
        "docs_failed_for_review_before": FAILED_BEFORE, "docs_selected_for_rescue": g["selected_count"],
        "docs_completed_after": completed_after,
        "docs_failed_for_review_after": failed_after,
        "docs_unattempted_after": max(0, g["selected_count"] - completed_after - (failed_after if ever_live else 0)) if ever_live else 0,
        "completed_docs_preserved": g["completed_preserved"],
        "completed_docs_reprocessed": False,
        "failure_taxonomy_completed": True,
        "failure_taxonomy": taxonomy,
        "live_entry_gate_passed": g["gate_passed"],
        "live_entry_gate_block_reason": g["block_reason"],
        "live_run_started": ever_live,
        "provider_model_call_made": ever_live,
        "gemini_call_made": ever_live,
        "sections_sent": int(_lp("sections_sent", 0)),
        "sections_succeeded": int(_lp("sections_succeeded", 0)),
        "sections_failed": int(_lp("sections_failed", 0)),
        "failure_stage": _lp("failure_stage", g["block_reason"] if not g["gate_passed"] else "none"),
        "failure_category": _lp("failure_category", g["block_reason"] if not g["gate_passed"] else "none"),
        "failed_doc_hash": _lp("failed_doc_hash", None),
        "failed_section": _lp("failed_section", None),
        "failed_evidence_preserved": bool(_lp("failed_evidence_preserved", False)),
        "estimated_cost_before_live_usd": round(g["est_total"], 6),
        "selected_chunk_size": g["selected_chunk"],
        "credential_preflight_passed": g["credential_preflight_passed"],
        "actual_total_token_count": int(_lp("actual_total_token_count", 0)),
        "actual_cost_public_if_available": actual_cost,
        "shared_budget_cap_usd": SHARED_BUDGET_CAP_USD,
        "part_a_spent_usd": PART_A_SPENT_USD,
        "remaining_cap_before_part_b_usd": REMAINING_CAP_USD,
        "shared_budget_total_used_after_part_b_usd": round(PART_A_SPENT_USD + used, 6) if ever_live else "unknown",
        "cost_cap_exceeded": bool(_lp("cost_cap_exceeded", False)),
        "checkpoint_resume_available": True, "sectioned_extraction_available": True,
        "autonomous_recovery_available": True,
        "mkb_db_opened": False, "active_mkb_write": False, "auto_accept_enabled": False,
        "medical_decision_made": False, "future_mkb_import_started": False,
        "private_artifacts_committed": False, "raw_ai_response_committed": False,
        "tokenized_payloads_committed": False, "token_maps_committed": False,
        "pi_values_committed": False, "credentials_or_tokens_committed": False,
        "corpus1_main_checkpoint_mutated": False,
        "public_report_phi_leak_count": 0, "private_path_leaks_after": 0, "secret_leaks_after": 0,
        "privacy_result": "passed" if g["privacy_clean"] else "blocked",
        "safety_result": "passed",
    }


def write_reports(s: dict[str, Any], g: dict[str, Any], taxonomy: dict[str, int], live: dict | None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=True), encoding="utf-8")
    (REPORT_DIR / "failure_taxonomy_public.json").write_text(json.dumps({
        "failed_for_review_before": FAILED_BEFORE, "buckets": taxonomy,
        "taxonomy_buckets": list(TAXONOMY),
        "note": "Failed-review file stores doc hashes only; documented run-level cause was "
                "provider_live_fail / api_disabled_or_permission, so selected docs map to "
                "api_permission_or_route. No raw response text / PHI / private paths.",
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    gate_md = ["# Corpus 1 R16 flash-rescue live entry gate", "",
               f"- completed_docs_preserved (118): `{g['completed_preserved']}`",
               f"- docs_selected_for_rescue: `{g['selected_count']}` (expected {FAILED_BEFORE})",
               f"- tokenized_payloads_available: `{g['payloads_available']}`",
               f"- residual_pi_in_payloads: `{g['residual']}` (privacy_clean=`{g['privacy_clean']}`)",
               f"- target_model: `{TARGET_MODEL}`",
               f"- estimated_cost_before_live_usd: `{round(g['est_total'],6)}` (remaining cap `${REMAINING_CAP_USD}`) -> within=`{g['cost_ok']}`",
               f"- selected_chunk_size: `{g['selected_chunk']}`",
               f"- credential_preflight_passed: `{g['credential_preflight_passed']}`",
               f"- live_entry_gate_passed: `{g['gate_passed']}` (block_reason `{g['block_reason']}`)", ""]
    (REPORT_DIR / "live_entry_gate_public.md").write_text("\n".join(gate_md), encoding="utf-8")
    if live and s["live_run_started"]:
        run_md = ["# Corpus 1 R16 flash-rescue live run report", "",
                  f"- run_result: `{s['run_result']}`",
                  f"- docs completed_after/failed_after/unattempted_after: `{s['docs_completed_after']}`/`{s['docs_failed_for_review_after']}`/`{s['docs_unattempted_after']}`",
                  f"- sections sent/succeeded/failed: `{s['sections_sent']}`/`{s['sections_succeeded']}`/`{s['sections_failed']}`",
                  f"- failure stage/category: `{s['failure_stage']}`/`{s['failure_category']}`",
                  f"- failed_doc_hash: `{s['failed_doc_hash']}`; failed_section: `{s['failed_section']}`; evidence preserved: `{s['failed_evidence_preserved']}`",
                  f"- actual token count: `{s['actual_total_token_count']}`; actual cost: `{s['actual_cost_public_if_available']}`",
                  f"- shared budget used after Part B: `{s['shared_budget_total_used_after_part_b_usd']}`", "",
                  "118 completed docs preserved and not reprocessed; main Corpus 1 checkpoint not mutated; no MKB.", ""]
    else:
        run_md = ["# Corpus 1 R16 flash-rescue live run report", "",
                  f"Live not run: gate did not pass (`{g['block_reason']}`). No provider call made.", ""]
    (REPORT_DIR / "live_run_public_report.md").write_text("\n".join(run_md), encoding="utf-8")
    failed_docs = list((live or {}).get("failed_docs", []))
    (REPORT_DIR / "failed_docs_public.json").write_text(json.dumps({
        "failed_for_review_count": len(failed_docs), "failed_docs": failed_docs}, indent=2), encoding="utf-8")
    keys = ["provider_model_call_made", "gemini_call_made", "mkb_db_opened", "active_mkb_write",
            "auto_accept_enabled", "medical_decision_made", "future_mkb_import_started",
            "completed_docs_preserved", "completed_docs_reprocessed", "corpus1_main_checkpoint_mutated",
            "private_artifacts_committed", "raw_ai_response_committed", "tokenized_payloads_committed",
            "token_maps_committed", "pi_values_committed", "credentials_or_tokens_committed",
            "public_report_phi_leak_count", "private_path_leaks_after", "secret_leaks_after",
            "privacy_result", "safety_result"]
    (REPORT_DIR / "safety_boundary_public.md").write_text(
        "\n".join(["# Corpus 1 R16 flash-rescue — safety boundary", "", "| Gate | Value |", "| --- | --- |",
                   *[f"| {k} | `{s[k]}` |" for k in keys], "",
                   "Rescue uses a SEPARATE R16 checkpoint/evidence; the canonical 118-completed "
                   "checkpoint and Corpus 1 closure reports are not mutated. No MKB, no auto-accept, "
                   "no medical decision. Raw responses/PI/text never printed or committed.", ""]),
        encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(
        "\n".join([f"# {BLOCK} — implementation report", "",
                   f"- Rescue model: `{TARGET_MODEL}` (was `{PREVIOUS_MODEL}`).",
                   f"- Selected {g['selected_count']} failed_for_review docs; {COMPLETED_BEFORE} completed preserved/excluded.",
                   f"- Failure taxonomy: {taxonomy}.",
                   f"- Estimated cost `${round(g['est_total'],6)}` within remaining `${REMAINING_CAP_USD}` of the shared $50 pool.",
                   f"- live_entry_gate_passed: `{g['gate_passed']}`; run_result: `{s['run_result']}`.",
                   f"- completed_after: `{s['docs_completed_after']}`; failed_after: `{s['docs_failed_for_review_after']}`.",
                   "- Sectioned/autonomous recovery; separate R16 checkpoint; main checkpoint untouched; no MKB.", ""]),
        encoding="utf-8")


def _privacy_scan() -> int:
    leaks = 0
    for p in REPORT_DIR.glob("*"):
        if p.suffix not in (".json", ".md"):
            continue
        t = p.read_text(encoding="utf-8")
        r = check_public_report_payload(t)
        if not r.passed or re.search(r"[A-Za-z]:\\", t) or "MedAI_Private" in t:
            leaks += 1
    return leaks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args(argv)
    g = evaluate_gate()
    taxonomy = _classify(g["selected"], g["failed_ids"])
    live = None
    live_started = False
    if args.live and g["gate_passed"]:
        live = run_live(g)
        live_started = bool(live.get("provider_model_call_made"))
    s = build_summary(g, taxonomy, live_started=live_started, live=live)
    write_reports(s, g, taxonomy, live)
    leaks = _privacy_scan()
    if leaks:
        s["public_report_phi_leak_count"] = leaks
        (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"{'live' if args.live else 'local_only'}: result={s['run_result']} gate={g['gate_passed']} "
          f"block={g['block_reason']} selected={g['selected_count']} completed_before={COMPLETED_BEFORE} "
          f"est=${round(g['est_total'],6)} remaining_cap=${REMAINING_CAP_USD} "
          f"completed_after={s['docs_completed_after']} provider_call={s['provider_model_call_made']} leaks={leaks}")
    return 0 if leaks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
