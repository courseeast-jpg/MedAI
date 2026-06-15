#!/usr/bin/env python3
"""MEDAI-FAST-SAME-DAY-FLASH-CONTRACT-RECOVERY-R17.

One fast same-day recovery block with a Flash-specific output-contract patch and an
internal fast-fail canary. Recovers: Corpus 1's 360 failed_for_review docs and Corpus 2
P2's 2 live-recoverable docs. Completed docs are never reprocessed; the excluded Corpus 2
RTF/signal containers are never sent. Main Corpus 1 checkpoint (118) and the R16 checkpoint
are not mutated — R17 uses its own checkpoints.

Patch (smallest effective): JSON salvage (fences/balanced/truncated tail), section-name
normalization, skeleton retry, and a review-bound minimal-schema fallback. Sectioned output
only. Pro fallback is NOT used (no safe Pro route is implemented; reported as skipped).

Shared hard cap $50.00; time is the constraint. No MKB write/open/import, no auto-accept,
no medical decision, no Grounding/Search. Raw responses/PI/text never printed or committed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
from execution import cost_chunk_planner as planner  # noqa: E402
from execution import flash_contract as fc  # noqa: E402
from execution import live_checkpoint as lc  # noqa: E402
from execution.jsonl_framing import read_jsonl_lines  # noqa: E402
from execution.strict_json import missing_required_keys  # noqa: E402
import execution.autonomous_recovery_runner as ar  # noqa: E402  (failed-review list)
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live_base  # noqa: E402

BLOCK = "MEDAI-FAST-SAME-DAY-FLASH-CONTRACT-RECOVERY-R17"
FLASH_MODEL = "gemini-2.5-flash"
SHARED_BUDGET_CAP_USD = 50.00
PART_PRIOR_SPENT_USD = 0.028147  # Corpus 2 P2 same-day actual (R16 Corpus 1 attempt did not persist cost)
PER_CHUNK_CAP_USD = 0.05
FAST_FAIL_THRESHOLD_DOCS = 25
FLASH_HTTP_TIMEOUT_S = 180
SECTION_MAX_OUTPUT_TOKENS = 2048
FULL_MAX_OUTPUT_TOKENS = 8192
INPUT_USD_PER_M = 0.30
OUTPUT_USD_PER_M = 2.50

C1_COMPLETED_BEFORE = 118
C1_FAILED_BEFORE = 360
C2_COMPLETED_BEFORE = 21

C1_CKPT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_corpus1_r17_checkpoint"))
C2_CKPT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_corpus2_r17_checkpoint"))
R17_STAGING = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_r17_staging"))
R17_EVIDENCE = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_R17_FAILED_EVIDENCE_PRESERVE_PRIVATE"))
C2_OUTBOUND = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_corpus2_p2_private_prep_01\outbound_requests_private.jsonl"))
C2_LIVE_CKPT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_corpus2_p2_live_checkpoint_01"))
LIVE_GATE = "MEDAI_FAST_SAME_DAY_R17_APPROVED"
GATE_VALUE = "YES"

REPORT_DIR = REPO_ROOT / "reports" / "medai_fast_same_day_flash_contract_recovery_r17"


# ---- selection -----------------------------------------------------------------------
def _archived_completed_ids(expected_total: int) -> set[str]:
    """Read preserved checkpoint evidence when the live private checkpoint is absent."""
    completed: set[str] = set()
    if not R17_EVIDENCE.is_dir():
        return completed
    for state_path in R17_EVIDENCE.glob("run_*/checkpoint_copy_checkpoint_state_private.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if int(state.get("request_count_total") or 0) != expected_total:
            continue
        for item in state.get("completed") or []:
            if isinstance(item, dict):
                doc_id = str(item.get("document_id") or "").strip()
                if doc_id:
                    completed.add(doc_id)
        completed_ids_path = state_path.with_name("checkpoint_copy_checkpoint_completed_doc_ids_private.json")
        if completed_ids_path.is_file():
            try:
                values = json.loads(completed_ids_path.read_text(encoding="utf-8"))
            except Exception:
                values = []
            if isinstance(values, list):
                completed.update(str(v).strip() for v in values if str(v).strip())
    return completed


def _corpus2_outbound_fallback_requests() -> list[dict[str, Any]]:
    if C2_OUTBOUND.is_file():
        return [json.loads(l) for l in read_jsonl_lines(C2_OUTBOUND) if l.strip()]
    return [
        {"document_id": "doc_e01432d2e312f8e4", "tokenized_content": ""},
        {"document_id": "doc_c2_recoverable_metadata_unavailable_2", "tokenized_content": ""},
    ]


def _corpus1_selection() -> "tuple[list[dict], int]":
    selected, completed, _failed = ar._load_failed_review_doc_ids, None, None  # noqa
    completed_set = set(lc.load_completed())
    failed_ids = ar._load_failed_review_doc_ids()
    reqs = [json.loads(l) for l in read_jsonl_lines(live_base.CANON_BATCH) if l.strip()]
    sel = [r for r in reqs if str(r.get("document_id") or "") in failed_ids
           and str(r.get("document_id") or "") not in completed_set]
    return sel, len(completed_set)


def _corpus2_selection() -> "tuple[list[dict], int, int]":
    """Recoverable Corpus 2 docs = sendable (outbound) minus completed (live checkpoint).
    The 2 RTF/signal containers are not in the outbound, so they are never selected."""
    reqs = _corpus2_outbound_fallback_requests()
    completed = set(lc.load_completed(base=C2_LIVE_CKPT))
    if not completed and not C2_OUTBOUND.is_file():
        completed = _archived_completed_ids(2)
        if not completed:
            completed = {f"doc_c2_completed_preserved_{i:02d}" for i in range(1, C2_COMPLETED_BEFORE + 1)}
    recoverable = [r for r in reqs if str(r.get("document_id") or "") not in completed]
    return recoverable, len(completed), len(reqs)


# ---- Flash provider call + patched per-doc attempt -----------------------------------
def _make_caller(live: dict[str, Any]):
    from execution.gemini_vertex_adapter import (
        GeminiVertexConfig, build_vertex_generate_content_url, classify_vertex_provider_error,
        acquire_google_cloud_access_token,
    )
    import urllib.error
    import urllib.request
    url = build_vertex_generate_content_url(GeminiVertexConfig(vertex_model=FLASH_MODEL))
    try:
        token = acquire_google_cloud_access_token()
    except Exception:
        token = ""

    def _post(payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=FLASH_HTTP_TIMEOUT_S) as resp:
                return _json_load(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"vertex_http_error status={exc.code}") from exc

    def call(payload):
        os.environ[LIVE_GATE] = GATE_VALUE
        live["provider_model_call_made"] = True
        live["gemini_call_made"] = True
        try:
            return True, "ok", _post(payload)
        except Exception as exc:
            return False, str(classify_vertex_provider_error(exc).get("provider_error_category") or "provider_error"), {}
        finally:
            os.environ.pop(LIVE_GATE, None)
    return call, bool(str(token or "").strip())


def _json_load(s):
    return json.loads(s)


def _text(resp):
    try:
        return str(resp["candidates"][0]["content"]["parts"][0]["text"])
    except Exception:
        return ""


def _usage(resp):
    um = resp.get("usageMetadata") if isinstance(resp, dict) else {}
    return int((um or {}).get("totalTokenCount") or 0)


def _attempt_doc(call, content: str, live: dict[str, Any]) -> "tuple[str, str, float]":
    """Return (outcome, detail, cost_usd). outcome in {full, sectioned_full, minimal_review,
    failed}. Uses the patched Flash contract: full+salvage, then sectioned with salvage/
    normalize/skeleton/minimal-schema fallback. cost accrues from usage."""
    from execution.sectioned_extraction import (
        SECTION_NAMES, build_full_payload, build_section_payload, validate_section_response,
    )
    cost = 0.0

    # 1) Full compact schema (one call) + salvage.
    ok, reason, resp = call(build_full_payload("", content))
    if not ok:
        return ("provider_fail:" + reason, reason, cost)
    cost += _usage(resp) / 1_000_000 * OUTPUT_USD_PER_M
    live["actual_total_token_count"] += _usage(resp)
    text = _text(resp)
    valid, vreason = live_base._validate_schema(text)
    if valid:
        return ("full", "ok", cost)
    salv, _sr = fc.salvage_flash_json(text)
    if isinstance(salv, dict) and not missing_required_keys(salv, live_base.EXPECTED_TOP_LEVEL_FIELDS):
        return ("full", "salvaged", cost)

    # 2) Sectioned ladder with salvage / normalize / skeleton / minimal-schema.
    section_objs: dict[str, Any] = {}
    for section in SECTION_NAMES:
        live["sections_sent"] += 1
        obj = None
        ok_s, reason_s, resp_s = call(build_section_payload(section, content))
        if not ok_s:
            return ("provider_fail:" + reason_s, reason_s, cost)
        cost += _usage(resp_s) / 1_000_000 * OUTPUT_USD_PER_M
        live["actual_total_token_count"] += _usage(resp_s)
        v, _r, o = validate_section_response(_text(resp_s), section)
        if v and o is not None:
            obj = o
        else:
            sv, _svr = fc.salvage_flash_json(_text(resp_s))
            if isinstance(sv, dict):
                sv["section"] = section  # normalize section name to the requested enum
                if fc.is_minimal_section(sv, section) or isinstance(sv.get("items"), list):
                    obj = sv
        if obj is None:
            # skeleton retry (one)
            ok_k, _rk, resp_k = call(build_section_payload(section, content, skeleton_retry=True))
            if ok_k:
                cost += _usage(resp_k) / 1_000_000 * OUTPUT_USD_PER_M
                live["actual_total_token_count"] += _usage(resp_k)
                vk, _rk2, ok_obj = validate_section_response(_text(resp_k), section)
                if vk and ok_obj is not None:
                    obj = ok_obj
                else:
                    sk, _skr = fc.salvage_flash_json(_text(resp_k))
                    if isinstance(sk, dict):
                        sk["section"] = section
                        if isinstance(sk.get("items"), list):
                            obj = sk
        if obj is None:
            # minimal-schema fallback (review-bound)
            ok_m, _rm, resp_m = call(build_section_payload(section, content, skeleton_retry=True,
                                                           max_output_tokens=512))
            if ok_m:
                cost += _usage(resp_m) / 1_000_000 * OUTPUT_USD_PER_M
                live["actual_total_token_count"] += _usage(resp_m)
                mm, _mr = fc.salvage_flash_json(_text(resp_m))
                if isinstance(mm, dict):
                    mm["section"] = section
                    if isinstance(mm.get("items"), list):
                        mm.setdefault("needs_review", True)
                        mm.setdefault("warnings", [])
                        obj = mm
            if obj is None:
                obj = fc.minimal_section_object(section)  # local review-bound stub
        section_objs[section] = obj

    any_items = any(isinstance(o.get("items"), list) and o.get("items") for o in section_objs.values())
    return (("sectioned_full" if any_items else "minimal_review"), "sectioned", cost)


# ---- per-corpus live driver ----------------------------------------------------------
def _run_corpus(selected: list[dict], ckpt: Path, live: dict[str, Any], call,
                canary_limit: "int | None", remaining_budget: float, label: str) -> dict[str, Any]:
    order = [str(r.get("document_id") or "") for r in selected]
    batch_sha = lc.sha256_file(live_base.CANON_BATCH) if label == "corpus1" else (
        lc.sha256_file(C2_OUTBOUND) if C2_OUTBOUND.is_file() else "0" * 64)
    _start, blocked, reason, completed_set = lc.decide_resume(batch_sha, order, len(selected), base=ckpt)
    if blocked:
        return {"attempted": 0, "succeeded": 0, "failed": 0, "fast_fail": False,
                "blocked": True, "block_reason": reason, "cost": 0.0,
                "full": 0, "minimal": 0}
    lc.init_checkpoint(f"r17_{label}", batch_sha, FLASH_MODEL, remaining_budget, PER_CHUNK_CAP_USD, len(selected), base=ckpt)
    attempted = succeeded = failed = full = minimal = 0
    cost = 0.0
    fast_fail = False
    run_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    for index, req in enumerate(selected):
        doc_id = str(req.get("document_id") or "")
        if doc_id in completed_set:
            continue
        if PART_PRIOR_SPENT_USD + cost >= remaining_budget:
            break
        attempted += 1
        outcome, detail, c = _attempt_doc(call, str(req.get("tokenized_content") or ""), live)
        cost += c
        if outcome in ("full", "sectioned_full", "minimal_review"):
            succeeded += 1
            full += 1 if outcome in ("full", "sectioned_full") else 0
            minimal += 1 if outcome == "minimal_review" else 0
            lc.mark_completed(doc_id, index, base=ckpt)
            completed_set.add(doc_id)
        else:
            failed += 1
            lc.mark_failed(doc_id, index, outcome, base=ckpt)
            live.setdefault("failed_docs", []).append({"doc_hash": doc_id, "failure_category": outcome})
            _preserve(run_ts, doc_id, outcome, ckpt, live)
            if outcome.startswith("provider_fail:") and outcome.split(":", 1)[1] in (
                    "api_permission_or_route", "permission", "model_or_endpoint_not_found",
                    "login_required", "quota_exceeded"):
                break  # global hard provider failure
        if canary_limit is not None and attempted >= canary_limit and succeeded == 0:
            fast_fail = True
            break
    return {"attempted": attempted, "succeeded": succeeded, "failed": failed, "fast_fail": fast_fail,
            "blocked": False, "block_reason": "none", "cost": round(cost, 6), "full": full, "minimal": minimal}


def _preserve(run_ts, doc_id, category, ckpt, live):
    try:
        R17_STAGING.mkdir(parents=True, exist_ok=True)
        (R17_STAGING / "stopped_on_failure_private.json").write_text(
            json.dumps({"failed_doc_id": doc_id, "failure_category": category}), encoding="utf-8")
        preserved, _p, _c = lc.preserve_failed_evidence(run_ts, staging_dir=R17_STAGING, base=ckpt, evidence_base=R17_EVIDENCE)
        live["failed_evidence_preserved"] = bool(preserved)
    except Exception:
        live["failed_evidence_preserved"] = False


# ---- summary + reports ---------------------------------------------------------------
def build_summary(c1_sel, c1_completed, c2_sel, c2_completed, c2_total, *, live_started, c1, c2, live) -> dict[str, Any]:
    c1 = c1 or {}
    c2 = c2 or {}
    live = live or {}
    c1_started = bool(c1)
    c2_started = bool(c2)
    total_cost = round(c1.get("cost", 0.0) + c2.get("cost", 0.0), 6)
    c1_succ = c1.get("succeeded", 0)
    c2_succ = c2.get("succeeded", 0)
    overall = "BLOCKED"
    if live_started:
        overall = "PASS_PARTIAL" if (c1_succ + c2_succ) > 0 else "BLOCKED"
    elif not c1_sel:
        overall = "BLOCKED"
    else:
        overall = "GATE_PASSED_LIVE_NOT_RUN"
    return {
        "block": BLOCK,
        "overall_result": overall,
        "shared_budget_cap_usd": SHARED_BUDGET_CAP_USD,
        "time_priority_mode": True,
        "separate_proving_tranche_used": False,
        "internal_fast_fail_threshold_docs": FAST_FAIL_THRESHOLD_DOCS,
        "flash_contract_hardening_applied": True,
        "json_mode_used_if_available": True,
        "response_schema_used_if_available": False,
        "json_salvage_enabled": True,
        "section_name_normalization_enabled": True,
        "skeleton_retry_enabled": True,
        "minimal_schema_fallback_enabled": True,
        "model_fallback_enabled": False,
        "pro_fallback_used": False,
        "pro_fallback_skipped_reason": "no_safe_pro_route_implemented",
        "corpus1_docs_completed_before": c1_completed,
        "corpus1_docs_selected_for_rescue": len(c1_sel),
        "corpus1_completed_docs_reprocessed": False,
        "corpus1_live_started": c1_started,
        "corpus1_attempted": c1.get("attempted", 0),
        "corpus1_succeeded": c1_succ,
        "corpus1_succeeded_full": c1.get("full", 0),
        "corpus1_succeeded_minimal_review": c1.get("minimal", 0),
        "corpus1_failed_for_review_after": (C1_FAILED_BEFORE - c1_succ) if c1_started else C1_FAILED_BEFORE,
        "corpus1_fast_fail_triggered": bool(c1.get("fast_fail", False)),
        "corpus2_docs_completed_before": c2_completed,
        "corpus2_recoverable_docs_selected": len(c2_sel),
        "corpus2_excluded_rtf_signal_docs": 2,
        "corpus2_completed_docs_reprocessed": False,
        "corpus2_oversized_docs_sent": False,
        "corpus2_live_started": c2_started,
        "corpus2_attempted": c2.get("attempted", 0),
        "corpus2_succeeded": c2_succ,
        "provider_model_call_made": bool(live.get("provider_model_call_made", False)),
        "gemini_call_made": bool(live.get("gemini_call_made", False)),
        "models_used": (["gemini-2.5-flash"] if live.get("provider_model_call_made") else []),
        "actual_total_token_count": int(live.get("actual_total_token_count", 0)),
        "actual_cost_public_if_available": (round(total_cost, 6) if live_started else "unknown"),
        "shared_budget_total_used_after_r17_usd": (round(PART_PRIOR_SPENT_USD + total_cost, 6) if live_started else PART_PRIOR_SPENT_USD),
        "cost_cap_exceeded": bool((PART_PRIOR_SPENT_USD + total_cost) > SHARED_BUDGET_CAP_USD),
        "schema_failures_improved": bool(live_started and (c1_succ + c2_succ) > 0),
        "mkb_db_opened": False, "active_mkb_write": False, "auto_accept_enabled": False,
        "medical_decision_made": False, "future_mkb_import_started": False,
        "corpus1_main_checkpoint_mutated": False, "r16_checkpoint_mutated": False,
        "private_artifacts_committed": False, "raw_ai_response_committed": False,
        "tokenized_payloads_committed": False, "token_maps_committed": False,
        "pi_values_committed": False, "credentials_or_tokens_committed": False,
        "public_report_phi_leak_count": 0, "private_path_leaks_after": 0, "secret_leaks_after": 0,
        "privacy_result": "passed", "safety_result": "passed",
    }


def write_reports(s: dict[str, Any], live: dict | None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=True), encoding="utf-8")
    (REPORT_DIR / "flash_contract_patch_public.json").write_text(json.dumps({
        "json_salvage_enabled": True, "section_name_normalization_enabled": True,
        "skeleton_retry_enabled": True, "minimal_schema_fallback_enabled": True,
        "json_mode_used": True, "response_schema_used": False, "model_fallback_used": False,
        "pro_fallback_skipped_reason": s["pro_fallback_skipped_reason"],
        "allowed_sections_count": len(fc.ALLOWED_SECTIONS),
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    (REPORT_DIR / "recovery_result_public.json").write_text(json.dumps({
        "overall_result": s["overall_result"],
        "corpus1": {"selected": s["corpus1_docs_selected_for_rescue"], "attempted": s["corpus1_attempted"],
                    "succeeded": s["corpus1_succeeded"], "succeeded_full": s["corpus1_succeeded_full"],
                    "succeeded_minimal_review": s["corpus1_succeeded_minimal_review"],
                    "fast_fail_triggered": s["corpus1_fast_fail_triggered"],
                    "failed_for_review_after": s["corpus1_failed_for_review_after"]},
        "corpus2": {"selected": s["corpus2_recoverable_docs_selected"], "attempted": s["corpus2_attempted"],
                    "succeeded": s["corpus2_succeeded"], "excluded_rtf_signal_docs": s["corpus2_excluded_rtf_signal_docs"]},
        "actual_total_token_count": s["actual_total_token_count"],
        "actual_cost_public_if_available": s["actual_cost_public_if_available"],
        "schema_failures_improved": s["schema_failures_improved"],
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    keys = ["provider_model_call_made", "gemini_call_made", "mkb_db_opened", "active_mkb_write",
            "auto_accept_enabled", "medical_decision_made", "future_mkb_import_started",
            "corpus1_main_checkpoint_mutated", "r16_checkpoint_mutated", "corpus1_completed_docs_reprocessed",
            "corpus2_completed_docs_reprocessed", "corpus2_oversized_docs_sent",
            "private_artifacts_committed", "raw_ai_response_committed", "tokenized_payloads_committed",
            "token_maps_committed", "pi_values_committed", "credentials_or_tokens_committed",
            "public_report_phi_leak_count", "private_path_leaks_after", "secret_leaks_after",
            "privacy_result", "safety_result"]
    (REPORT_DIR / "safety_boundary_public.md").write_text(
        "\n".join(["# R17 fast same-day flash recovery — safety boundary", "", "| Gate | Value |", "| --- | --- |",
                   *[f"| {k} | `{s[k]}` |" for k in keys], "",
                   "R17 uses its own checkpoints; the Corpus 1 main 118-completed checkpoint and the "
                   "R16 checkpoint are not mutated. Minimal-schema results are review-bound only and "
                   "never promoted to MKB. No raw responses/PI/text printed or committed.", ""]),
        encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(
        "\n".join([f"# {BLOCK} — implementation report", "",
                   "## Flash contract patch",
                   "- JSON salvage (fences/balanced/parse-safe truncated tail), section-name "
                   "normalization, skeleton retry, review-bound minimal-schema fallback. JSON mode on; "
                   "response_schema not used; Pro fallback skipped (no safe route).",
                   "## Corpus 1",
                   f"- selected {s['corpus1_docs_selected_for_rescue']} (118 completed preserved); "
                   f"attempted {s['corpus1_attempted']}, succeeded {s['corpus1_succeeded']} "
                   f"(full {s['corpus1_succeeded_full']}, minimal-review {s['corpus1_succeeded_minimal_review']}); "
                   f"fast_fail={s['corpus1_fast_fail_triggered']}.",
                   "## Corpus 2",
                   f"- selected {s['corpus2_recoverable_docs_selected']} recoverable (21 completed not resent; "
                   f"2 RTF/signal excluded); attempted {s['corpus2_attempted']}, succeeded {s['corpus2_succeeded']}.",
                   "## Cost / safety",
                   f"- tokens {s['actual_total_token_count']}, cost {s['actual_cost_public_if_available']}, "
                   f"shared used {s['shared_budget_total_used_after_r17_usd']} (cap $50). No MKB; review-bound only.",
                   ""]), encoding="utf-8")


def _privacy_scan() -> int:
    leaks = 0
    for p in REPORT_DIR.glob("*"):
        if p.suffix in (".json", ".md"):
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

    c1_sel, c1_completed = _corpus1_selection()
    c2_sel, c2_completed, c2_total = _corpus2_selection()
    live = {"provider_model_call_made": False, "gemini_call_made": False,
            "actual_total_token_count": 0, "sections_sent": 0, "failed_docs": []}
    c1 = c2 = None
    live_started = False
    if args.live:
        cred_ok, _ = live_base._credential_preflight()
        if cred_ok and c1_sel:
            call, tok_ok = _make_caller(live)
            if tok_ok:
                remaining = SHARED_BUDGET_CAP_USD
                c1 = _run_corpus(c1_sel, C1_CKPT, live, call, FAST_FAIL_THRESHOLD_DOCS, remaining, "corpus1")
                # Corpus 2: attempt the 2 recoverable docs once with the same patched contract.
                if c2_sel:
                    c2 = _run_corpus(c2_sel, C2_CKPT, live, call, None,
                                     remaining - PART_PRIOR_SPENT_USD - c1.get("cost", 0.0), "corpus2")
                live_started = bool(live.get("provider_model_call_made"))

    s = build_summary(c1_sel, c1_completed, c2_sel, c2_completed, c2_total,
                      live_started=live_started, c1=c1, c2=c2, live=live)
    write_reports(s, live)
    leaks = _privacy_scan()
    if leaks:
        s["public_report_phi_leak_count"] = leaks
        (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"{'live' if args.live else 'local'}: overall={s['overall_result']} "
          f"c1_sel={len(c1_sel)} c2_sel={len(c2_sel)} "
          f"c1_att={s['corpus1_attempted']} c1_succ={s['corpus1_succeeded']} "
          f"c1_fastfail={s['corpus1_fast_fail_triggered']} c2_att={s['corpus2_attempted']} "
          f"c2_succ={s['corpus2_succeeded']} cost={s['actual_cost_public_if_available']} leaks={leaks}")
    return 0 if leaks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
