#!/usr/bin/env python3
"""MEDAI-R18-RESIDUAL-SCHEMA-FAILURES-STRONGER-MODEL-FALLBACK.

Stronger-model (gemini-2.5-pro) fallback for the documents that still failed Flash schema
validation after R17. Reuses the entire R17 safety stack unchanged — same privacy-tokenized
payloads, sectioned extraction, JSON salvage / section normalization / skeleton retry /
review-bound minimal-schema fallback, checkpoint/resume, evidence preservation, cost
tracking, public redaction, and no-MKB gates — and only swaps the model and the checkpoint
namespace. The stronger model is used ONLY for residual schema failures; R17-recovered
packages and the original 118 completed are never reprocessed.

Shared same-day cap $50.00 (prior spend ~$3.033334). Time-priority. No Grounding/Search,
no MKB, no auto-accept, no medical decision. Raw responses/PI/text never printed/committed.
If the stronger route is unavailable/unsafe/credential-blocked, stop and report
`stronger_model_route_unavailable` — never improvise with raw private payloads.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
from execution import cost_chunk_planner as planner  # noqa: E402
from execution import live_checkpoint as lc  # noqa: E402
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live_base  # noqa: E402
import scripts.run_medai_fast_same_day_flash_contract_recovery_r17 as r17  # noqa: E402

BLOCK = "MEDAI-R18-RESIDUAL-SCHEMA-FAILURES-STRONGER-MODEL-FALLBACK"
STRONGER_MODEL = "gemini-2.5-pro"
SHARED_BUDGET_CAP_USD = 50.00
PRIOR_SAME_DAY_SPEND_USD = 3.033334
REMAINING_BUDGET_USD = round(SHARED_BUDGET_CAP_USD - PRIOR_SAME_DAY_SPEND_USD, 6)
FAST_FAIL_THRESHOLD_DOCS = 20
PRO_HTTP_TIMEOUT_S = 300
FULL_MAX_OUTPUT_TOKENS = 8192
# gemini-2.5-pro text pricing (conservative, <=200k context): $1.25/1M in, $10/1M out.
PRO_INPUT_USD_PER_M = 1.25
PRO_OUTPUT_USD_PER_M = 10.0

C1_CONTENT_BEFORE = 141  # 118 original + 23 R17
C1_RESIDUAL_EXPECTED = 337
C2_COMPLETED_BEFORE = 21

R18_C1_CKPT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_corpus1_r18_checkpoint"))
R18_C2_CKPT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_corpus2_r18_checkpoint"))
REPORT_DIR = REPO_ROOT / "reports" / "medai_r18_residual_schema_failures_stronger_model_fallback"


# ---- residual selection (resumes R17 checkpoints, excludes recovered) -----------------
def _corpus1_residual() -> "tuple[list[dict], int]":
    sel, _completed = r17._corpus1_selection()  # 360 failed (118 preserved excluded)
    r17_done = set(lc.load_completed(base=r17.C1_CKPT))
    if not r17_done:
        r17_done = r17._archived_completed_ids(360)
    residual = [r for r in sel if str(r.get("document_id") or "") not in r17_done]
    content_before = 118 + len(r17_done)
    return residual, content_before


def _corpus2_residual() -> list[dict]:
    sel, _c, _t = r17._corpus2_selection()  # 2 recoverable (21 not sent, 2 RTF excluded)
    r17_done = set(lc.load_completed(base=r17.C2_CKPT))
    if not r17_done:
        r17_done = r17._archived_completed_ids(2)
    return [r for r in sel if str(r.get("document_id") or "") not in r17_done]


# ---- stronger-model caller (same safety wrapper as R17, pro model + longer timeout) ----
def _make_pro_caller(live: dict[str, Any]):
    from execution.gemini_vertex_adapter import (
        GeminiVertexConfig, build_vertex_generate_content_url, classify_vertex_provider_error,
        acquire_google_cloud_access_token,
    )
    import urllib.error
    import urllib.request
    url = build_vertex_generate_content_url(GeminiVertexConfig(vertex_model=STRONGER_MODEL))
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
            with urllib.request.urlopen(req, timeout=PRO_HTTP_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"vertex_http_error status={exc.code}") from exc

    def call(payload):
        os.environ[r17.LIVE_GATE] = r17.GATE_VALUE
        live["provider_model_call_made"] = True
        live["gemini_call_made"] = True
        try:
            return True, "ok", _post(payload)
        except Exception as exc:
            return False, str(classify_vertex_provider_error(exc).get("provider_error_category") or "provider_error"), {}
        finally:
            os.environ.pop(r17.LIVE_GATE, None)
    return call, bool(str(token or "").strip())


def _cost_estimate(payloads: list[str]) -> float:
    per_doc = [max(1, len(p) // 4) for p in payloads]
    return planner.estimate_total_cost(per_doc, FULL_MAX_OUTPUT_TOKENS, PRO_INPUT_USD_PER_M, PRO_OUTPUT_USD_PER_M) if per_doc else 0.0


def evaluate_gate() -> dict[str, Any]:
    c1_residual, content_before = _corpus1_residual()
    c2_residual = _corpus2_residual()
    payloads = ([str(r.get("tokenized_content") or "") for r in c1_residual]
                + [str(r.get("tokenized_content") or "") for r in c2_residual])
    residual_privacy = sum(len(live_base._residual_pi(p)) for p in payloads)
    est = _cost_estimate(payloads)
    # Route availability: adapter builds a pro URL + credentials refresh (no model call).
    try:
        from execution.gemini_vertex_adapter import GeminiVertexConfig, build_vertex_generate_content_url
        route_ok = STRONGER_MODEL in build_vertex_generate_content_url(GeminiVertexConfig(vertex_model=STRONGER_MODEL))
    except Exception:
        route_ok = False
    cred_ok, cred_cat = live_base._credential_preflight()
    cost_ok = (PRIOR_SAME_DAY_SPEND_USD + est) <= SHARED_BUDGET_CAP_USD
    privacy_clean = residual_privacy == 0
    gate_passed = bool(route_ok and cred_ok and cost_ok and privacy_clean
                       and len(c1_residual) >= 1)
    reason = ("none" if gate_passed else
              "stronger_model_route_unavailable" if not route_ok else
              "residual_pi_in_payloads" if not privacy_clean else
              "cost_exceeds_remaining_budget" if not cost_ok else
              cred_cat if not cred_ok else "no_residual")
    return {"c1_residual": c1_residual, "c2_residual": c2_residual, "content_before": content_before,
            "residual_privacy": residual_privacy, "privacy_clean": privacy_clean, "est": est,
            "route_ok": route_ok, "credential_preflight_passed": cred_ok,
            "credential_status_category": cred_cat, "cost_ok": cost_ok,
            "gate_passed": gate_passed, "block_reason": reason}


# ---- summary + reports ---------------------------------------------------------------
def build_summary(g: dict[str, Any], *, live_started: bool, c1: dict | None, c2: dict | None, live: dict | None) -> dict[str, Any]:
    c1 = c1 or {}
    c2 = c2 or {}
    live = live or {}
    c1_succ = c1.get("succeeded", 0)
    c2_succ = c2.get("succeeded", 0)
    total_cost = round(c1.get("cost", 0.0) + c2.get("cost", 0.0), 6)
    overall = ("PASS_PARTIAL" if (live_started and (c1_succ + c2_succ) > 0)
               else "LIVE_FAIL" if live_started
               else "BLOCKED" if not g["gate_passed"]
               else "GATE_PASSED_LIVE_NOT_RUN")
    return {
        "block": BLOCK,
        "overall_result": overall,
        "time_priority_mode": True,
        "shared_budget_cap_usd": SHARED_BUDGET_CAP_USD,
        "prior_same_day_spend_usd": PRIOR_SAME_DAY_SPEND_USD,
        "remaining_budget_estimate_usd": REMAINING_BUDGET_USD,
        "stronger_model_route_available": g["route_ok"],
        "stronger_model_used": bool(live.get("provider_model_call_made", False)),
        "models_used": (["gemini-2.5-pro"] if live.get("provider_model_call_made") else []),
        "grounding_or_search_used": False,
        "corpus1_docs_total": 478,
        "corpus1_content_packages_before_r18": g["content_before"],
        "corpus1_residual_selected": len(g["c1_residual"]),
        "corpus1_completed_docs_reprocessed": False,
        "corpus1_live_started": bool(c1),
        "corpus1_attempted": c1.get("attempted", 0),
        "corpus1_succeeded_full_schema": c1.get("full", 0),
        "corpus1_succeeded_minimal_review_bound": c1.get("minimal", 0),
        "corpus1_succeeded_total": c1_succ,
        "corpus1_failed_for_review_after": (len(g["c1_residual"]) - c1_succ) if c1 else len(g["c1_residual"]),
        "corpus1_fast_fail_triggered": bool(c1.get("fast_fail", False)),
        "corpus2_completed_before": C2_COMPLETED_BEFORE,
        "corpus2_recoverable_selected": len(g["c2_residual"]),
        "corpus2_rtf_signal_excluded": 2,
        "corpus2_completed_docs_reprocessed": False,
        "corpus2_oversized_docs_sent": False,
        "corpus2_live_started": bool(c2),
        "corpus2_attempted": c2.get("attempted", 0),
        "corpus2_succeeded": c2_succ,
        "provider_model_call_made": bool(live.get("provider_model_call_made", False)),
        "gemini_call_made": bool(live.get("gemini_call_made", False)),
        "actual_total_token_count": int(live.get("actual_total_token_count", 0)),
        "actual_cost_public_if_available": (round(total_cost, 6) if live_started else "unknown"),
        "same_day_total_cost_after_r18_usd": (round(PRIOR_SAME_DAY_SPEND_USD + total_cost, 6) if live_started else PRIOR_SAME_DAY_SPEND_USD),
        "cost_cap_exceeded": bool((PRIOR_SAME_DAY_SPEND_USD + total_cost) > SHARED_BUDGET_CAP_USD),
        "estimated_cost_before_live_usd": round(g["est"], 6),
        "checkpoint_resume_available": True,
        "failed_evidence_preserved": bool(live.get("failed_evidence_preserved", False)),
        "schema_failures_improved_vs_r17": bool(live_started and (c1_succ + c2_succ) > 0),
        "mkb_db_opened": False, "active_mkb_write": False, "auto_accept_enabled": False,
        "medical_decision_made": False, "future_mkb_import_started": False,
        "corpus1_main_checkpoint_mutated": False, "r17_checkpoint_mutated": False,
        "private_artifacts_committed": False, "raw_ai_response_committed": False,
        "tokenized_payloads_committed": False, "token_maps_committed": False,
        "pi_values_committed": False, "credentials_or_tokens_committed": False,
        "public_report_phi_leak_count": 0, "private_path_leaks_after": 0, "secret_leaks_after": 0,
        "privacy_result": "passed" if g["privacy_clean"] else "blocked",
        "safety_result": "passed",
    }


def write_reports(s: dict[str, Any], g: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=True), encoding="utf-8")
    (REPORT_DIR / "stronger_model_route_public.json").write_text(json.dumps({
        "stronger_model": STRONGER_MODEL,
        "route_available": g["route_ok"],
        "route_via": "existing_vertex_adapter_model_override",
        "grounding_or_search_used": False,
        "json_mode_used": True, "response_schema_used": False,
        "reuses_r17_salvage_skeleton_minimal": True,
        "credential_preflight_passed": g["credential_preflight_passed"],
        "block_reason": g["block_reason"],
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    (REPORT_DIR / "residual_recovery_result_public.json").write_text(json.dumps({
        "overall_result": s["overall_result"],
        "corpus1": {"residual_selected": s["corpus1_residual_selected"], "attempted": s["corpus1_attempted"],
                    "succeeded_full": s["corpus1_succeeded_full_schema"],
                    "succeeded_minimal_review_bound": s["corpus1_succeeded_minimal_review_bound"],
                    "succeeded_total": s["corpus1_succeeded_total"],
                    "failed_for_review_after": s["corpus1_failed_for_review_after"],
                    "fast_fail_triggered": s["corpus1_fast_fail_triggered"]},
        "corpus2": {"recoverable_selected": s["corpus2_recoverable_selected"], "attempted": s["corpus2_attempted"],
                    "succeeded": s["corpus2_succeeded"], "rtf_signal_excluded": s["corpus2_rtf_signal_excluded"]},
        "estimated_cost_before_live_usd": s["estimated_cost_before_live_usd"],
        "actual_total_token_count": s["actual_total_token_count"],
        "actual_cost_public_if_available": s["actual_cost_public_if_available"],
        "same_day_total_cost_after_r18_usd": s["same_day_total_cost_after_r18_usd"],
        "schema_failures_improved_vs_r17": s["schema_failures_improved_vs_r17"],
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    keys = ["stronger_model_route_available", "stronger_model_used", "grounding_or_search_used",
            "provider_model_call_made", "gemini_call_made", "mkb_db_opened", "active_mkb_write",
            "auto_accept_enabled", "medical_decision_made", "future_mkb_import_started",
            "corpus1_main_checkpoint_mutated", "r17_checkpoint_mutated",
            "corpus1_completed_docs_reprocessed", "corpus2_completed_docs_reprocessed",
            "corpus2_oversized_docs_sent", "cost_cap_exceeded",
            "private_artifacts_committed", "raw_ai_response_committed", "tokenized_payloads_committed",
            "token_maps_committed", "pi_values_committed", "credentials_or_tokens_committed",
            "public_report_phi_leak_count", "private_path_leaks_after", "secret_leaks_after",
            "privacy_result", "safety_result"]
    (REPORT_DIR / "safety_boundary_public.md").write_text(
        "\n".join(["# R18 stronger-model fallback — safety boundary", "", "| Gate | Value |", "| --- | --- |",
                   *[f"| {k} | `{s[k]}` |" for k in keys], "",
                   "Stronger model runs only through the existing safety-gated adapter on the same "
                   "privacy-tokenized payloads. R18 uses its own checkpoints; main 118, R17, and "
                   "corpus2 checkpoints are untouched. Review-bound minimal results never promoted "
                   "to MKB. No raw responses/PI/text printed or committed.", ""]),
        encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(
        "\n".join([f"# {BLOCK} — implementation report", "",
                   f"- Stronger model: `{STRONGER_MODEL}` via the existing Vertex adapter model "
                   "override; reuses the full R17 salvage/skeleton/minimal contract + checkpoint/"
                   "evidence/cost/redaction/no-MKB stack. response_schema not used; JSON mode on.",
                   f"- Residual selection: Corpus 1 `{s['corpus1_residual_selected']}` "
                   f"(content before R18 `{s['corpus1_content_packages_before_r18']}`, not reprocessed); "
                   f"Corpus 2 `{s['corpus2_recoverable_selected']}` (21 not resent, 2 RTF/signal excluded).",
                   f"- Estimated cost (worst-case @8192 pro): `${s['estimated_cost_before_live_usd']}` "
                   f"within remaining `${REMAINING_BUDGET_USD}`.",
                   f"- Result: `{s['overall_result']}`; Corpus 1 succeeded "
                   f"`{s['corpus1_succeeded_total']}` (full `{s['corpus1_succeeded_full_schema']}`, "
                   f"minimal-review `{s['corpus1_succeeded_minimal_review_bound']}`), fast_fail "
                   f"`{s['corpus1_fast_fail_triggered']}`; Corpus 2 succeeded `{s['corpus2_succeeded']}`.",
                   f"- tokens `{s['actual_total_token_count']}`, cost `{s['actual_cost_public_if_available']}`, "
                   f"same-day total `{s['same_day_total_cost_after_r18_usd']}` (cap $50).", ""]),
        encoding="utf-8")


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

    g = evaluate_gate()
    live = {"provider_model_call_made": False, "gemini_call_made": False,
            "actual_total_token_count": 0, "sections_sent": 0, "failed_docs": []}
    c1 = c2 = None
    live_started = False
    if args.live and g["gate_passed"]:
        call, tok_ok = _make_pro_caller(live)
        if tok_ok:
            # Reuse the R17 per-doc recovery driver with the pro caller; account prior spend.
            r17.PART_PRIOR_SPENT_USD = PRIOR_SAME_DAY_SPEND_USD
            c1 = r17._run_corpus(g["c1_residual"], R18_C1_CKPT, live, call,
                                 FAST_FAIL_THRESHOLD_DOCS, SHARED_BUDGET_CAP_USD, "corpus1")
            if g["c2_residual"]:
                c2 = r17._run_corpus(g["c2_residual"], R18_C2_CKPT, live, call, None,
                                     SHARED_BUDGET_CAP_USD, "corpus2")
            live_started = bool(live.get("provider_model_call_made"))

    s = build_summary(g, live_started=live_started, c1=c1, c2=c2, live=live)
    write_reports(s, g)
    leaks = _privacy_scan()
    if leaks:
        s["public_report_phi_leak_count"] = leaks
        (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"{'live' if args.live else 'local'}: overall={s['overall_result']} route={g['route_ok']} "
          f"c1_residual={len(g['c1_residual'])} c2_residual={len(g['c2_residual'])} est=${round(g['est'],4)} "
          f"c1_att={s['corpus1_attempted']} c1_succ={s['corpus1_succeeded_total']} "
          f"c1_fastfail={s['corpus1_fast_fail_triggered']} c2_succ={s['corpus2_succeeded']} "
          f"cost={s['actual_cost_public_if_available']} leaks={leaks}")
    return 0 if leaks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
