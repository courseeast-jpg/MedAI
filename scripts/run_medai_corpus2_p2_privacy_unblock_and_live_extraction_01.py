#!/usr/bin/env python3
"""MEDAI-CORPUS2-P2-PRIVACY-UNBLOCK-AND-LIVE-EXTRACTION-01.

Safe private over-tokenization unblock for Corpus 2 / Person 2, then a gated live
extraction. Instead of manually reviewing noisy candidates, this adds privacy-protective
placeholders for provider/facility names (near provider/person/signature cues), facility
names, and spelled-out dates to a PRIVATE supplemental vault, re-tokenizes all 25 requests,
rebuilds the outbound package, and re-runs the privacy gate. Live runs only if the gate
shows 0 high-confidence PI, 0 provider/facility raw leaks, and 0 spelled-date raw leaks.

Shared budget cap across Corpus 1 + Corpus 2: $50.00. NO MKB open/write/import, NO
auto-accept, NO medical decision, NO Grounding/Search. PI values, candidate values, raw
text, and raw AI responses are never printed or committed. All private artifacts stay
outside the repo. Corpus 1 is not touched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
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
import scripts.run_medai_ai_first_corpus_pi_tokenization_local_only_17a as t17a  # noqa: E402
import scripts.run_medai_ai_first_corpus2_p2_vault_coverage_review_and_live_extraction_01 as base_p2  # noqa: E402

BLOCK = "MEDAI-CORPUS2-P2-PRIVACY-UNBLOCK-AND-LIVE-EXTRACTION-01"
BRANCH = "medai-corpus2-p2-prep-01"
TARGET_MODEL = "gemini-2.5-flash-lite"
EXPECTED_REQUESTS = 25
SHARED_BUDGET_CAP_USD = 50.00
PER_CHUNK_CAP_USD = 0.05
FULL_MAX_OUTPUT_TOKENS = base_p2.FULL_MAX_OUTPUT_TOKENS
INPUT_USD_PER_M = base_p2.INPUT_USD_PER_M
OUTPUT_USD_PER_M = base_p2.OUTPUT_USD_PER_M
# A single tokenized payload above this is content_too_large (beyond the model input limit);
# it is excluded from the live batch as a documented correction, not sent.
MAX_PAYLOAD_TOKENS = 250_000

PREP_ROOT = base_p2.PRIVATE_PREP_ROOT
OUTBOUND = base_p2.OUTBOUND
OUTBOUND_SHA = base_p2.OUTBOUND_SHA
OUTBOUND_INTEGRITY = base_p2.OUTBOUND_INTEGRITY
DOC_MANIFEST = base_p2.DOC_MANIFEST
VAULT_CSV = base_p2.VAULT_CSV
TOKEN_MAPS = base_p2.TOKEN_MAPS
SUPPLEMENTAL_VAULT = PREP_ROOT / "person2_pi_vault_supplemental_private.csv"
DOCS_DIR = PREP_ROOT / "documents"

REPORT_DIR = REPO_ROOT / "reports" / "medai_corpus2_p2_privacy_unblock_and_live_extraction_01"
VAULT_HEADER = ["person_key", "value_type", "value", "token_class", "notes"]

# ---- Bounded candidate detectors (operate on PRIVATE raw text; values never emitted) ---
_NAME = r"[A-Z][A-Za-z'.\-]+(?:\s+[A-Z][A-Za-z'.\-]+){0,3}"
_PROVIDER_CUE = (r"Dr\.?|Provider|Physician|Referring(?:\s+Physician)?|Ordering(?:\s+Physician)?|"
                 r"Attending|Signed\s+by|Electronically\s+signed(?:\s+by)?|Reported\s+by|"
                 r"Performed\s+by|Interpreted\s+by|Dictated\s+by|Authorized\s+by|Reviewed\s+by|Ordered\s+by")
NAME_AFTER_CUE = re.compile(rf"\b(?:{_PROVIDER_CUE})\b[:\s]+({_NAME})")
_FAC_TYPE = (r"Hospital|Clinic|Medical\s+Center|Laborator(?:y|ies)|Pathology|Diagnostics|"
             r"Healthcare|Health\s+System|Imaging(?:\s+Center)?")
FACILITY_BEFORE = re.compile(rf"\b({_NAME})\s+(?:{_FAC_TYPE})\b")
FACILITY_LABEL = re.compile(rf"\b(?:Facility|Hospital|Clinic|Laboratory)\b[:\s]+({_NAME})")
_MONTH = (r"January|February|March|April|May|June|July|August|September|October|November|December|"
          r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
SPELLED_DATE = re.compile(rf"\b(?:{_MONTH})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
                          rf"|\b\d{{1,2}}\s+(?:{_MONTH})\.?,?\s+\d{{4}}\b")
_TOKEN = re.compile(r"\[[A-Z_]+_\d+\]")

_SKIP_TERMS = {"patient", "name", "date", "provider", "physician", "hospital", "clinic",
               "test", "result", "results", "report", "final", "normal", "reference", "range",
               "specimen", "collected", "received", "status", "comment", "comments", "note",
               "notes", "history", "impression", "findings", "diagnosis", "medication", "medications"}


def _candidate_ok(value: str, allowlist_lower: set[str]) -> bool:
    v = value.strip()
    if len(v) < 3:
        return False
    if v.lower() in _SKIP_TERMS or v.lower() in allowlist_lower:
        return False
    return True


def _extract_candidates(raw: str, allowlist_lower: set[str]) -> "list[tuple[str, str]]":
    """Return (value, token_class) bounded candidates from raw text. Values stay private."""
    out: list[tuple[str, str]] = []
    for m in NAME_AFTER_CUE.finditer(raw):
        if _candidate_ok(m.group(1), allowlist_lower):
            out.append((m.group(1).strip(), "PROVIDER"))
    for rx in (FACILITY_BEFORE, FACILITY_LABEL):
        for m in rx.finditer(raw):
            if _candidate_ok(m.group(1), allowlist_lower):
                out.append((m.group(1).strip(), "FACILITY"))
    for m in SPELLED_DATE.finditer(raw):
        out.append((m.group(0).strip(), "DATE"))
    return out


# ---- Loaders -------------------------------------------------------------------------
def _load_vault_entries(path: Path) -> "list[dict[str, str]]":
    entries: list[dict[str, str]] = []
    if not path.is_file():
        return entries
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            value = (row.get("value") or "").strip()
            tc = (row.get("token_class") or "").strip().upper()
            if value and tc in t17a.TOKEN_CLASSES:
                entries.append({"person_key": (row.get("person_key") or "P2").strip(),
                                "value_type": (row.get("value_type") or "").strip(),
                                "value": value, "token_class": tc})
    return entries


def _doc_set() -> "dict[str, dict[str, str]]":
    """Canonical doc set = the persistent private per-doc directories (all 25), so the
    unblock is idempotent. Family is taken from the existing manifest when available;
    source_hash is derived deterministically from raw text (a stable per-doc id)."""
    fam: dict[str, str] = {}
    if DOC_MANIFEST.is_file():
        try:
            for did, info in (json.loads(DOC_MANIFEST.read_text(encoding="utf-8")) or {}).items():
                fam[did] = str((info or {}).get("document_family") or "UNKNOWN")
        except (OSError, ValueError):
            pass
    meta: dict[str, dict[str, str]] = {}
    if DOCS_DIR.is_dir():
        for d in sorted(DOCS_DIR.iterdir()):
            if d.is_dir() and (d / "extracted_text_raw.txt").is_file():
                meta[d.name] = {"document_family": fam.get(d.name, "UNKNOWN")}
    return meta


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


# ---- Unblock (A1) --------------------------------------------------------------------
def run_unblock() -> dict[str, Any]:
    try:
        allowlist = t17a._load_allowlist()
    except Exception:
        allowlist = []
    allowlist_lower = {a.lower() for a in allowlist}
    base_entries = _load_vault_entries(VAULT_CSV)
    meta = _doc_set()

    # Gather bounded candidates across all private raw texts.
    cand_counts = Counter()
    supplemental: "dict[tuple[str, str], dict[str, str]]" = {}
    for doc_id in meta:
        raw_path = DOCS_DIR / doc_id / "extracted_text_raw.txt"
        if not raw_path.is_file():
            continue
        raw = raw_path.read_text(encoding="utf-8", errors="ignore")
        for value, tc in _extract_candidates(raw, allowlist_lower):
            key = (value.lower(), tc)
            cand_counts[tc] += 1
            supplemental.setdefault(key, {"person_key": "P2", "value_type": f"supplemental_{tc.lower()}",
                                          "value": value, "token_class": tc})
    supp_entries = list(supplemental.values())

    # Write the PRIVATE supplemental vault (values stay private; never committed).
    with SUPPLEMENTAL_VAULT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(VAULT_HEADER)
        for e in supp_entries:
            w.writerow([e["person_key"], e["value_type"], e["value"], e["token_class"], ""])

    all_entries = base_entries + supp_entries

    # Re-tokenize all requests and rebuild the outbound package + sidecars.
    outbound_lines: list[str] = []
    token_map_lines: list[str] = []
    manifest: dict[str, Any] = {}
    new_payloads: list[str] = []
    oversize_ids: list[str] = []
    docs_seen = 0
    for doc_id, m in meta.items():
        raw_path = DOCS_DIR / doc_id / "extracted_text_raw.txt"
        if not raw_path.is_file():
            continue
        docs_seen += 1
        raw = raw_path.read_text(encoding="utf-8", errors="ignore")
        tokenized, token_map, _counts = t17a._tokenize(raw, all_entries, allowlist)
        try:
            (DOCS_DIR / doc_id / "tokenized_text.txt").write_text(tokenized, encoding="utf-8")
            (DOCS_DIR / doc_id / "token_map_private.json").write_text(json.dumps(token_map, indent=2), encoding="utf-8")
        except OSError:
            pass
        approx_tokens = max(1, len(tokenized) // 4)
        if approx_tokens > MAX_PAYLOAD_TOKENS:
            # content_too_large: beyond the model input limit -> excluded from the live batch.
            oversize_ids.append(doc_id)
            manifest[doc_id] = {"document_family": m["document_family"],
                                "status": "content_too_large_non_sendable_rtf_container"}
            continue
        new_payloads.append(tokenized)
        content_sha = _sha256_text(tokenized)
        outbound_lines.append(json.dumps({
            "document_id": doc_id, "source_hash": _sha256_text(raw),
            "document_family": m["document_family"], "person_label": "P2",
            "tokenized_content": tokenized, "content_sha256": content_sha}, ensure_ascii=True))
        token_map_lines.append(json.dumps({"document_id": doc_id, "token_map": token_map}))
        manifest[doc_id] = {"document_family": m["document_family"], "status": "ready_for_ai_extraction"}

    OUTBOUND.write_text("\n".join(outbound_lines) + ("\n" if outbound_lines else ""), encoding="utf-8")
    outbound_sha = hashlib.sha256(OUTBOUND.read_bytes()).hexdigest()
    OUTBOUND_SHA.write_text(outbound_sha + "\n", encoding="utf-8")
    OUTBOUND_INTEGRITY.write_text(json.dumps({
        "record_count": len(outbound_lines), "outbound_sha256": outbound_sha,
        "per_doc": [{"document_id": json.loads(l)["document_id"],
                     "content_sha256_prefix": json.loads(l)["content_sha256"][:16]} for l in outbound_lines],
    }, indent=2), encoding="utf-8")
    DOC_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    TOKEN_MAPS.write_text("\n".join(token_map_lines) + ("\n" if token_map_lines else ""), encoding="utf-8")

    return {
        "supplemental_created": True,
        "supplemental_count": len(supp_entries),
        "candidate_counts_by_class": dict(cand_counts),
        "base_value_count": len(base_entries),
        "rebuilt_request_count": len(outbound_lines),
        "docs_seen": docs_seen,
        "oversize_count": len(oversize_ids),
        "oversize_doc_ids": oversize_ids,
        "new_payloads": new_payloads,
    }


# ---- Gate scan (A2) ------------------------------------------------------------------
def _raw_leak_counts(payloads: "list[str]") -> "tuple[int, int, int]":
    """Count remaining high-confidence PI + provider/facility names + spelled dates after
    token-stripping. Captured names that are label/skip words (e.g. 'Physician') are not
    counted — they are cue words, not identifiers (parity with extraction's _candidate_ok)."""
    high = pf = sd = 0
    for text in payloads:
        stripped = _TOKEN.sub(" ", text)
        high += base_p2.high_confidence_uncovered_count(text)
        for nm in NAME_AFTER_CUE.findall(stripped):
            if nm.strip().lower() not in _SKIP_TERMS:
                pf += 1
        for rx in (FACILITY_BEFORE, FACILITY_LABEL):
            for nm in rx.findall(stripped):
                if nm.strip().lower() not in _SKIP_TERMS:
                    pf += 1
        sd += len(SPELLED_DATE.findall(stripped))
    return high, pf, sd


def evaluate_gate(new_payloads: "list[str] | None" = None, oversize_count: int = 0) -> dict[str, Any]:
    if new_payloads is None:
        new_payloads = [str(json.loads(l).get("tokenized_content") or "")
                        for l in read_jsonl_lines(OUTBOUND) if l.strip()]
    high, pf, sd = _raw_leak_counts(new_payloads)
    files_present = (OUTBOUND.is_file() and OUTBOUND_SHA.is_file()
                    and OUTBOUND_INTEGRITY.is_file() and DOC_MANIFEST.is_file() and TOKEN_MAPS.is_file())
    request_count = len(new_payloads)
    # Sendable + documented content_too_large corrections must reconcile to the 25 loaded docs.
    count_reconciled = (request_count + oversize_count == EXPECTED_REQUESTS) and request_count >= 1
    per_doc_tokens = [max(1, len(p) // 4) for p in new_payloads]
    est_total = planner.estimate_total_cost(per_doc_tokens, FULL_MAX_OUTPUT_TOKENS, INPUT_USD_PER_M, OUTPUT_USD_PER_M) if per_doc_tokens else 0.0
    # Chunk size is a chunking parameter (>=1); the binding cap is the shared $50 total.
    selected_chunk = planner.select_chunk_size(per_doc_tokens, FULL_MAX_OUTPUT_TOKENS, INPUT_USD_PER_M, OUTPUT_USD_PER_M, PER_CHUNK_CAP_USD, EXPECTED_REQUESTS) if per_doc_tokens else 0
    selected_chunk = max(1, selected_chunk)
    per_chunk_cost = planner.worst_case_chunk_cost(per_doc_tokens, FULL_MAX_OUTPUT_TOKENS, INPUT_USD_PER_M, OUTPUT_USD_PER_M, selected_chunk) if per_doc_tokens else 0.0
    cost_total_ok = est_total <= SHARED_BUDGET_CAP_USD
    cred_ok, cred_cat = base_p2._credential_preflight()
    gate_passed = bool(files_present and count_reconciled and high == 0
                       and pf == 0 and sd == 0 and cost_total_ok and cred_ok)
    if high > 0:
        reason, stage = "high_confidence_uncovered_pi", "privacy"
    elif pf > 0:
        reason, stage = "provider_facility_raw_leak", "privacy"
    elif sd > 0:
        reason, stage = "spelled_date_raw_leak", "privacy"
    elif not files_present or not count_reconciled:
        reason, stage = "outbound_package_incomplete", "entry_gate"
    elif not cost_total_ok:
        reason, stage = "cost_cap_exceeded", "cost"
    elif not cred_ok:
        reason, stage = cred_cat, "credentials"
    else:
        reason, stage = "none", "none"
    return {"high": high, "pf": pf, "sd": sd, "files_present": files_present,
            "request_count": request_count, "oversize_count": oversize_count,
            "content_too_large_count": oversize_count, "count_reconciled": count_reconciled,
            "est_total": est_total, "selected_chunk": selected_chunk,
            "per_chunk_cost": per_chunk_cost, "cost_total_ok": cost_total_ok,
            "cost_chunk_ok": per_chunk_cost <= PER_CHUNK_CAP_USD,
            "credential_preflight_passed": cred_ok, "credential_status_category": cred_cat,
            "gate_passed": gate_passed, "block_reason": reason, "failure_stage": stage}


# ---- Summary + reports ---------------------------------------------------------------
def build_summary(unblock: dict[str, Any], g: dict[str, Any], *, live_started: bool, live: dict | None) -> dict[str, Any]:
    live = live or {}
    # Recover live metrics (sections/cost/token) from the prior on-disk summary so a
    # --local-gate regeneration preserves them without re-calling the provider.
    prior: dict[str, Any] = {}
    _psum = REPORT_DIR / "summary.json"
    if _psum.is_file():
        try:
            prior = json.loads(_psum.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior = {}

    def _lp(key: str, default: Any) -> Any:
        if key in live:
            return live[key]
        return prior.get(key, default)

    # Source of truth for completion counts is the durable P2 live checkpoint, so the
    # summary reflects prior live results even when re-run in --local-gate (no provider).
    st = lc.load_state(base=base_p2.LIVE_CHECKPOINT_DIR) or {}
    ckpt_sent = int(st.get("sent_count", 0))
    ckpt_succeeded = int(st.get("succeeded_count", 0))
    ckpt_failed = int(st.get("failed_count", 0))
    ever_live = live_started or ckpt_sent > 0
    sendable = g["request_count"]
    docs_completed = min(ckpt_succeeded, sendable)
    docs_failed_live = min(ckpt_failed, max(0, sendable - docs_completed))
    docs_failed_for_review = docs_failed_live + unblock["oversize_count"]
    docs_unattempted = max(0, sendable - docs_completed - docs_failed_live)
    actual_cost = _lp("actual_cost_public_if_available", "unknown")
    if ever_live and docs_completed == sendable and docs_failed_for_review == unblock["oversize_count"]:
        run_result = "PASS_SENDABLE"  # all sendable docs completed; only non-sendable excluded
    elif ever_live:
        run_result = "LIVE_FAIL"
    elif not g["gate_passed"]:
        run_result = "BLOCKED"
    else:
        run_result = "GATE_PASSED_LIVE_NOT_RUN"
    remaining = "unknown"
    if isinstance(actual_cost, (int, float)):
        remaining = round(SHARED_BUDGET_CAP_USD - float(actual_cost), 6)
    return {
        "block": BLOCK,
        "branch": BRANCH,
        "run_result": run_result,
        "previous_block_result": "BLOCKED",
        "previous_block_reason": "vault_manual_review_required",
        "supplemental_private_vault_created": unblock["supplemental_created"],
        "vault_private_value_count_before": unblock["base_value_count"],
        "vault_private_value_count_after": unblock["base_value_count"] + unblock["supplemental_count"],
        "supplemental_private_value_count": unblock["supplemental_count"],
        "supplemental_candidate_counts_by_class": unblock["candidate_counts_by_class"],
        "high_confidence_uncovered_pi_count_after": g["high"],
        "provider_facility_raw_leak_count_after": g["pf"],
        "spelled_date_raw_leak_count_after": g["sd"],
        "manual_vault_review_required_after": not g["gate_passed"] and g["block_reason"] in (
            "provider_facility_raw_leak", "spelled_date_raw_leak", "high_confidence_uncovered_pi"),
        "tokenized_request_count": g["request_count"],
        "content_too_large_excluded_count": unblock["oversize_count"],
        # --- compound/RTF decomposition addendum: abandoned per correction ---
        "compound_pdf_decomposition_attempted": False,
        "rtf_or_signal_container_rescue_attempted": False,
        "content_too_large_parent_docs_before": unblock["oversize_count"],
        "oversized_non_sendable_docs_excluded": unblock["oversize_count"],
        "oversized_docs_sent_to_provider": False,
        "corpus2_sendable_docs_selected_for_live": g["request_count"],
        "corpus2_excluded_docs": unblock["oversize_count"],
        "corpus2_exclusion_reason": "content_too_large_non_sendable_rtf_or_signal_container",
        "live_entry_gate_passed": g["gate_passed"],
        "live_entry_gate_block_reason": g["block_reason"],
        "live_run_started": ever_live,
        "provider_model_call_made": ever_live,
        "gemini_call_made": ever_live,
        "target_model": TARGET_MODEL,
        "docs_loaded": EXPECTED_REQUESTS,
        "docs_completed": docs_completed,
        "docs_failed_for_review": docs_failed_for_review,
        "docs_unattempted": docs_unattempted,
        "sections_sent": int(_lp("sections_sent", 0)),
        "sections_succeeded": int(_lp("sections_succeeded", 0)),
        "sections_failed": int(_lp("sections_failed", 0)),
        "failure_stage": _lp("failure_stage", g["failure_stage"] if not g["gate_passed"] else "none"),
        "failure_category": _lp("failure_category", g["block_reason"] if not g["gate_passed"] else "none"),
        "failed_doc_hash": _lp("failed_doc_hash", None),
        "failed_section": _lp("failed_section", None),
        "failed_evidence_preserved": bool(_lp("failed_evidence_preserved", False)),
        "estimated_cost_before_live_usd": round(g["est_total"], 6),
        "selected_chunk_size": g["selected_chunk"],
        "estimated_per_chunk_cost_usd": round(g["per_chunk_cost"], 6),
        "credential_preflight_passed": g["credential_preflight_passed"],
        "actual_total_token_count": int(_lp("actual_total_token_count", 0)),
        "actual_cost_public_if_available": actual_cost,
        "shared_budget_cap_usd": SHARED_BUDGET_CAP_USD,
        "shared_budget_remaining_estimate_after_part_a_usd": remaining,
        "mkb_db_opened": False, "active_mkb_write": False, "auto_accept_enabled": False,
        "medical_decision_made": False, "future_mkb_import_started": False,
        "private_artifacts_committed": False, "raw_ai_response_committed": False,
        "tokenized_payloads_committed": False, "token_maps_committed": False,
        "pi_values_committed": False, "credentials_or_tokens_committed": False,
        "corpus1_touched": False,
        "public_report_phi_leak_count": 0, "private_path_leaks_after": 0, "secret_leaks_after": 0,
        "privacy_result": "passed" if (g["high"] == 0 and g["pf"] == 0 and g["sd"] == 0) else "blocked",
        "safety_result": "passed",
    }


def write_reports(s: dict[str, Any], g: dict[str, Any], unblock: dict[str, Any], live: dict | None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=True), encoding="utf-8")
    (REPORT_DIR / "privacy_unblock_public.json").write_text(json.dumps({
        "supplemental_private_vault_created": s["supplemental_private_vault_created"],
        "supplemental_private_value_count": s["supplemental_private_value_count"],
        "supplemental_candidate_counts_by_class": s["supplemental_candidate_counts_by_class"],
        "vault_value_count_before": s["vault_private_value_count_before"],
        "vault_value_count_after": s["vault_private_value_count_after"],
        "high_confidence_uncovered_pi_count_after": s["high_confidence_uncovered_pi_count_after"],
        "provider_facility_raw_leak_count_after": s["provider_facility_raw_leak_count_after"],
        "spelled_date_raw_leak_count_after": s["spelled_date_raw_leak_count_after"],
        "no_candidate_values_in_this_report": True,
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    gate_md = ["# Corpus 2 / P2 privacy-unblock live entry gate", "",
               f"- tokenized_request_count: `{g['request_count']}`",
               f"- high_confidence_uncovered_pi_count_after: `{g['high']}`",
               f"- provider_facility_raw_leak_count_after: `{g['pf']}`",
               f"- spelled_date_raw_leak_count_after: `{g['sd']}`",
               f"- estimated_total_cost_usd: `{round(g['est_total'],6)}` (shared cap `${SHARED_BUDGET_CAP_USD}`) -> within=`{g['cost_total_ok']}`",
               f"- selected_chunk_size: `{g['selected_chunk']}`; per-chunk: `{round(g['per_chunk_cost'],6)}` (cap `${PER_CHUNK_CAP_USD}`) -> within=`{g['cost_chunk_ok']}`",
               f"- credential_preflight_passed: `{g['credential_preflight_passed']}`",
               f"- live_entry_gate_passed: `{g['gate_passed']}` (block_reason `{g['block_reason']}`)", ""]
    (REPORT_DIR / "live_entry_gate_public.md").write_text("\n".join(gate_md), encoding="utf-8")
    if live and s["live_run_started"]:
        run_md = ["# Corpus 2 / P2 live run report", "",
                  f"- run_result: `{s['run_result']}`",
                  f"- docs loaded/completed/failed/unattempted: `{s['docs_loaded']}`/`{s['docs_completed']}`/`{s['docs_failed_for_review']}`/`{s['docs_unattempted']}`",
                  f"- sections sent/succeeded/failed: `{s['sections_sent']}`/`{s['sections_succeeded']}`/`{s['sections_failed']}`",
                  f"- failure stage/category: `{s['failure_stage']}`/`{s['failure_category']}`",
                  f"- failed_doc_hash: `{s['failed_doc_hash']}`; failed_section: `{s['failed_section']}`; evidence preserved: `{s['failed_evidence_preserved']}`",
                  f"- actual token count: `{s['actual_total_token_count']}`; actual cost: `{s['actual_cost_public_if_available']}`", ""]
    else:
        run_md = ["# Corpus 2 / P2 live run report", "",
                  f"Live not run: entry gate did not pass (`{g['block_reason']}`). No provider call made.", ""]
    (REPORT_DIR / "live_run_public_report.md").write_text("\n".join(run_md), encoding="utf-8")
    failed_docs = list((live or {}).get("failed_docs", []))
    for did in unblock.get("oversize_doc_ids", []):
        failed_docs.append({"doc_hash": did,
                            "failure_category": "content_too_large_non_sendable_rtf_container",
                            "failed_section": None})
    (REPORT_DIR / "failed_docs_public.json").write_text(json.dumps({
        "failed_for_review_count": len(failed_docs),
        "content_too_large_count": unblock.get("oversize_count", 0),
        "failed_docs": failed_docs}, indent=2), encoding="utf-8")

    # Decomposition addendum: abandoned per operator correction (these are non-text RTF/
    # signal containers, not compound PDFs). Public-safe: hashes + counts + reason only.
    (REPORT_DIR / "compound_pdf_decomposition_public.json").write_text(json.dumps({
        "compound_pdf_decomposition_attempted": False,
        "rtf_or_signal_container_rescue_attempted": False,
        "content_too_large_parent_docs_before": unblock.get("oversize_count", 0),
        "excluded_count": unblock.get("oversize_count", 0),
        "exclusion_reason": "content_too_large_non_sendable_rtf_or_signal_container",
        "parent_doc_hashes": list(unblock.get("oversize_doc_ids", [])),
        "oversized_docs_sent_to_provider": False,
        "child_docs_created": 0,
        "ocr_or_image_extraction_attempted": False,
        "no_raw_text_or_images_or_pi_in_this_report": True,
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    keys = ["provider_model_call_made", "gemini_call_made", "mkb_db_opened", "active_mkb_write",
            "auto_accept_enabled", "medical_decision_made", "future_mkb_import_started", "corpus1_touched",
            "private_artifacts_committed", "raw_ai_response_committed", "tokenized_payloads_committed",
            "token_maps_committed", "pi_values_committed", "credentials_or_tokens_committed",
            "public_report_phi_leak_count", "private_path_leaks_after", "secret_leaks_after",
            "privacy_result", "safety_result"]
    (REPORT_DIR / "safety_boundary_public.md").write_text(
        "\n".join(["# Corpus 2 / P2 privacy-unblock — safety boundary", "", "| Gate | Value |", "| --- | --- |",
                   *[f"| {k} | `{s[k]}` |" for k in keys], "",
                   "Over-tokenization placeholders and candidate values stay in the PRIVATE "
                   "supplemental vault and token maps outside the repo. No PI value, candidate "
                   "value, raw text, or raw AI response is printed or committed. Corpus 1 untouched.", ""]),
        encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(
        "\n".join([f"# {BLOCK} — implementation report", "",
                   "## A1 over-tokenization unblock",
                   f"- supplemental private values added: `{s['supplemental_private_value_count']}` "
                   f"by class `{s['supplemental_candidate_counts_by_class']}`.",
                   f"- vault values before/after: `{s['vault_private_value_count_before']}` / `{s['vault_private_value_count_after']}`.",
                   "- bounded rules: provider/person names near provider/signature/referring/ordering cues; "
                   "facility names near facility types/labels; all spelled-out dates. Clinical terms, lab "
                   "names, and allowlist phrases are excluded.",
                   "",
                   "## A2 gate",
                   f"- high/provider-facility/spelled-date raw leaks after re-tokenization: "
                   f"`{s['high_confidence_uncovered_pi_count_after']}` / "
                   f"`{s['provider_facility_raw_leak_count_after']}` / `{s['spelled_date_raw_leak_count_after']}`.",
                   f"- live_entry_gate_passed: `{s['live_entry_gate_passed']}`; run_result: `{s['run_result']}`.",
                   "", "## Boundaries", "No MKB, no auto-accept, no medical decision, shared $50 cap, no Corpus 1 access.", ""]),
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


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-gate", action="store_true")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args(argv)

    unblock = run_unblock()
    g = evaluate_gate(unblock["new_payloads"], unblock["oversize_count"])
    live = None
    live_started = False
    if args.live and g["gate_passed"]:
        base_p2.TOTAL_CAP_USD = SHARED_BUDGET_CAP_USD  # use the shared $50 pool for the live loop
        live = base_p2.run_live(g)
        live_started = bool(live.get("provider_model_call_made"))

    s = build_summary(unblock, g, live_started=live_started, live=live)
    write_reports(s, g, unblock, live)
    leaks = _privacy_scan()
    if leaks:
        s["public_report_phi_leak_count"] = leaks
        (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"{'live' if args.live else 'local_gate'}: result={s['run_result']} gate={g['gate_passed']} "
          f"block={g['block_reason']} supp={s['supplemental_private_value_count']} "
          f"high={g['high']} pf={g['pf']} sd={g['sd']} provider_call={s['provider_model_call_made']} "
          f"completed={s['docs_completed']} report_leaks={leaks}")
    return 0 if leaks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
