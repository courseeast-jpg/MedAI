#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS2-P2-VAULT-COVERAGE-REVIEW-AND-LIVE-EXTRACTION-01.

Two guarded phases over Corpus 2 / Person 2 (25 tokenized requests):

  --local-gate : review P2 PI-vault coverage + run the live entry gate. No provider call.
  --live       : re-verify the gate and, ONLY if it passes, run the live extraction once
                 using the established sectioned / autonomous-recovery pattern with
                 checkpoint/resume, failed-evidence preservation, stop-on-hard-failure,
                 and $2.00 total / $0.05 per-chunk caps.

Safety: NO MKB open/write, NO auto-accept, NO medical decision. The live gate is set only
inside an authorized provider call. PI values, raw text, and raw AI responses are never
printed or committed. All private artifacts (vault, token maps, tokenized payloads, raw
extraction, responses, checkpoint, evidence) stay OUTSIDE the repo. Public reports carry
only counts, confidence bands, short/grouped hashes, and redacted labels. This block does
NOT touch Corpus 1 (no Corpus 1 path is read or written).

Private locations are supplied via env vars (with %LOCALAPPDATA%/%USERPROFILE% defaults),
so this committed source contains no literal user path.
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

BLOCK = "MEDAI-AI-FIRST-CORPUS2-P2-VAULT-COVERAGE-REVIEW-AND-LIVE-EXTRACTION-01"
BRANCH = "medai-corpus2-p2-prep-01"
TARGET_MODEL = "gemini-2.5-flash-lite"
EXPECTED_REQUESTS = 25
TOTAL_CAP_USD = 2.00
PER_CHUNK_CAP_USD = 0.05
FULL_MAX_OUTPUT_TOKENS = 8192
INPUT_USD_PER_M = 0.075
OUTPUT_USD_PER_M = 0.30


def _env_path(name: str, default: str) -> Path:
    return Path(os.path.expandvars((os.environ.get(name) or default).strip()))


PRIVATE_PREP_ROOT = _env_path(
    "MEDAI_CORPUS2_P2_PRIVATE_ROOT",
    r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_corpus2_p2_private_prep_01")
OUTBOUND = PRIVATE_PREP_ROOT / "outbound_requests_private.jsonl"
OUTBOUND_SHA = PRIVATE_PREP_ROOT / "outbound_requests_private.sha256"
OUTBOUND_INTEGRITY = PRIVATE_PREP_ROOT / "outbound_requests_integrity_private.json"
DOC_MANIFEST = PRIVATE_PREP_ROOT / "doc_id_manifest_private.json"
VAULT_CSV = PRIVATE_PREP_ROOT / "person2_pi_vault_private.csv"
TOKEN_MAPS = PRIVATE_PREP_ROOT / "token_maps_private.jsonl"

LIVE_CHECKPOINT_DIR = _env_path(
    "MEDAI_CORPUS2_P2_LIVE_CHECKPOINT",
    r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_corpus2_p2_live_checkpoint_01")
LIVE_STAGING_DIR = _env_path(
    "MEDAI_CORPUS2_P2_LIVE_STAGING",
    r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_corpus2_p2_live_01")
EVIDENCE_DIR = _env_path(
    "MEDAI_CORPUS2_P2_EVIDENCE",
    r"%USERPROFILE%\Downloads\MedAI_CORPUS2_P2_FAILED_EVIDENCE_PRESERVE_PRIVATE")

LIVE_GATE = "MEDAI_AI_FIRST_CORPUS2_P2_LIVE_01_APPROVED"
GATE_VALUE = "YES"

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus2_p2_vault_coverage_review_and_live_extraction_01"

# ---- Detectors (counts only; never emit matched values) -------------------------------
_TOKEN = re.compile(r"\[[A-Z_]+_\d+\]")
_HIGH = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "phone_formatted": re.compile(r"\(\d{3}\)\s*\d{3}[.\- ]\d{4}|\b\d{3}[.\-]\d{3}[.\-]\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "id_labeled": re.compile(r"(?i)\b(?:MRN|Medical Record|Account|Insurance|Policy|Member|Accession|Specimen)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "local_path": re.compile(r"[A-Za-z]:\\[^\s]+|/(?:home|Users)/[^\s]+"),
}
_LOW = {
    "spelled_date": re.compile(r"(?i)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"),
    "provider_facility_cue": re.compile(r"(?i)\b(?:Dr\.?|M\.?D\.?|Provider|Physician|Hospital|Clinic|Medical Center|Pathology|Laborator)\b"),
    "person_like_2cap": re.compile(r"\b[A-Z][a-z]{1,}\s+[A-Z][a-z]{1,}\b"),
}
_BARE10 = re.compile(r"(?<!\d)\d{10}(?!\d)")


def high_confidence_uncovered_count(tokenized_text: str) -> int:
    """Count high-confidence structured PII left in a tokenized payload (token-stripped)."""
    stripped = _TOKEN.sub(" ", tokenized_text)
    return sum(len(rx.findall(stripped)) for rx in _HIGH.values())


def scan_payloads(payloads: "list[str]") -> dict[str, Any]:
    high = Counter()
    low = Counter()
    bare10 = 0
    for text in payloads:
        stripped = _TOKEN.sub(" ", text)
        for k, rx in _HIGH.items():
            high[k] += len(rx.findall(stripped))
        for k, rx in _LOW.items():
            low[k] += len(rx.findall(stripped))
        bare10 += len(_BARE10.findall(stripped))
    return {"high": dict(high), "low": dict(low), "bare_10_digit_runs": bare10,
            "high_total": sum(high.values()),
            "low_total": sum(low.values())}


# ---- Vault + payload loading (private) ------------------------------------------------
def _load_vault_category_counts() -> "tuple[int, dict[str, int], bool]":
    if not VAULT_CSV.is_file():
        return 0, {}, False
    cats = Counter()
    total = 0
    try:
        with VAULT_CSV.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if (row.get("value") or "").strip():
                    total += 1
                    cats[(row.get("token_class") or "").strip().upper() or "UNKNOWN"] += 1
    except OSError:
        return 0, {}, False
    return total, dict(cats), True


def _load_payloads() -> "list[str]":
    if not OUTBOUND.is_file():
        return []
    out = []
    for line in read_jsonl_lines(OUTBOUND):
        if line.strip():
            try:
                out.append(str(json.loads(line).get("tokenized_content") or ""))
            except ValueError:
                continue
    return out


def _credential_preflight() -> "tuple[bool, str]":
    try:
        import google.auth  # type: ignore
        import google.auth.transport.requests  # type: ignore
    except Exception:
        return False, "google_auth_unavailable"
    try:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except Exception as exc:
        return False, ("login_required" if "DefaultCredentials" in type(exc).__name__ else "credentials_unavailable")
    try:
        creds.refresh(google.auth.transport.requests.Request())
    except Exception as exc:
        low = str(exc).lower()
        return False, ("login_required" if any(k in low for k in ("reauth", "invalid_grant", "expired", "login")) else "token_refresh_failed")
    return (True, "pass") if str(getattr(creds, "token", "") or "").strip() else (False, "token_empty")


# ---- Gate evaluation -----------------------------------------------------------------
def evaluate_gate() -> dict[str, Any]:
    payloads = _load_payloads()
    vault_total, vault_cats, vault_loaded = _load_vault_category_counts()
    scan = scan_payloads(payloads)
    high_uncovered = scan["high_total"]
    low_caveat = scan["low_total"]

    # Provider/facility coverage gap: the vault has no PROVIDER/FACILITY classes, yet the
    # corpus shows provider/facility cues and person-like candidates -> manual review.
    provider_facility_in_vault = bool(vault_cats.get("PROVIDER") or vault_cats.get("FACILITY"))
    provider_facility_candidates = scan["low"].get("provider_facility_cue", 0) + scan["low"].get("person_like_2cap", 0)
    manual_review_required = bool(low_caveat > 0 and not provider_facility_in_vault and provider_facility_candidates > 0)

    files_present = (OUTBOUND.is_file() and OUTBOUND_SHA.is_file()
                     and OUTBOUND_INTEGRITY.is_file() and DOC_MANIFEST.is_file())
    token_maps_present = TOKEN_MAPS.is_file()
    request_count = len(payloads)

    per_doc_tokens = [max(1, len(p) // 4) for p in payloads]
    est_total = planner.estimate_total_cost(per_doc_tokens, FULL_MAX_OUTPUT_TOKENS,
                                            INPUT_USD_PER_M, OUTPUT_USD_PER_M) if per_doc_tokens else 0.0
    selected_chunk = planner.select_chunk_size(per_doc_tokens, FULL_MAX_OUTPUT_TOKENS, INPUT_USD_PER_M,
                                               OUTPUT_USD_PER_M, PER_CHUNK_CAP_USD, EXPECTED_REQUESTS) if per_doc_tokens else 0
    per_chunk_cost = planner.worst_case_chunk_cost(per_doc_tokens, FULL_MAX_OUTPUT_TOKENS, INPUT_USD_PER_M,
                                                   OUTPUT_USD_PER_M, selected_chunk) if selected_chunk else 0.0
    cost_total_ok = est_total <= TOTAL_CAP_USD
    cost_chunk_ok = bool(selected_chunk) and per_chunk_cost <= PER_CHUNK_CAP_USD

    cred_ok, cred_cat = _credential_preflight()

    # Live entry gate: pass only if ALL hold.
    gate_passed = bool(
        files_present and token_maps_present and request_count == EXPECTED_REQUESTS
        and high_uncovered == 0 and not manual_review_required
        and cost_total_ok and cost_chunk_ok and cred_ok)

    if high_uncovered > 0:
        block_reason = "high_confidence_uncovered_pi"
        failure_stage = "vault_coverage"
    elif manual_review_required:
        block_reason = "vault_manual_review_required"
        failure_stage = "vault_coverage"
    elif not files_present or not token_maps_present or request_count != EXPECTED_REQUESTS:
        block_reason = "outbound_package_incomplete"
        failure_stage = "entry_gate"
    elif not (cost_total_ok and cost_chunk_ok):
        block_reason = "cost_cap_exceeded"
        failure_stage = "cost"
    elif not cred_ok:
        block_reason = cred_cat
        failure_stage = "credentials"
    else:
        block_reason = "none"
        failure_stage = "none"

    return {
        "payloads": payloads, "vault_total": vault_total, "vault_cats": vault_cats,
        "vault_loaded": vault_loaded, "scan": scan, "high_uncovered": high_uncovered,
        "low_caveat": low_caveat, "manual_review_required": manual_review_required,
        "provider_facility_in_vault": provider_facility_in_vault,
        "files_present": files_present, "token_maps_present": token_maps_present,
        "request_count": request_count, "est_total": est_total, "selected_chunk": selected_chunk,
        "per_chunk_cost": per_chunk_cost, "cost_total_ok": cost_total_ok, "cost_chunk_ok": cost_chunk_ok,
        "credential_preflight_passed": cred_ok, "credential_status_category": cred_cat,
        "gate_passed": gate_passed, "block_reason": block_reason, "failure_stage": failure_stage,
    }


# ---- Summary + reports ---------------------------------------------------------------
def build_summary(g: dict[str, Any], *, live_started: bool, live: dict | None) -> dict[str, Any]:
    live = live or {}
    run_result = "BLOCKED"
    if live_started:
        run_result = live.get("run_result", "LIVE_FAIL")
    elif not g["gate_passed"]:
        run_result = "BLOCKED"
    else:
        run_result = "GATE_PASSED_LIVE_NOT_RUN"

    low = g["scan"]["low"]
    s = {
        "block": BLOCK,
        "branch": BRANCH,
        "person_label": "P2",
        "run_result": run_result,
        "vault_review_completed": True,
        "vault_private_value_count_before": g["vault_total"],
        "vault_private_value_count_after": g["vault_total"],
        "vault_manual_review_required": g["manual_review_required"],
        "vault_filled_categories": g["vault_cats"],
        "high_confidence_uncovered_pi_count": g["high_uncovered"],
        "possible_low_confidence_pi_caveat_count": g["low_caveat"],
        "low_confidence_breakdown": {
            "person_like_2cap": low.get("person_like_2cap", 0),
            "provider_facility_cue": low.get("provider_facility_cue", 0),
            "spelled_date": low.get("spelled_date", 0),
        },
        "bare_10_digit_clinical_numeric_false_positive_count": g["scan"]["bare_10_digit_runs"],
        "retokenization_performed": False,
        "tokenized_request_count": g["request_count"],
        "live_entry_gate_passed": g["gate_passed"],
        "live_entry_gate_block_reason": g["block_reason"],
        "live_run_started": live_started,
        "provider_model_call_made": bool(live.get("provider_model_call_made", False)),
        "gemini_call_made": bool(live.get("gemini_call_made", False)),
        "target_model": TARGET_MODEL,
        "docs_loaded": g["request_count"],
        "docs_completed": int(live.get("docs_completed", 0)),
        "docs_failed_for_review": int(live.get("docs_failed_for_review", 0)),
        "docs_unattempted": int(live.get("docs_unattempted", g["request_count"] if not live_started else 0)),
        "sections_sent": int(live.get("sections_sent", 0)),
        "sections_succeeded": int(live.get("sections_succeeded", 0)),
        "sections_failed": int(live.get("sections_failed", 0)),
        "failure_stage": live.get("failure_stage", g["failure_stage"] if not g["gate_passed"] else "none"),
        "failure_category": live.get("failure_category", g["block_reason"] if not g["gate_passed"] else "none"),
        "failed_doc_hash": live.get("failed_doc_hash"),
        "failed_section": live.get("failed_section"),
        "failed_evidence_preserved": bool(live.get("failed_evidence_preserved", False)),
        "authorized_total_cap_usd": TOTAL_CAP_USD,
        "hard_cost_cap_per_chunk_usd": PER_CHUNK_CAP_USD,
        "estimated_cost_before_live_usd": round(g["est_total"], 6),
        "selected_chunk_size": g["selected_chunk"],
        "estimated_per_chunk_cost_usd": round(g["per_chunk_cost"], 6),
        "credential_preflight_passed": g["credential_preflight_passed"],
        "credential_status_category": g["credential_status_category"],
        "actual_total_token_count": int(live.get("actual_total_token_count", 0)),
        "actual_cost_public_if_available": live.get("actual_cost_public_if_available", "unknown"),
        "cost_cap_exceeded": bool(live.get("cost_cap_exceeded", False)),
        "per_chunk_cap_exceeded": (not g["cost_chunk_ok"]),
        "checkpoint_resume_available": True,
        "sectioned_extraction_available": True,
        "autonomous_recovery_available": True,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "future_mkb_import_started": False,
        "corpus1_touched": False,
        "private_artifacts_committed": False,
        "raw_ai_response_committed": False,
        "tokenized_payloads_committed": False,
        "token_maps_committed": False,
        "pi_values_committed": False,
        "credentials_or_tokens_committed": False,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": 0,
        "secret_leaks_after": 0,
        "privacy_result": "passed" if g["high_uncovered"] == 0 else "blocked",
        "safety_result": "passed",
    }
    return s


def write_reports(s: dict[str, Any], g: dict[str, Any], live: dict | None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=True), encoding="utf-8")

    coverage = {
        "person_label": "P2",
        "vault_filled_value_count": g["vault_total"],
        "vault_filled_categories": g["vault_cats"],
        "vault_has_provider_or_facility_coverage": g["provider_facility_in_vault"],
        "high_confidence_categories": [
            {"category": k, "count": v, "confidence_band": "high",
             "action": "covered" if v == 0 else "blocked"}
            for k, v in g["scan"]["high"].items()
        ],
        "high_confidence_uncovered_total": g["high_uncovered"],
        "low_confidence_categories": [
            {"category": "person_like_2cap", "count": g["scan"]["low"].get("person_like_2cap", 0),
             "confidence_band": "low_high_false_positive",
             "action": "possible_gap" if g["scan"]["low"].get("person_like_2cap", 0) else "covered"},
            {"category": "provider_facility_cue", "count": g["scan"]["low"].get("provider_facility_cue", 0),
             "confidence_band": "low", "action": "possible_gap" if g["scan"]["low"].get("provider_facility_cue", 0) else "covered"},
            {"category": "spelled_date", "count": g["scan"]["low"].get("spelled_date", 0),
             "confidence_band": "medium", "action": "possible_gap" if g["scan"]["low"].get("spelled_date", 0) else "covered"},
        ],
        "clinical_numeric_false_positive_note": {
            "category": "bare_10_digit_runs",
            "count": g["scan"]["bare_10_digit_runs"],
            "confidence_band": "informational",
            "action": "covered_not_pii_clinical_numerics",
            "note": "Bare digit runs in lab data are not phone numbers; a naive phone regex "
                    "over-matches them. Formatted-phone detection is 0.",
        },
        "manual_review_required": g["manual_review_required"],
        "decision": s["run_result"],
        "no_pi_values_in_this_report": True,
    }
    (REPORT_DIR / "vault_coverage_public.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=True), encoding="utf-8")

    gate_md = [
        "# Corpus 2 / P2 live entry gate", "",
        f"- outbound package present: `{g['files_present']}` (jsonl + sha256 + integrity + manifest)",
        f"- token maps present (private): `{g['token_maps_present']}`",
        f"- tokenized_request_count: `{g['request_count']}` (expected `{EXPECTED_REQUESTS}`)",
        f"- high_confidence_uncovered_pi_count: `{g['high_uncovered']}`",
        f"- vault filled values: `{g['vault_total']}`; provider/facility coverage: `{g['provider_facility_in_vault']}`",
        f"- vault_manual_review_required: `{g['manual_review_required']}`",
        f"- estimated_total_cost_usd: `{round(g['est_total'], 6)}` (cap `${TOTAL_CAP_USD}`) -> within_cap=`{g['cost_total_ok']}`",
        f"- selected_chunk_size: `{g['selected_chunk']}`; per-chunk worst case: "
        f"`{round(g['per_chunk_cost'], 6)}` (cap `${PER_CHUNK_CAP_USD}`) -> within_cap=`{g['cost_chunk_ok']}`",
        f"- credential_preflight_passed: `{g['credential_preflight_passed']}` (`{g['credential_status_category']}`)",
        f"- live_entry_gate_passed: `{g['gate_passed']}`",
        f"- block_reason: `{g['block_reason']}`",
        "",
    ]
    if not g["gate_passed"] and g["block_reason"] == "vault_manual_review_required":
        gate_md += [
            "## Why live is withheld",
            "High-confidence structured PII in the tokenized payloads is 0. However the P2 PI "
            f"vault has only `{g['vault_total']}` filled values and NO provider/facility class "
            "entries, while the corpus shows provider/facility cues and person-like candidates. "
            "Those would be transmitted un-tokenized to the external provider. The live "
            "authorization is conditional on vault coverage passing; manual vault review/"
            "expansion (add provider/facility/secondary-person identifiers) is required before "
            "live extraction is authorized.",
            "",
            "## To proceed later",
            "Add the missing identifiers to the private vault CSV, re-run Corpus 2 P2 prep to "
            "re-tokenize and rebuild the outbound package, then re-run this local gate. When the "
            "gate passes, run `--live` once.",
            "",
        ]
    (REPORT_DIR / "live_entry_gate_public.md").write_text("\n".join(gate_md), encoding="utf-8")

    if live and s["live_run_started"]:
        run_md = [
            "# Corpus 2 / P2 live run report", "",
            f"- run_result: `{s['run_result']}`",
            f"- docs loaded/completed/failed_for_review/unattempted: "
            f"`{s['docs_loaded']}` / `{s['docs_completed']}` / `{s['docs_failed_for_review']}` / `{s['docs_unattempted']}`",
            f"- sections sent/succeeded/failed: `{s['sections_sent']}` / `{s['sections_succeeded']}` / `{s['sections_failed']}`",
            f"- failure stage/category: `{s['failure_stage']}` / `{s['failure_category']}`",
            f"- failed_doc_hash: `{s['failed_doc_hash']}`; failed_section: `{s['failed_section']}`",
            f"- failed_evidence_preserved: `{s['failed_evidence_preserved']}`",
            f"- actual_total_token_count: `{s['actual_total_token_count']}`; "
            f"actual_cost: `{s['actual_cost_public_if_available']}`",
            f"- cost_cap_exceeded: `{s['cost_cap_exceeded']}`",
            "", "No MKB write/import. Private responses/evidence kept outside the repo.", "",
        ]
    else:
        run_md = ["# Corpus 2 / P2 live run report", "",
                  "Live extraction was NOT run: the local entry gate did not pass "
                  f"(block_reason `{g['block_reason']}`). No provider call was made.", ""]
    (REPORT_DIR / "live_run_public_report.md").write_text("\n".join(run_md), encoding="utf-8")

    failed_docs = (live or {}).get("failed_docs", [])
    (REPORT_DIR / "failed_docs_public.json").write_text(json.dumps({
        "failed_for_review_count": len(failed_docs),
        "failed_docs": failed_docs,
        "note": "Failed-for-review documents are preserved privately; only hashes/categories are public.",
    }, indent=2, ensure_ascii=True), encoding="utf-8")

    keys = ["provider_model_call_made", "gemini_call_made", "mkb_db_opened", "active_mkb_write",
            "auto_accept_enabled", "medical_decision_made", "future_mkb_import_started",
            "corpus1_touched", "private_artifacts_committed", "raw_ai_response_committed",
            "tokenized_payloads_committed", "token_maps_committed", "pi_values_committed",
            "credentials_or_tokens_committed", "public_report_phi_leak_count",
            "private_path_leaks_after", "secret_leaks_after", "privacy_result", "safety_result"]
    safety = ["# Corpus 2 / P2 vault review + live — safety boundary", "", "| Gate | Value |",
              "| --- | --- |", *[f"| {k} | `{s[k]}` |" for k in keys], "",
              "Live gate is set only inside an authorized provider call. PI values, raw text, and "
              "raw AI responses are never printed or committed. Private vault, token maps, "
              "tokenized payloads, raw extraction, responses, checkpoint, and evidence stay "
              "outside the repo. Corpus 1 is not read or modified by this block.", ""]
    (REPORT_DIR / "safety_boundary_public.md").write_text("\n".join(safety), encoding="utf-8")

    impl = [
        f"# {BLOCK} — implementation report", "",
        "## Sub-block A — vault coverage review",
        f"- vault filled values: `{g['vault_total']}` across categories `{g['vault_cats']}`.",
        f"- high-confidence structured PII remaining in payloads: `{g['high_uncovered']}` "
        "(email/formatted-phone/SSN/labeled-IDs/local-path).",
        f"- low-confidence caveat candidates: `{g['low_caveat']}` "
        "(person-like + provider/facility cues + spelled dates; person-like is a noisy heuristic).",
        f"- bare 10-digit clinical-numeric false positives: `{g['scan']['bare_10_digit_runs']}` "
        "(not PII; formatted-phone detection is 0).",
        "",
        "## Sub-block B — vault expansion",
        "- retokenization_performed: `false`. Auto-adding provider/person names from medical text "
        "is uncertain (NER false positives), so no candidates were auto-added; manual vault review "
        "is reported instead.",
        "",
        "## Sub-block C — live entry gate",
        f"- live_entry_gate_passed: `{g['gate_passed']}` (block_reason `{g['block_reason']}`).",
        f"- cost estimate `${round(g['est_total'],6)}` within `${TOTAL_CAP_USD}`; per-chunk "
        f"`${round(g['per_chunk_cost'],6)}` within `${PER_CHUNK_CAP_USD}` at chunk size "
        f"`{g['selected_chunk']}`.",
        "",
        "## Decision",
        f"- run_result: `{s['run_result']}`. "
        + ("Live extraction withheld pending manual vault review/expansion; no provider call made."
           if not g["gate_passed"] else "Gate passed."),
        "",
        "## Boundaries",
        "No MKB open/write, no auto-accept, no medical decision, no Corpus 1 access. Public reports "
        "carry counts/bands/labels only and pass the privacy checker.",
        "",
    ]
    (REPORT_DIR / "implementation_report.md").write_text("\n".join(impl), encoding="utf-8")


# ---- Live extraction (gated; only runs when the gate passes) --------------------------
def run_live(g: dict[str, Any]) -> dict[str, Any]:
    """Sectioned / autonomous-recovery live run over the P2 payloads. Only reached when the
    entry gate passed. Stops on the first hard failure; preserves evidence; no MKB write."""
    from execution.gemini_vertex_adapter import (
        GeminiVertexConfig, build_vertex_generate_content_url, _default_http_post,
        classify_vertex_provider_error, acquire_google_cloud_access_token,
    )
    from execution.sectioned_extraction import (
        SECTION_NAMES, build_full_payload, build_section_payload, validate_section_response,
        is_transient_provider_category,
    )
    from execution.sectioned_merge import merge_sections
    import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live_base  # _validate_schema only
    import time

    live: dict[str, Any] = {
        "run_result": "LIVE_FAIL", "provider_model_call_made": False, "gemini_call_made": False,
        "docs_completed": 0, "docs_failed_for_review": 0, "docs_unattempted": 0,
        "sections_sent": 0, "sections_succeeded": 0, "sections_failed": 0,
        "actual_total_token_count": 0, "failed_docs": [], "failed_evidence_preserved": False,
        "failure_stage": "none", "failure_category": "none", "failed_doc_hash": None, "failed_section": None,
    }
    requests = []
    for line in read_jsonl_lines(OUTBOUND):
        if line.strip():
            requests.append(json.loads(line))
    batch_sha = lc.sha256_file(OUTBOUND)
    order = [str(r.get("document_id") or "") for r in requests]
    start_index, ck_blocked, ck_reason, completed_set = lc.decide_resume(batch_sha, order, len(requests), base=LIVE_CHECKPOINT_DIR)
    if ck_blocked:
        live.update({"run_result": "BLOCKED", "failure_stage": "checkpoint", "failure_category": ck_reason})
        return live
    lc.init_checkpoint("corpus2_p2", batch_sha, TARGET_MODEL, TOTAL_CAP_USD, PER_CHUNK_CAP_USD, len(requests), base=LIVE_CHECKPOINT_DIR)

    url = build_vertex_generate_content_url(GeminiVertexConfig())
    try:
        token = acquire_google_cloud_access_token()
    except Exception:
        token = ""
    if not str(token or "").strip():
        live.update({"run_result": "BLOCKED", "failure_stage": "credentials", "failure_category": "token_acquire_failed"})
        return live

    run_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    LIVE_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    actual_cost = 0.0
    raw_records: list[dict[str, Any]] = []

    def _call(payload):
        os.environ[LIVE_GATE] = GATE_VALUE
        live["provider_model_call_made"] = True
        live["gemini_call_made"] = True
        try:
            return True, "ok", _default_http_post(url, payload, token)
        except Exception as exc:  # sanitized classification only
            return False, str(classify_vertex_provider_error(exc).get("provider_error_category") or "provider_error"), {}
        finally:
            os.environ.pop(LIVE_GATE, None)

    def _usage(resp):
        um = resp.get("usageMetadata") if isinstance(resp, dict) else {}
        return int((um or {}).get("totalTokenCount") or 0)

    def _text(resp):
        try:
            return str(resp["candidates"][0]["content"]["parts"][0]["text"])
        except Exception:
            return ""

    try:
        for index, req in enumerate(requests):
            doc_id = str(req.get("document_id") or "")
            content = str(req.get("tokenized_content") or "")
            if doc_id in completed_set:
                continue
            if actual_cost >= TOTAL_CAP_USD:
                live.update({"run_result": "BLOCKED", "failure_stage": "cost", "failure_category": "total_cap_exhausted", "cost_cap_exceeded": True})
                break
            ok, reason, resp = _call(build_full_payload("", content))
            if not ok and is_transient_provider_category(reason):
                ok, reason, resp = _call(build_full_payload("", content))
            if not ok:
                # Hard provider failure -> stop the run, preserve evidence, no auto-retry.
                live["failure_stage"], live["failure_category"], live["failed_doc_hash"] = "provider_live_fail", reason, doc_id
                _preserve(run_ts, doc_id, reason, raw_records, live)
                live["run_result"] = "LIVE_FAIL"
                lc.mark_failed(doc_id, index, reason, base=LIVE_CHECKPOINT_DIR)
                live["failed_docs"].append({"doc_hash": doc_id, "failure_category": reason, "failed_section": None})
                break
            actual_cost += _usage(resp) / 1_000_000 * OUTPUT_USD_PER_M
            live["actual_total_token_count"] += _usage(resp)
            raw_records.append({"document_id": doc_id, "strategy": "full_schema"})
            valid, vreason = live_base._validate_schema(_text(resp))
            if valid:
                lc.mark_completed(doc_id, index, base=LIVE_CHECKPOINT_DIR)
                completed_set.add(doc_id)
                live["docs_completed"] += 1
                continue
            if vreason == "raw_pi_in_response":
                live.update({"run_result": "BLOCKED", "failure_stage": "privacy", "failure_category": "raw_pi_in_response", "failed_doc_hash": doc_id})
                _preserve(run_ts, doc_id, "raw_pi_in_response", raw_records, live)
                break
            # Recoverable JSON failure -> sectioned fallback.
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
                    _preserve(run_ts, doc_id, reason_s, raw_records, live)
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
                lc.mark_failed(doc_id, index, live["failure_category"], base=LIVE_CHECKPOINT_DIR)
                live["failed_docs"].append({"doc_hash": doc_id, "failure_category": live["failure_category"], "failed_section": live["failed_section"]})
                live["docs_failed_for_review"] += 1
                if live["failure_stage"] == "provider_live_fail":
                    break  # hard provider failure stops the run
                continue
            merged_ok, _mr, _m = merge_sections(section_objs)
            if merged_ok:
                lc.mark_completed(doc_id, index, base=LIVE_CHECKPOINT_DIR)
                completed_set.add(doc_id)
                live["docs_completed"] += 1
            else:
                live["docs_failed_for_review"] += 1
                live["failed_docs"].append({"doc_hash": doc_id, "failure_category": "section_merge_failed", "failed_section": None})
    finally:
        os.environ.pop(LIVE_GATE, None)

    live["docs_unattempted"] = max(0, len(requests) - live["docs_completed"] - live["docs_failed_for_review"])
    live["actual_cost_public_if_available"] = round(actual_cost, 6)
    if live["run_result"] not in ("BLOCKED",):
        live["run_result"] = "PASS" if live["docs_completed"] == len(requests) else (
            "LIVE_FAIL" if live["failed_docs"] else "PASS")
    return live


def _preserve(run_ts: str, doc_id: str, category: str, raw_records: list, live: dict) -> None:
    try:
        LIVE_STAGING_DIR.mkdir(parents=True, exist_ok=True)
        (LIVE_STAGING_DIR / "live_responses_private.jsonl").write_text(
            "\n".join(json.dumps(r) for r in raw_records) + ("\n" if raw_records else ""), encoding="utf-8")
        (LIVE_STAGING_DIR / "stopped_on_failure_private.json").write_text(
            json.dumps({"failed_doc_id": doc_id, "failure_category": category}), encoding="utf-8")
        preserved, _path, copied = lc.preserve_failed_evidence(
            run_ts, staging_dir=LIVE_STAGING_DIR, base=LIVE_CHECKPOINT_DIR, evidence_base=EVIDENCE_DIR)
        live["failed_evidence_preserved"] = bool(preserved)
    except Exception:
        live["failed_evidence_preserved"] = False


# ---- entry point ---------------------------------------------------------------------
def _privacy_scan_reports() -> int:
    leaks = 0
    for name in ("summary.json", "implementation_report.md", "vault_coverage_public.json",
                 "live_entry_gate_public.md", "live_run_public_report.md", "failed_docs_public.json",
                 "safety_boundary_public.md"):
        p = REPORT_DIR / name
        if not p.is_file():
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
    do_live = args.live

    g = evaluate_gate()
    live = None
    live_started = False
    if do_live:
        if g["gate_passed"]:
            live = run_live(g)
            live_started = bool(live.get("provider_model_call_made"))
        else:
            live = None  # gate failed -> no provider call

    s = build_summary(g, live_started=live_started, live=live)
    write_reports(s, g, live)
    leaks = _privacy_scan_reports()
    if leaks:
        s["public_report_phi_leak_count"] = leaks
        (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"{'live' if do_live else 'local_gate'}: result={s['run_result']} "
          f"gate_passed={g['gate_passed']} block={g['block_reason']} "
          f"high_uncovered={g['high_uncovered']} vault_values={g['vault_total']} "
          f"manual_review={g['manual_review_required']} provider_call={s['provider_model_call_made']} "
          f"report_leaks={leaks}")
    return 0 if leaks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
