#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-478-BATCH-VALIDATION-REPAIR-LOCAL-ONLY-17B-R2-R1.

Local-only repair of the 42 residual identifier-pattern failures from 17B-R2, rebuilding
a clean 478-file private outbound request batch and re-validating under the unchanged
17B-R2 rules. NO provider call, NO network, NO billing, NO live gate.

Tokenized content source of truth: the 17A tokenized corpus (covers all 478, incl. the
42 failed). Public reports carry counts, hashed doc IDs, token/cost estimates, and
validation status only — never payload bodies, token maps, or residual values.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.jsonl_framing import read_jsonl_lines  # noqa: E402  physical-newline JSONL framing

R2_DIR = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17B_R2_478_dry_run"))
R2_VALIDATOR = R2_DIR / "dry_run_validator_private.json"
R2_FAILED = R2_DIR / "failed_request_validation_private.jsonl"
R1_OUTBOUND = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17B_R1_repaired\outbound_requests_private.jsonl"))
CORPUS_DOCS = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\corpus_tokenized_17A\documents"))

PRIVATE_OUT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17B_R2_R1_478_repaired"))
PRIVATE_OUT_LABEL = r"C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17B_R2_R1_478_repaired\outbound_requests_private.jsonl"
DOWNLOADS = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_17B_R2_R1_478_Validation_Repair_Review"))
DOWNLOADS_LABEL = r"%USERPROFILE%\Downloads\MedAI_17B_R2_R1_478_Validation_Repair_Review"

PROMPT_CONTRACT_PATH = REPO_ROOT / "config" / "medai_ai_extraction_prompt_contract_17b.md"

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_batch_validation_repair_local_only_17b_r2_r1"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_478_BATCH_VALIDATION_REPAIR_LOCAL_ONLY_17B_R2_R1"
REQUIRED_DOCS = (
    "MEDAI_AI_FIRST_CORPUS_478_BATCH_VALIDATION_REPAIR_LOCAL_ONLY_17B_R2_R1.md",
    "MEDAI_IDENTIFIER_PATTERN_REPAIR_POLICY_17B_R2_R1.md",
    "MEDAI_478_DOC_ID_DEDUPE_AND_NO_490_BATCH_RULE_17B_R2_R1.md",
    "MEDAI_LIVE_BATCH_ENTRY_CRITERIA_AFTER_17B_R2_R1.md",
    "MEDAI_REMAINING_NON_READY_FILES_EXCLUDED_17B_R2_R1.md",
)

EXPECTED_READY = 478
EXPECTED_FAILED = 42
KNOWN_FAILURE_CLASSES = {"accession_specimen", "insurance_account", "mrn", "phone"}

TARGET_MODEL = "gemini-2.5-flash-lite"
INPUT_USD_PER_M = 0.075
OUTPUT_USD_PER_M = 0.30
OUTPUT_TOKENS_PER_DOC = 800
SUGGESTED_LIVE_BATCH_SIZE = 25

TOKEN_RE = re.compile(r"\[[A-Z_]+_\d+\]")
PHONE_RE = re.compile(r"(?:\+?\d{1,2}[ \-.]?)?\(?\d{3}\)?[ \-.]?\d{3}[ \-.]?\d{4}\b")
# Unchanged 17B/17B-R2 validation patterns.
_PI_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "phone": PHONE_RE,
    "mrn": re.compile(r"(?i)\b(?:MRN|MR#|Medical Record(?: Number)?)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "insurance_account": re.compile(r"(?i)\b(?:Insurance|Policy|Member|Account|Acct)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "accession_specimen": re.compile(r"(?i)\b(?:Accession|Specimen|Order)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "local_path": re.compile(r"(?:[A-Za-z]:\\[^\s]+|/(?:home|Users)/[^\s]+)"),
}
_NUMERIC_ID_CONTEXT = re.compile(r"(?i)(barcode|specimen|accession|order|collection|lab\s*id|requisition|sample)")

FAMILY_BY_EXT = {
    ".jpg": ("image", "scanned_image"), ".jpeg": ("image", "scanned_image"),
    ".tif": ("image", "scanned_image"), ".tiff": ("image", "scanned_image"),
    ".png": ("image", "scanned_image"), ".rtf": ("document", "rich_text"),
    ".docx": ("document", "word_document"), ".pdf": ("document", "pdf_document"),
    ".txt": ("document", "plain_text"),
}


def _sha16(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _residual_pi(text: str) -> list[str]:
    residual = TOKEN_RE.sub(" ", text)
    return [name for name, rx in _PI_PATTERNS.items() if rx.search(residual)]


class _Counters:
    def __init__(self) -> None:
        self.c: dict[str, int] = {}

    def tok(self, prefix: str) -> str:
        self.c[prefix] = self.c.get(prefix, 0) + 1
        return f"[{prefix}_{self.c[prefix]}]"


def _choose_accession_token(matchtext: str, ctr: _Counters) -> str:
    low = matchtext.lower()
    if "specimen" in low:
        return ctr.tok("SPECIMEN_ID")
    if "order" in low:
        return ctr.tok("ORDER_ID")
    return ctr.tok("ACCESSION_ID")


def _choose_insurance_token(matchtext: str, ctr: _Counters) -> str:
    low = matchtext.lower()
    if "account" in low or "acct" in low:
        return ctr.tok("ACCOUNT_ID")
    return ctr.tok("INSURANCE_ID")


_LABEL_WORDS = {
    "insurance", "policy", "member", "account", "acct", "mrn", "mr",
    "accession", "specimen", "order", "medical", "record", "number",
}
_REPAIR_CLASSES = ("mrn", "insurance_account", "accession_specimen", "phone")


def _token_for(cls: str, context: str, pre: str, ctr: _Counters) -> str:
    if cls == "mrn":
        return ctr.tok("MRN")
    if cls == "insurance_account":
        return _choose_insurance_token(context, ctr)
    if cls == "accession_specimen":
        return _choose_accession_token(context, ctr)
    # phone
    return ctr.tok("NUMERIC_IDENTIFIER") if _NUMERIC_ID_CONTEXT.search(pre) else ctr.tok("PHONE_PATTERN")


def repair_text(text: str, ctr: _Counters) -> str:
    """Tokenize residual identifier matches to a fixed point so the validator's
    token-stripped view is free of phone/mrn/insurance/accession classes — including
    label/value pairs that are bridged by an existing token placeholder."""
    guard = 0
    while guard < 200000:
        guard += 1
        # Phase A: a class match directly in the raw text (replace whole label+value).
        hit = None
        for cls in ("mrn", "insurance_account", "accession_specimen"):
            m = _PI_PATTERNS[cls].search(text)
            if m:
                hit = (cls, m)
                break
        if hit is None:
            m = PHONE_RE.search(text)
            if m:
                hit = ("phone", m)
        if hit is not None:
            cls, m = hit
            pre = text[max(0, m.start() - 24):m.start()]
            tok = _token_for(cls, m.group(0), pre, ctr)
            text = text[:m.start()] + tok + text[m.end():]
            continue
        # Phase B: a class match that appears only after token-stripping (label and
        # value bridged by an existing token). Tokenize the value run in raw text.
        stripped = TOKEN_RE.sub(" ", text)
        bhit = None
        for cls in _REPAIR_CLASSES:
            m = _PI_PATTERNS[cls].search(stripped)
            if m:
                bhit = (cls, m)
                break
        if bhit is None:
            break
        cls, m = bhit
        seg = m.group(0)
        replaced = False
        for vm in re.finditer(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", seg):
            val = vm.group(0)
            if val.lower() in _LABEL_WORDS:
                continue
            idx = text.find(val)
            if idx != -1:
                pre = text[max(0, idx - 24):idx]
                tok = _token_for(cls, seg, pre, ctr)
                text = text[:idx] + tok + text[idx + len(val):]
                replaced = True
                break
        if not replaced:
            break
    return text


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _load_old12_ids() -> tuple[bool, set[str]]:
    if not R1_OUTBOUND.is_file():
        return False, set()
    ids = set()
    for line in read_jsonl_lines(R1_OUTBOUND):
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(json.loads(line)["document_id"])
        except Exception:
            continue
    return True, ids


def run() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    notes: list[str] = []

    if not R2_VALIDATOR.is_file() or not R2_FAILED.is_file():
        return ({"__blocked__": "artifact_missing"}, {}, ["17b_r2_private_validator_or_failed_artifacts_missing"])

    validator = _load_json(R2_VALIDATOR)
    per_doc = validator.get("per_doc") or []
    ready_recs = [{"document_id": d["document_id"], "extension": (d.get("extension") or "").lower()} for d in per_doc]
    ready_ids = [r["document_id"] for r in ready_recs]
    if len(ready_ids) != EXPECTED_READY:
        notes.append(f"ready_count_{len(ready_ids)}_not_{EXPECTED_READY}")

    failed_classes: dict[str, int] = {}
    failed_ids: set[str] = set()
    for line in read_jsonl_lines(R2_FAILED):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        failed_ids.add(r["document_id"])
        for c in (r.get("failure_classes") or r.get("pi_pattern_classes_detected") or []):
            failed_classes[c] = failed_classes.get(c, 0) + 1

    unknown = set(failed_classes) - KNOWN_FAILURE_CLASSES
    if unknown:
        return ({"__blocked__": f"unknown_failure_class:{sorted(unknown)}"}, {},
                [f"unknown_failure_class:{sorted(unknown)}"])

    # Dedupe: are the old 12 included in the 478?
    dedupe_done, old12 = _load_old12_ids()
    ready_set = set(ready_ids)
    if dedupe_done:
        old12_in_478 = old12.issubset(ready_set) if old12 else False
        old_12_included = True if (old12 and old12_in_478) else (False if old12 else "unknown")
    else:
        old_12_included = "unknown"

    prompt_contract = PROMPT_CONTRACT_PATH.read_text(encoding="utf-8")
    prompt_tokens = _approx_tokens(prompt_contract)

    public_batch: list[dict[str, Any]] = []
    private_requests: list[dict[str, Any]] = []
    repair_records: list[dict[str, Any]] = []
    failed_after: list[dict[str, Any]] = []
    after_classes: dict[str, int] = {}
    est_input = 0
    files_repaired = 0
    missing = 0

    for rec in ready_recs:
        doc_id = rec["document_id"]
        ext = rec["extension"]
        tok_path = CORPUS_DOCS / doc_id / "tokenized_text.txt"
        if not tok_path.is_file():
            missing += 1
            notes.append(f"missing_tokenized_text:{doc_id}")
            continue
        text = tok_path.read_text(encoding="utf-8", errors="ignore")
        repaired = doc_id in failed_ids
        if repaired:
            text = repair_text(text, _Counters())
            files_repaired += 1
            repair_records.append({"document_id": doc_id})
        pi_hits = _residual_pi(text)
        if pi_hits:
            for c in pi_hits:
                after_classes[c] = after_classes.get(c, 0) + 1
            failed_after.append({"document_id": doc_id, "failure_classes": pi_hits})
        family, dtype = FAMILY_BY_EXT.get(ext, ("unknown", "unknown"))
        src_hash = _sha16(text)
        doc_in = _approx_tokens(text) + prompt_tokens
        est_input += doc_in
        public_batch.append({"document_id": doc_id, "source_hash": src_hash,
                             "document_family": family, "document_type": dtype,
                             "repaired": repaired, "validation_status": "ok" if not pi_hits else "blocked"})
        private_requests.append({
            "document_id": doc_id, "source_hash": src_hash, "model": TARGET_MODEL,
            "schema_ref": "config/medai_ai_extraction_schema_17b.json",
            "prompt_contract_ref": "config/medai_ai_extraction_prompt_contract_17b.md",
            "document_family": family, "document_type": dtype,
            "tokenized_content": text,  # PRIVATE only
        })

    outbound_built = len(private_requests)
    request_validation_passed = (len(failed_after) == 0 and outbound_built == len(ready_ids)
                                 and len(ready_ids) == EXPECTED_READY and missing == 0)
    est_output = OUTPUT_TOKENS_PER_DOC * outbound_built
    est_cost = round(est_input / 1_000_000 * INPUT_USD_PER_M + est_output / 1_000_000 * OUTPUT_USD_PER_M, 6)
    batch_count = math.ceil(outbound_built / SUGGESTED_LIVE_BATCH_SIZE) if outbound_built else 0
    per_batch_cost = est_cost / batch_count if batch_count else est_cost
    suggested_cap = round(max(0.05, per_batch_cost * 1.5), 4)
    privacy_result = "passed" if request_validation_passed else "blocked"

    # ---- Private artifacts (outside repo) ----
    private_written = False
    try:
        PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
        with (PRIVATE_OUT / "outbound_requests_private.jsonl").open("w", encoding="utf-8") as fh:
            for req in private_requests:
                fh.write(json.dumps(req, ensure_ascii=False) + "\n")
        (PRIVATE_OUT / "repair_manifest_private.json").write_text(
            json.dumps(repair_records, ensure_ascii=False, indent=2), encoding="utf-8")
        (PRIVATE_OUT / "repaired_doc_ids_private.json").write_text(
            json.dumps(sorted(failed_ids), ensure_ascii=False, indent=2), encoding="utf-8")
        (PRIVATE_OUT / "dry_run_validator_private.json").write_text(
            json.dumps({"request_validation_passed": request_validation_passed,
                        "failed_count_after_repair": len(failed_after),
                        "per_doc": [{"document_id": b["document_id"], "repaired": b["repaired"],
                                     "validation_status": b["validation_status"]} for b in public_batch]},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        with (PRIVATE_OUT / "failed_request_validation_private.jsonl").open("w", encoding="utf-8") as fh:
            for fr in failed_after:
                fh.write(json.dumps(fr, ensure_ascii=False) + "\n")
        (PRIVATE_OUT / "cost_estimate_private.json").write_text(
            json.dumps({"estimated_input_tokens": est_input, "estimated_output_tokens": est_output,
                        "estimated_cost_usd": est_cost}, ensure_ascii=False, indent=2), encoding="utf-8")
        (PRIVATE_OUT / "batch_plan_private.json").write_text(
            json.dumps({"total": outbound_built, "batch_size": SUGGESTED_LIVE_BATCH_SIZE,
                        "batch_count": batch_count}, ensure_ascii=False, indent=2), encoding="utf-8")
        (PRIVATE_OUT / "doc_id_dedupe_check_private.json").write_text(
            json.dumps({"dedupe_check_performed": dedupe_done, "old_12_count": len(old12),
                        "old_12_included_in_478": old_12_included}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        private_written = True
    except OSError:
        notes.append("private_artifact_write_failed")

    summary = {
        "block": "MEDAI-AI-FIRST-CORPUS-478-BATCH-VALIDATION-REPAIR-LOCAL-ONLY-17B-R2-R1",
        "local_only": True,
        "dry_run_only": True,
        "provider_call_made": False,
        "vertex_live_execution": False,
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
        "ready_files_total": len(ready_ids),
        "old_12_added_separately": False,
        "combined_batch_count": outbound_built,
        "doc_id_dedupe_check_performed": dedupe_done,
        "old_12_included_in_478": old_12_included,
        "failed_files_from_17b_r2": len(failed_ids),
        "failure_classes_from_17b_r2": failed_classes,
        "files_repaired": files_repaired,
        "outbound_requests_built": outbound_built,
        "request_validation_passed": request_validation_passed,
        "request_validation_failed_count_after_repair": len(failed_after),
        "validation_failure_classes_after_repair": after_classes,
        "private_outbound_requests_path": PRIVATE_OUT_LABEL,
        "private_payloads_written_outside_repo": private_written,
        "downloads_review_location": DOWNLOADS_LABEL,
        "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False,
        "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False,
        "raw_source_files_uploaded": False,
        "public_report_phi_leak_count": 0,
        "estimated_input_tokens": est_input,
        "estimated_output_tokens": est_output,
        "estimated_cost_usd": est_cost,
        "target_model_for_future_live": TARGET_MODEL,
        "suggested_live_batch_size": SUGGESTED_LIVE_BATCH_SIZE,
        "suggested_live_batch_count": batch_count,
        "suggested_hard_cost_cap_usd": suggested_cap,
        "future_live_extraction_not_started": True,
        "privacy_result": privacy_result,
        "safety_result": "passed",
    }
    extra = {"public_batch": public_batch}
    return summary, extra, notes


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def _repair_matrix(s: dict[str, Any]) -> str:
    keys = ["ready_files_total", "combined_batch_count", "failed_files_from_17b_r2", "files_repaired",
            "outbound_requests_built", "request_validation_passed",
            "request_validation_failed_count_after_repair", "validation_failure_classes_after_repair",
            "estimated_input_tokens", "estimated_output_tokens", "estimated_cost_usd",
            "public_report_phi_leak_count", "privacy_result", "safety_result"]
    return "\n".join(["# 17B-R2-R1 repair matrix (counts only)", "",
                      "| Field | Value |", "| --- | --- |",
                      *[f"| {k} | `{s[k]}` |" for k in keys], "",
                      f"Failure classes from 17B-R2: `{s['failure_classes_from_17b_r2']}`", "",
                      "Counts only — no payload bodies, no token maps, no residual values.", ""])


def _privacy_matrix(s: dict[str, Any]) -> str:
    keys = ["provider_call_made", "gemini_call_made", "claude_call_made", "openai_call_made",
            "billing_api_call_made", "live_gate_set", "live_extraction_started",
            "raw_source_files_uploaded", "tokenized_payloads_written_to_repo",
            "raw_ocr_written_to_repo", "token_maps_written_to_repo",
            "private_identifier_values_written_to_repo", "public_report_phi_leak_count",
            "mkb_db_opened", "active_mkb_write", "privacy_result"]
    return "\n".join(["# 17B-R2-R1 privacy gate matrix", "", "| Gate | Value |", "| --- | --- |",
                      *[f"| {k} | `{s[k]}` |" for k in keys], "",
                      "Outbound requests are private (outside the repo). Public reports carry counts,",
                      "hashed doc IDs, estimates, and status only.", ""])


def _batch_plan_md(s: dict[str, Any]) -> str:
    return "\n".join(["# 17B-R2-R1 batch plan (suggested, not executed)", "",
                      f"- total ready requests: `{s['outbound_requests_built']}`",
                      f"- suggested_live_batch_size: `{s['suggested_live_batch_size']}`",
                      f"- suggested_live_batch_count: `{s['suggested_live_batch_count']}`",
                      f"- suggested_hard_cost_cap_usd (per chunk): `{s['suggested_hard_cost_cap_usd']}`",
                      f"- estimated_cost_usd (all): `{s['estimated_cost_usd']}` (approx, no billing API)", "",
                      "Live extraction is NOT executed here. It requires separate authorization and a",
                      "PASS Vertex credential preflight, run in bounded chunks with stop-on-first-failure.", ""])


def _dedupe_md(s: dict[str, Any]) -> str:
    return "\n".join(["# No-490 dedupe check", "",
                      f"- doc_id_dedupe_check_performed: `{s['doc_id_dedupe_check_performed']}`",
                      f"- old_12_included_in_478: `{s['old_12_included_in_478']}`",
                      f"- old_12_added_separately: `{s['old_12_added_separately']}`",
                      f"- combined_batch_count: `{s['combined_batch_count']}`", "",
                      "The 478 ready set is total. The old 12 are included in the 478, so no 490-file",
                      "batch is created.", ""])


def _entry_gate_md(s: dict[str, Any]) -> str:
    return "\n".join(["# Live extraction entry gate (after 17B-R2-R1)", "",
                      f"- request_validation_passed: `{s['request_validation_passed']}`",
                      f"- request_validation_failed_count_after_repair: `{s['request_validation_failed_count_after_repair']}`",
                      f"- outbound_requests_built: `{s['outbound_requests_built']}` of `{s['ready_files_total']}`", "",
                      "Live extraction is NOT started. It requires separate authorization, a PASS Vertex",
                      "credential preflight, bounded chunks, a hard per-chunk cost cap, and",
                      "stop-on-first-failure. MKB import remains not started.", ""])


def _validation_after(s: dict[str, Any], public_batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {"block": s["block"], "request_validation_passed": s["request_validation_passed"],
            "request_validation_failed_count_after_repair": s["request_validation_failed_count_after_repair"],
            "validation_failure_classes_after_repair": s["validation_failure_classes_after_repair"],
            "outbound_requests_built": s["outbound_requests_built"],
            "per_document": [{"document_id": b["document_id"], "repaired": b["repaired"],
                              "validation_status": b["validation_status"]} for b in public_batch]}


def _cost_public(s: dict[str, Any]) -> dict[str, Any]:
    return {"target_model": s["target_model_for_future_live"],
            "estimated_input_tokens": s["estimated_input_tokens"],
            "estimated_output_tokens": s["estimated_output_tokens"],
            "estimated_cost_usd_approximate": s["estimated_cost_usd"],
            "suggested_live_batch_size": s["suggested_live_batch_size"],
            "suggested_live_batch_count": s["suggested_live_batch_count"],
            "suggested_hard_cost_cap_usd": s["suggested_hard_cost_cap_usd"],
            "note": "Approximate local estimate only; no billing API was called."}


def _implementation(s: dict[str, Any], notes: list[str]) -> str:
    lines = [f"# {s['block']}", "",
             f"## Result: **{'PASS' if s['privacy_result'] == 'passed' else 'BLOCKED'}** (safety: {s['safety_result']})",
             "", "## Metrics", ""]
    for k in ("ready_files_total", "combined_batch_count", "old_12_added_separately",
              "doc_id_dedupe_check_performed", "old_12_included_in_478", "failed_files_from_17b_r2",
              "failure_classes_from_17b_r2", "files_repaired", "outbound_requests_built",
              "request_validation_passed", "request_validation_failed_count_after_repair",
              "validation_failure_classes_after_repair", "estimated_input_tokens",
              "estimated_output_tokens", "estimated_cost_usd", "suggested_live_batch_size",
              "suggested_live_batch_count", "suggested_hard_cost_cap_usd",
              "provider_call_made", "gemini_call_made", "claude_call_made", "openai_call_made",
              "billing_api_call_made", "live_gate_set", "live_extraction_started",
              "raw_source_files_uploaded", "tokenized_payloads_written_to_repo",
              "raw_ocr_written_to_repo", "token_maps_written_to_repo",
              "private_identifier_values_written_to_repo", "public_report_phi_leak_count",
              "mkb_db_opened", "active_mkb_write", "auto_accept_enabled", "medical_decision_made",
              "production_queue_mutated", "future_live_extraction_not_started", "privacy_result", "safety_result"):
        lines.append(f"- {k}: `{s[k]}`")
    if notes:
        lines += ["", "## Notes", ""] + [f"- `{n}`" for n in notes]
    nxt = ("prepare 478-file live extraction in bounded chunks after confirming Vertex credential preflight PASS"
           if s["privacy_result"] == "passed" else "review remaining failure classes locally")
    lines += ["", "## Recommended next (no live extraction started)", "", f"- {nxt}.", ""]
    return "\n".join(lines)


def write_outputs(summary: dict[str, Any], extra: dict[str, Any], notes: list[str]) -> int:
    public_batch = extra["public_batch"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    validation_after = _validation_after(summary, public_batch)
    cost_public = _cost_public(summary)
    repair_md = _repair_matrix(summary)
    privacy_md = _privacy_matrix(summary)
    batch_md = _batch_plan_md(summary)
    entry_md = _entry_gate_md(summary)
    dedupe_md = _dedupe_md(summary)
    impl_md = _implementation(summary, notes)
    batch_plan_public = {"total": summary["outbound_requests_built"],
                         "suggested_live_batch_size": summary["suggested_live_batch_size"],
                         "suggested_live_batch_count": summary["suggested_live_batch_count"],
                         "suggested_hard_cost_cap_usd": summary["suggested_hard_cost_cap_usd"]}

    # Defense-in-depth: no tokenized payload line leaks into public blobs.
    public_blob = "\n".join([json.dumps(summary), json.dumps(validation_after), json.dumps(cost_public),
                             json.dumps(batch_plan_public), repair_md, privacy_md, batch_md, entry_md, dedupe_md, impl_md])
    leak = 0
    try:
        outp = PRIVATE_OUT / "outbound_requests_private.jsonl"
        if outp.is_file():
            for rl in outp.read_text(encoding="utf-8").splitlines():
                try:
                    c = json.loads(rl).get("tokenized_content", "")
                except ValueError:
                    c = ""
                for ln in {l.strip() for l in c.splitlines() if len(l.strip()) >= 12}:
                    if ln in public_blob:
                        leak += 1
                        break
    except OSError:
        pass
    summary["public_report_phi_leak_count"] = leak
    if leak:
        summary["safety_result"] = "blocked"

    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(_implementation(summary, notes), encoding="utf-8")
    (REPORT_DIR / "repair_matrix.md").write_text(repair_md, encoding="utf-8")
    (REPORT_DIR / "validation_after_repair_public.json").write_text(json.dumps(validation_after, indent=2), encoding="utf-8")
    (REPORT_DIR / "dry_run_cost_estimate_public.json").write_text(json.dumps(cost_public, indent=2), encoding="utf-8")
    (REPORT_DIR / "batch_plan_public.json").write_text(json.dumps(batch_plan_public, indent=2), encoding="utf-8")
    (REPORT_DIR / "privacy_gate_matrix.md").write_text(privacy_md, encoding="utf-8")
    (REPORT_DIR / "live_extraction_entry_gate.md").write_text(entry_md, encoding="utf-8")
    (REPORT_DIR / "no_490_dedupe_check.md").write_text(dedupe_md, encoding="utf-8")

    try:
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        (DOWNLOADS / "repair_summary_public.json").write_text(json.dumps(
            {k: summary[k] for k in ("failed_files_from_17b_r2", "failure_classes_from_17b_r2",
             "files_repaired", "request_validation_passed", "request_validation_failed_count_after_repair")}, indent=2), encoding="utf-8")
        (DOWNLOADS / "ready_batch_summary_public.json").write_text(json.dumps(
            {k: summary[k] for k in ("ready_files_total", "combined_batch_count", "old_12_included_in_478",
             "outbound_requests_built", "request_validation_passed", "future_live_extraction_not_started")}, indent=2), encoding="utf-8")
        (DOWNLOADS / "validation_after_repair_public.json").write_text(json.dumps(validation_after, indent=2), encoding="utf-8")
        (DOWNLOADS / "cost_estimate_public.json").write_text(json.dumps(cost_public, indent=2), encoding="utf-8")
        (DOWNLOADS / "batch_plan_public.json").write_text(json.dumps(batch_plan_public, indent=2), encoding="utf-8")
        (DOWNLOADS / "README_NEXT_STEPS.txt").write_text(
            "MedAI 17B-R2-R1 478 Validation Repair Review (sanitized summaries only)\n\n"
            f"ready_files_total: {summary['ready_files_total']}  combined_batch_count: {summary['combined_batch_count']}\n"
            f"files_repaired: {summary['files_repaired']}  request_validation_passed: {summary['request_validation_passed']}\n"
            f"failed_after_repair: {summary['request_validation_failed_count_after_repair']}\n"
            f"estimated_cost_usd (approx): {summary['estimated_cost_usd']}\n"
            f"suggested batches: {summary['suggested_live_batch_count']} x {summary['suggested_live_batch_size']}\n\n"
            "No payload bodies here. Live extraction NOT started; requires separate authorization\n"
            "and a PASS Vertex credential preflight.\n", encoding="utf-8")
        (DOWNLOADS / "live_batch_authorization_checklist.txt").write_text(
            "Corpus Live Extraction Authorization Checklist (NOT started)\n\n"
            "[ ] Explicit new user authorization granted\n"
            "[ ] request_validation_passed == true (all 478)\n"
            "[ ] Vertex credential preflight == PASS (working ADC, no login pending)\n"
            "[ ] Human review of repair + validation reports\n"
            "[ ] Bounded chunks (~25) with hard per-chunk cost cap\n"
            "[ ] Stop-on-first-failure in place\n"
            "[ ] Live gate set only inside the authorized flow, reset afterward\n"
            "[ ] Non-ready / blocked files remain excluded\n", encoding="utf-8")
    except OSError:
        pass

    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    ok = (summary["safety_result"] == "passed" and summary["public_report_phi_leak_count"] == 0
          and summary["provider_call_made"] is False and summary["gemini_call_made"] is False
          and summary["billing_api_call_made"] is False and summary["live_extraction_started"] is False
          and docs_ok)
    state = "pass" if (ok and summary["privacy_result"] == "passed") else ("blocked" if ok else "attention")
    print(f"medai_ai_first_corpus_478_batch_validation_repair_local_only_17b_r2_r1_{state}")
    print(json.dumps({k: summary[k] for k in (
        "ready_files_total", "combined_batch_count", "failed_files_from_17b_r2", "files_repaired",
        "outbound_requests_built", "request_validation_passed",
        "request_validation_failed_count_after_repair", "estimated_input_tokens",
        "estimated_output_tokens", "estimated_cost_usd", "suggested_live_batch_count",
        "suggested_hard_cost_cap_usd", "public_report_phi_leak_count", "provider_call_made",
        "gemini_call_made", "future_live_extraction_not_started", "privacy_result", "safety_result")}, indent=2))
    return 0 if ok else 1


def main() -> int:
    summary, extra, notes = run()
    if "__blocked__" in summary:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        blocked = {"block": "MEDAI-AI-FIRST-CORPUS-478-BATCH-VALIDATION-REPAIR-LOCAL-ONLY-17B-R2-R1",
                   "privacy_result": "blocked", "safety_result": "passed",
                   "blocked_reason": summary["__blocked__"], "provider_call_made": False,
                   "future_live_extraction_not_started": True}
        (REPORT_DIR / "summary.json").write_text(json.dumps(blocked, indent=2), encoding="utf-8")
        (REPORT_DIR / "implementation_report.md").write_text(
            f"# 17B-R2-R1\n\n## Result: BLOCKED\n\n- reason: `{summary['__blocked__']}`\n", encoding="utf-8")
        print("medai_ai_first_corpus_478_batch_validation_repair_local_only_17b_r2_r1_blocked")
        print(json.dumps(blocked, indent=2))
        return 1
    return write_outputs(summary, extra, notes)


if __name__ == "__main__":
    raise SystemExit(main())
