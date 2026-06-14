#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-17B-PHONE-PATTERN-REPAIR-LOCAL-ONLY-17B-R1.

Local-only repair of the 4 tokenized payloads that failed 17B dry-run validation due
to residual phone-like / 10-digit-style numeric sequences. Repairs those payloads by
tokenizing the flagged sequences, rebuilds the 12-request private outbound batch, and
re-validates with the unchanged 17B rules.

NO provider call, NO network, NO billing, NO live gate. Private repaired payloads and
outbound requests are written outside the repo. Public reports carry counts, hashed
document IDs, token counts, estimates, and validation status only — never payload
bodies, token maps, PI values, or the flagged numeric values.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

PRIVATE_17B = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17B_dry_run"))
VALIDATOR_17B = PRIVATE_17B / "dry_run_validator_private.json"
PRIVATE_CORPUS = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\corpus_tokenized_17A"))
CORPUS_MANIFEST = PRIVATE_CORPUS / "corpus_manifest_private.jsonl"
CORPUS_DOCS = PRIVATE_CORPUS / "documents"

PRIVATE_OUT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17B_R1_repaired"))
PRIVATE_OUT_LABEL = r"C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17B_R1_repaired\outbound_requests_private.jsonl"
DOWNLOADS = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_17B_R1_Phone_Pattern_Repair_Review"))
DOWNLOADS_LABEL = r"%USERPROFILE%\Downloads\MedAI_17B_R1_Phone_Pattern_Repair_Review"

SCHEMA_PATH = REPO_ROOT / "config" / "medai_ai_extraction_schema_17b.json"
PROMPT_CONTRACT_PATH = REPO_ROOT / "config" / "medai_ai_extraction_prompt_contract_17b.md"
READINESS_PUBLIC = REPO_ROOT / "reports" / "medai_ai_first_corpus_pi_tokenization_local_only_17a" / "ai_extraction_readiness_public.json"

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17b_phone_pattern_repair_local_only_17b_r1"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
REPAIR_MATRIX_MD = REPORT_DIR / "repair_matrix.md"
VALIDATION_AFTER_REPAIR_JSON = REPORT_DIR / "validation_after_repair_public.json"
DRY_RUN_COST_PUBLIC_JSON = REPORT_DIR / "dry_run_cost_estimate_public.json"
LIVE_17C_ENTRY_GATE_MD = REPORT_DIR / "live_17c_entry_gate.md"

DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17B_PHONE_PATTERN_REPAIR_LOCAL_ONLY_17B_R1"
REQUIRED_DOCS = (
    "MEDAI_AI_FIRST_CORPUS_17B_PHONE_PATTERN_REPAIR_LOCAL_ONLY_17B_R1.md",
    "MEDAI_PHONE_PATTERN_REPAIR_POLICY_17B_R1.md",
    "MEDAI_17C_ENTRY_CRITERIA_AFTER_17B_R1.md",
    "MEDAI_REMAINING_BLOCKED_CORPUS_FOLLOWUP_17A_R2.md",
)

DEDICATED_GATE_NAME = "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED"
ACTIVE_VALUES = ("1", "true", "TRUE", "yes", "YES", "enabled", "ENABLED", "True")

TARGET_MODEL = "gemini-2.5-flash-lite"
INPUT_USD_PER_M = 0.075
OUTPUT_USD_PER_M = 0.30
OUTPUT_TOKENS_PER_DOC = 800

TOKEN_RE = re.compile(r"\[[A-Z_]+_\d+\]")
PHONE_RE = re.compile(r"(?:\+?\d{1,2}[ \-.]?)?\(?\d{3}\)?[ \-.]?\d{3}[ \-.]?\d{4}\b")
# Same 17B validation patterns (unchanged — validator is NOT weakened).
_PI_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "phone": PHONE_RE,
    "mrn": re.compile(r"(?i)\b(?:MRN|MR#|Medical Record(?: Number)?)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "insurance_account": re.compile(r"(?i)\b(?:Insurance|Policy|Member|Account|Acct)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "accession_specimen": re.compile(r"(?i)\b(?:Accession|Specimen|Order)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "local_path": re.compile(r"(?:[A-Za-z]:\\[^\s]+|/(?:home|Users)/[^\s]+)"),
}
_NUMERIC_ID_CONTEXT = re.compile(r"(?i)(barcode|specimen|accession|order|lab\s*id|requisition|sample)")

FAMILY_BY_EXT = {
    ".jpg": ("image", "scanned_image"), ".jpeg": ("image", "scanned_image"),
    ".tif": ("image", "scanned_image"), ".tiff": ("image", "scanned_image"),
    ".png": ("image", "scanned_image"), ".rtf": ("document", "rich_text"),
    ".docx": ("document", "word_document"), ".pdf": ("document", "pdf_document"),
    ".txt": ("document", "plain_text"),
}


def _gate_active() -> bool:
    v = os.environ.get(DEDICATED_GATE_NAME)
    return v is not None and v.strip() in ACTIVE_VALUES


def _sha16(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _validate_no_raw_pi(text: str) -> list[str]:
    residual = TOKEN_RE.sub(" ", text)
    return [name for name, rx in _PI_PATTERNS.items() if rx.search(residual)]


def _repair_phone_patterns(text: str) -> tuple[str, int, int]:
    """Tokenize flagged phone-like sequences to a fixed point so that BOTH the raw
    repaired text and the validator's token-stripped view are free of phone matches.

    Context near a match decides the token class ([NUMERIC_IDENTIFIER_n] when a
    barcode/specimen/order/accession cue is nearby, else [PHONE_PATTERN_n]).
    Returns (repaired_text, phone_token_count, numeric_id_token_count).
    """
    phone_n = 0
    numid_n = 0

    def _tok(is_numid: bool) -> str:
        nonlocal phone_n, numid_n
        if is_numid:
            numid_n += 1
            return f"[NUMERIC_IDENTIFIER_{numid_n}]"
        phone_n += 1
        return f"[PHONE_PATTERN_{phone_n}]"

    guard = 0
    while guard < 20000:
        guard += 1
        # Phase A: a phone match directly in the raw text.
        m = PHONE_RE.search(text)
        if m:
            pre = text[max(0, m.start() - 24):m.start()]
            text = text[:m.start()] + _tok(bool(_NUMERIC_ID_CONTEXT.search(pre))) + text[m.end():]
            continue
        # Phase B: a phone match that appears only after token-stripping (digit groups
        # bridged by a removed token). Tokenize a digit run from that bridged match.
        stripped = TOKEN_RE.sub(" ", text)
        ms = PHONE_RE.search(stripped)
        if not ms:
            break
        replaced = False
        for dm in re.finditer(r"\d{3,}", ms.group(0)):
            digits = dm.group(0)
            idx = text.find(digits)
            if idx != -1:
                pre = text[max(0, idx - 24):idx]
                text = text[:idx] + _tok(bool(_NUMERIC_ID_CONTEXT.search(pre))) + text[idx + len(digits):]
                replaced = True
                break
        if not replaced:
            break
    return text, phone_n, numid_n


def _load_ready_recs() -> list[dict[str, str]]:
    recs = []
    if CORPUS_MANIFEST.is_file():
        for line in CORPUS_MANIFEST.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "ready_for_ai_extraction":
                recs.append(r)
    return recs


def _load_failed_ids() -> tuple[list[str], set[str]]:
    """Return (failed_ids, all_failure_classes)."""
    failed_ids = []
    classes: set[str] = set()
    d = json.loads(VALIDATOR_17B.read_text(encoding="utf-8"))
    for r in d.get("per_doc", []):
        if r.get("validation_status") != "ok":
            failed_ids.append(r["document_id"])
            for c in r.get("pi_pattern_classes_detected", []):
                classes.add(c)
    return failed_ids, classes


def run() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    notes: list[str] = []
    if _gate_active():
        raise SystemExit("ABORT: live gate is active; refusing to run 17B-R1 repair.")

    readiness = json.loads(READINESS_PUBLIC.read_text(encoding="utf-8"))
    ready_from_17a = int(readiness.get("total_files_ready_for_ai_extraction", 0))
    blocked_excluded = int(readiness.get("total_files_blocked_for_ai_extraction", 0))

    failed_ids, failure_classes = _load_failed_ids()
    # Safety: refuse if any non-phone failure class is present (not handled here).
    non_phone = failure_classes - {"phone"}
    if non_phone:
        raise SystemExit(f"ABORT: unhandled failure classes present: {sorted(non_phone)}")
    failed_reason = "phone_pattern"

    prompt_contract = PROMPT_CONTRACT_PATH.read_text(encoding="utf-8")
    prompt_tokens = _approx_tokens(prompt_contract)
    ready_recs = _load_ready_recs()

    public_batch: list[dict[str, Any]] = []
    private_requests: list[dict[str, Any]] = []
    repair_records: list[dict[str, Any]] = []
    est_input = 0
    est_output = 0
    residual_phone_failures = 0
    files_repaired = 0

    for rec in ready_recs:
        doc_id = rec.get("document_id", "")
        ext = (rec.get("extension") or "").lower()
        tok_path = CORPUS_DOCS / doc_id / "tokenized_text.txt"
        if not tok_path.is_file():
            notes.append(f"missing_tokenized_text:{doc_id}")
            continue
        text = tok_path.read_text(encoding="utf-8", errors="ignore")

        repaired = doc_id in failed_ids
        phone_tok = numid_tok = 0
        if repaired:
            text, phone_tok, numid_tok = _repair_phone_patterns(text)
            files_repaired += 1
            repair_records.append({
                "document_id": doc_id,
                "phone_pattern_tokens_added": phone_tok,
                "numeric_identifier_tokens_added": numid_tok,
            })

        pi_hits = _validate_no_raw_pi(text)
        if "phone" in pi_hits:
            residual_phone_failures += 1
        valid = not pi_hits

        family, dtype = FAMILY_BY_EXT.get(ext, ("unknown", "unknown"))
        src_hash = _sha16(text)
        doc_in = _approx_tokens(text) + prompt_tokens

        public_batch.append({
            "document_id": doc_id,
            "source_hash": src_hash,
            "document_family": family,
            "document_type": dtype,
            "repaired": repaired,
            "estimated_input_tokens": doc_in,
            "validation_status": "ok" if valid else "blocked",
            "pi_pattern_classes_detected": pi_hits,  # class names only, never values
        })
        if valid:
            est_input += doc_in
            est_output += OUTPUT_TOKENS_PER_DOC
            private_requests.append({
                "document_id": doc_id, "source_hash": src_hash, "model": TARGET_MODEL,
                "schema_ref": "config/medai_ai_extraction_schema_17b.json",
                "prompt_contract_ref": "config/medai_ai_extraction_prompt_contract_17b.md",
                "document_family": family, "document_type": dtype,
                "tokenized_content": text,  # PRIVATE only (repaired where applicable)
            })

    outbound_built = len(private_requests)
    request_validation_passed = (outbound_built == len(ready_recs)) and (residual_phone_failures == 0) and (len(ready_recs) > 0)
    est_cost = round(est_input / 1_000_000 * INPUT_USD_PER_M + est_output / 1_000_000 * OUTPUT_USD_PER_M, 6)
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
            json.dumps({"residual_phone_pattern_failures_after_repair": residual_phone_failures,
                        "per_doc": [{"document_id": b["document_id"], "repaired": b["repaired"],
                                     "validation_status": b["validation_status"],
                                     "pi_pattern_classes_detected": b["pi_pattern_classes_detected"]}
                                    for b in public_batch]}, ensure_ascii=False, indent=2), encoding="utf-8")
        (PRIVATE_OUT / "cost_estimate_private.json").write_text(
            json.dumps({"target_model": TARGET_MODEL, "estimated_input_tokens": est_input,
                        "estimated_output_tokens": est_output, "estimated_cost_usd": est_cost},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        private_written = True
    except OSError:
        notes.append("private_artifact_write_failed")

    summary = {
        "block": "MEDAI-AI-FIRST-CORPUS-17B-PHONE-PATTERN-REPAIR-LOCAL-ONLY-17B-R1",
        "local_only": True,
        "dry_run_only": True,
        "provider_call_made": False,
        "vertex_live_execution": False,
        "gemini_call_made": False,
        "claude_call_made": False,
        "openai_call_made": False,
        "billing_api_call_made": False,
        "future_live_gate_set": False,
        "future_live_gate_environment_active": _gate_active(),
        "seventeen_c_live_started": False,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "production_queue_mutated": False,
        "ready_files_from_17a": ready_from_17a,
        "failed_files_from_17b": len(failed_ids),
        "failed_reason": failed_reason,
        "files_repaired": files_repaired,
        "outbound_requests_built": outbound_built,
        "request_validation_passed": request_validation_passed,
        "residual_phone_pattern_failures_after_repair": residual_phone_failures,
        "private_outbound_requests_path": PRIVATE_OUT_LABEL,
        "private_payloads_written_outside_repo": private_written,
        "downloads_review_location": DOWNLOADS_LABEL,
        "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False,
        "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False,
        "public_report_phi_leak_count": 0,
        "estimated_input_tokens": est_input,
        "estimated_output_tokens": est_output,
        "estimated_cost_usd": est_cost,
        "target_model_for_future_live": TARGET_MODEL,
        "blocked_files_excluded": blocked_excluded,
        "future_17c_live_not_started": True,
        "privacy_result": privacy_result,
        "safety_result": "passed",
    }
    extra = {"public_batch": public_batch, "repair_records": repair_records}
    return summary, extra, notes


def _repair_matrix(summary: dict[str, Any], public_batch: list[dict[str, Any]]) -> str:
    rows = [
        f"| {b['document_id']} | `{b['repaired']}` | {b['estimated_input_tokens']} | "
        f"`{b['validation_status']}` |"
        for b in public_batch
    ]
    keys = ["failed_files_from_17b", "failed_reason", "files_repaired", "outbound_requests_built",
            "request_validation_passed", "residual_phone_pattern_failures_after_repair",
            "estimated_input_tokens", "estimated_output_tokens", "estimated_cost_usd",
            "public_report_phi_leak_count", "privacy_result", "safety_result"]
    return "\n".join(
        ["# 17B-R1 repair matrix (counts/booleans only)", "",
         "| Document (hashed) | Repaired | Est input tokens | Validation |",
         "| --- | --- | --- | --- |", *rows, "",
         "| Field | Value |", "| --- | --- |", *[f"| {k} | `{summary[k]}` |" for k in keys], "",
         "Counts/booleans only — no payload bodies, no token maps, no flagged numeric values.",
         "Repaired payloads and outbound requests remain private, outside git.", ""]
    )


def _live_gate_md(summary: dict[str, Any]) -> str:
    return "\n".join(
        ["# 17C live entry gate (after 17B-R1)", "",
         f"- request_validation_passed: `{summary['request_validation_passed']}`",
         f"- residual_phone_pattern_failures_after_repair: `{summary['residual_phone_pattern_failures_after_repair']}`",
         f"- outbound_requests_built: `{summary['outbound_requests_built']}` of "
         f"`{summary['ready_files_from_17a']}` ready",
         f"- estimated_cost_usd (approx): `{summary['estimated_cost_usd']}`",
         f"- target_model_for_future_live: `{summary['target_model_for_future_live']}`", "",
         "17C is NOT started. A future small live batch requires explicit new authorization,",
         "a confirmed hard cost cap, a bounded batch size, stop-on-first-failure, and the",
         "dedicated live gate set only inside the authorized 17C flow. The 587 blocked files",
         "remain excluded until fixed.", ""]
    )


def _implementation(summary: dict[str, Any], notes: list[str]) -> str:
    lines = [
        "# MEDAI-AI-FIRST-CORPUS-17B-PHONE-PATTERN-REPAIR-LOCAL-ONLY-17B-R1",
        "",
        f"## Result: **{'PASS' if summary['privacy_result'] == 'passed' else 'BLOCKED'}** "
        f"(safety: {summary['safety_result']})",
        "",
        "## Result",
        "",
        "- Repaired the 4 phone-pattern failures by tokenizing flagged numeric sequences,",
        "  then rebuilt and re-validated the 12-request private outbound batch.",
        "- Validator rules unchanged; payloads repaired first. No provider call, no network,",
        "  no billing, no live gate activation.",
        "",
        "## Metrics",
        "",
    ]
    for key in (
        "ready_files_from_17a", "failed_files_from_17b", "failed_reason", "files_repaired",
        "outbound_requests_built", "request_validation_passed",
        "residual_phone_pattern_failures_after_repair", "estimated_input_tokens",
        "estimated_output_tokens", "estimated_cost_usd", "target_model_for_future_live",
        "private_payloads_written_outside_repo", "tokenized_payloads_written_to_repo",
        "raw_ocr_written_to_repo", "token_maps_written_to_repo",
        "private_identifier_values_written_to_repo", "public_report_phi_leak_count",
        "provider_call_made", "gemini_call_made", "claude_call_made", "openai_call_made",
        "billing_api_call_made", "future_live_gate_environment_active",
        "seventeen_c_live_started", "mkb_db_opened", "active_mkb_write", "auto_accept_enabled",
        "medical_decision_made", "production_queue_mutated", "blocked_files_excluded",
        "future_17c_live_not_started", "privacy_result", "safety_result",
    ):
        lines.append(f"- {key}: `{summary[key]}`")
    if notes:
        lines += ["", "## Notes", ""] + [f"- `{n}`" for n in notes]
    nxt = ("prepare 17C small live batch authorization for the 12 validated requests"
           if summary["privacy_result"] == "passed"
           else "review remaining validator failures locally")
    lines += ["", "## Recommended next (NO live batch started automatically)", "",
              f"- {nxt}. 17C requires explicit new authorization and is not started.", ""]
    return "\n".join(lines)


def write_outputs(summary: dict[str, Any], extra: dict[str, Any], notes: list[str]) -> int:
    public_batch = extra["public_batch"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    validation_after = {
        "block": summary["block"],
        "request_validation_passed": summary["request_validation_passed"],
        "residual_phone_pattern_failures_after_repair": summary["residual_phone_pattern_failures_after_repair"],
        "outbound_requests_built": summary["outbound_requests_built"],
        "per_document": [
            {"document_id": b["document_id"], "repaired": b["repaired"],
             "validation_status": b["validation_status"],
             "pi_pattern_classes_detected": b["pi_pattern_classes_detected"]}
            for b in public_batch
        ],
    }
    cost_public = {
        "target_model": summary["target_model_for_future_live"],
        "estimated_input_tokens": summary["estimated_input_tokens"],
        "estimated_output_tokens": summary["estimated_output_tokens"],
        "estimated_cost_usd_approximate": summary["estimated_cost_usd"],
        "input_usd_per_million_tokens": INPUT_USD_PER_M,
        "output_usd_per_million_tokens": OUTPUT_USD_PER_M,
        "note": "Approximate local estimate only; no billing API was called.",
    }
    matrix_md = _repair_matrix(summary, public_batch)
    gate_md = _live_gate_md(summary)
    impl_md = _implementation(summary, notes)

    # Defense-in-depth: no repaired payload line leaks into public blobs.
    public_blob = "\n".join([json.dumps(summary), json.dumps(validation_after),
                             json.dumps(cost_public), matrix_md, gate_md, impl_md])
    leak = 0
    try:
        outp = PRIVATE_OUT / "outbound_requests_private.jsonl"
        if outp.is_file():
            for rl in outp.read_text(encoding="utf-8").splitlines():
                try:
                    content = json.loads(rl).get("tokenized_content", "")
                except ValueError:
                    content = ""
                for ln in {l.strip() for l in content.splitlines() if len(l.strip()) >= 8}:
                    if ln in public_blob:
                        leak += 1
    except OSError:
        pass
    summary["public_report_phi_leak_count"] = leak
    if leak:
        summary["safety_result"] = "failed"

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VALIDATION_AFTER_REPAIR_JSON.write_text(json.dumps(validation_after, indent=2), encoding="utf-8")
    DRY_RUN_COST_PUBLIC_JSON.write_text(json.dumps(cost_public, indent=2), encoding="utf-8")
    REPAIR_MATRIX_MD.write_text(matrix_md, encoding="utf-8")
    LIVE_17C_ENTRY_GATE_MD.write_text(gate_md, encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation(summary, notes), encoding="utf-8")

    # ---- Downloads review copies (sanitized only) ----
    try:
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        (DOWNLOADS / "repair_summary_public.json").write_text(
            json.dumps({k: summary[k] for k in ("failed_files_from_17b", "failed_reason",
                        "files_repaired", "request_validation_passed",
                        "residual_phone_pattern_failures_after_repair")}, indent=2), encoding="utf-8")
        (DOWNLOADS / "ready_batch_summary_public.json").write_text(
            json.dumps({k: summary[k] for k in ("ready_files_from_17a", "blocked_files_excluded",
                        "outbound_requests_built", "request_validation_passed",
                        "future_17c_live_not_started")}, indent=2), encoding="utf-8")
        (DOWNLOADS / "cost_estimate_public.json").write_text(json.dumps(cost_public, indent=2), encoding="utf-8")
        (DOWNLOADS / "README_NEXT_STEPS.txt").write_text(
            "MedAI 17B-R1 Phone-Pattern Repair Review\n\n"
            "Sanitized summaries only. No payload bodies, no token maps, no PI values, no\n"
            "flagged numeric values are included.\n\n"
            f"Failed files repaired: {summary['files_repaired']} (reason: {summary['failed_reason']})\n"
            f"Request validation passed: {summary['request_validation_passed']}\n"
            f"Residual phone-pattern failures after repair: {summary['residual_phone_pattern_failures_after_repair']}\n"
            f"Outbound requests built (private): {summary['outbound_requests_built']}\n\n"
            "17C live batch is NOT started and requires explicit new authorization.\n",
            encoding="utf-8")
        (DOWNLOADS / "live_17C_authorization_checklist.txt").write_text(
            "17C Small Live Batch Authorization Checklist (NOT started)\n\n"
            "[ ] Explicit new user authorization granted\n"
            "[ ] request_validation_passed == true (all 12)\n"
            "[ ] residual_phone_pattern_failures_after_repair == 0\n"
            "[ ] Human review of repair + dry-run outputs complete\n"
            "[ ] Hard cost cap confirmed\n"
            "[ ] Bounded batch size + per-call limit set\n"
            "[ ] Stop-on-first-failure in place\n"
            "[ ] Dedicated live gate set only inside the authorized 17C flow, reset afterward\n"
            "[ ] 587 blocked files remain excluded\n",
            encoding="utf-8")
    except OSError:
        notes.append("downloads_copy_failed")

    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    ok = (
        summary["safety_result"] == "passed"
        and summary["public_report_phi_leak_count"] == 0
        and summary["provider_call_made"] is False
        and summary["gemini_call_made"] is False
        and summary["billing_api_call_made"] is False
        and summary["future_live_gate_environment_active"] is False
        and summary["seventeen_c_live_started"] is False
        and summary["future_17c_live_not_started"] is True
        and docs_ok
    )
    state = "pass" if (ok and summary["privacy_result"] == "passed") else ("blocked" if ok else "attention")
    print(f"medai_ai_first_corpus_17b_phone_pattern_repair_local_only_17b_r1_{state}")
    print(json.dumps({k: summary[k] for k in (
        "failed_files_from_17b", "failed_reason", "files_repaired", "outbound_requests_built",
        "request_validation_passed", "residual_phone_pattern_failures_after_repair",
        "estimated_input_tokens", "estimated_output_tokens", "estimated_cost_usd",
        "public_report_phi_leak_count", "provider_call_made", "gemini_call_made",
        "billing_api_call_made", "future_live_gate_environment_active",
        "seventeen_c_live_started", "future_17c_live_not_started", "privacy_result",
        "safety_result")}, indent=2))
    return 0 if ok else 1


def main() -> int:
    summary, extra, notes = run()
    return write_outputs(summary, extra, notes)


if __name__ == "__main__":
    raise SystemExit(main())
