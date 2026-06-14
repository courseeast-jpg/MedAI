#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-EXTRACTION-SCHEMA-BATCH-DRY-RUN-17B.

Dry-run preparation of AI-first extraction for the 12 tokenized corpus files that
passed 17A. Builds private outbound request JSONL, validates each request for residual
raw PI, and estimates tokens/cost locally. NO provider call, NO network, NO billing,
NO live gate.

Private outbound artifacts are written outside the repo. Public repo reports contain
only counts, schema names, hashed document IDs, token counts, estimated tokens/cost,
and validation status — never tokenized payloads, raw OCR, token maps, or PI values.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

READINESS_PUBLIC = REPO_ROOT / "reports" / "medai_ai_first_corpus_pi_tokenization_local_only_17a" / "ai_extraction_readiness_public.json"
PRIVATE_CORPUS = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\corpus_tokenized_17A"))
CORPUS_MANIFEST = PRIVATE_CORPUS / "corpus_manifest_private.jsonl"
CORPUS_DOCS = PRIVATE_CORPUS / "documents"

PRIVATE_OUT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17B_dry_run"))
PRIVATE_OUT_LABEL = r"C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17B_dry_run\outbound_requests_private.jsonl"
DOWNLOADS = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_17B_AI_Extraction_Dry_Run_Review"))
DOWNLOADS_LABEL = r"%USERPROFILE%\Downloads\MedAI_17B_AI_Extraction_Dry_Run_Review"

SCHEMA_PATH = REPO_ROOT / "config" / "medai_ai_extraction_schema_17b.json"
PROMPT_CONTRACT_PATH = REPO_ROOT / "config" / "medai_ai_extraction_prompt_contract_17b.md"

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_extraction_schema_batch_dry_run_17b"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
READY_BATCH_PUBLIC_JSON = REPORT_DIR / "ready_batch_public.json"
SCHEMA_VALIDATION_PUBLIC_JSON = REPORT_DIR / "schema_validation_public.json"
DRY_RUN_COST_PUBLIC_JSON = REPORT_DIR / "dry_run_cost_estimate_public.json"
PRIVACY_GATE_MATRIX_MD = REPORT_DIR / "privacy_gate_matrix.md"
LIVE_17C_ENTRY_GATE_MD = REPORT_DIR / "live_17c_entry_gate.md"

DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_EXTRACTION_SCHEMA_BATCH_DRY_RUN_17B"
REQUIRED_DOCS = (
    "MEDAI_AI_FIRST_CORPUS_EXTRACTION_SCHEMA_BATCH_DRY_RUN_17B.md",
    "MEDAI_AI_EXTRACTION_JSON_SCHEMA_17B.md",
    "MEDAI_AI_EXTRACTION_PROMPT_CONTRACT_17B.md",
    "MEDAI_BATCH_DRY_RUN_AND_COST_CONTROL_17B.md",
    "MEDAI_17C_SMALL_LIVE_BATCH_ENTRY_CRITERIA.md",
    "MEDAI_BLOCKED_CORPUS_EXTRACTION_UNAVAILABLE_FOLLOWUP_17A_R2.md",
)

DEDICATED_GATE_NAME = "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED"
ACTIVE_VALUES = ("1", "true", "TRUE", "yes", "YES", "enabled", "ENABLED", "True")

TARGET_MODEL = "gemini-2.5-flash-lite"
# Approximate, configurable pricing (USD per 1M tokens). Not from any billing API.
INPUT_USD_PER_M = 0.075
OUTPUT_USD_PER_M = 0.30
OUTPUT_TOKENS_PER_DOC = 800  # rough schema-bounded output assumption

TOKEN_RE = re.compile(r"\[[A-Z_]+_\d+\]")
# Residual raw-PI patterns (scanned over text with token placeholders removed).
_PI_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?:\+?\d{1,2}[ \-.]?)?\(?\d{3}\)?[ \-.]?\d{3}[ \-.]?\d{4}\b"),
    "mrn": re.compile(r"(?i)\b(?:MRN|MR#|Medical Record(?: Number)?)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "insurance_account": re.compile(r"(?i)\b(?:Insurance|Policy|Member|Account|Acct)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "accession_specimen": re.compile(r"(?i)\b(?:Accession|Specimen|Order)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "local_path": re.compile(r"(?:[A-Za-z]:\\[^\s]+|/(?:home|Users)/[^\s]+)"),
}

FAMILY_BY_EXT = {
    ".jpg": ("image", "scanned_image"),
    ".jpeg": ("image", "scanned_image"),
    ".tif": ("image", "scanned_image"),
    ".tiff": ("image", "scanned_image"),
    ".png": ("image", "scanned_image"),
    ".rtf": ("document", "rich_text"),
    ".docx": ("document", "word_document"),
    ".pdf": ("document", "pdf_document"),
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
    hits = []
    for name, rx in _PI_PATTERNS.items():
        if rx.search(residual):
            hits.append(name)
    return hits


def _load_ready_ids() -> list[dict[str, str]]:
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


def run() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    notes: list[str] = []
    if _gate_active():
        raise SystemExit("ABORT: live gate is active; refusing to run 17B dry run.")

    readiness = json.loads(READINESS_PUBLIC.read_text(encoding="utf-8"))
    ready_from_17a = int(readiness.get("total_files_ready_for_ai_extraction", 0))
    blocked_excluded = int(readiness.get("total_files_blocked_for_ai_extraction", 0))

    schema_obj = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    prompt_contract = PROMPT_CONTRACT_PATH.read_text(encoding="utf-8")
    prompt_tokens = _approx_tokens(prompt_contract)

    ready_recs = _load_ready_ids()
    if len(ready_recs) != ready_from_17a:
        notes.append(f"manifest_ready_count_{len(ready_recs)}_vs_readiness_{ready_from_17a}")

    public_batch: list[dict[str, Any]] = []
    private_requests: list[dict[str, Any]] = []
    est_input_tokens = 0
    est_output_tokens = 0
    validation_failures = 0
    all_tokenized_for_scan: list[str] = []

    for rec in ready_recs:
        doc_id = rec.get("document_id", "")
        ext = (rec.get("extension") or "").lower()
        tok_path = CORPUS_DOCS / doc_id / "tokenized_text.txt"
        if not tok_path.is_file():
            notes.append(f"missing_tokenized_text:{doc_id}")
            continue
        text = tok_path.read_text(encoding="utf-8", errors="ignore")
        all_tokenized_for_scan.append(text)
        family, dtype = FAMILY_BY_EXT.get(ext, ("unknown", "unknown"))
        pi_hits = _validate_no_raw_pi(text)
        valid = not pi_hits
        if not valid:
            validation_failures += 1

        doc_in_tokens = _approx_tokens(text) + prompt_tokens
        src_hash = _sha16(text)

        # Public per-doc record: hash id + counts + status only. NO content.
        public_batch.append({
            "document_id": doc_id,
            "source_hash": src_hash,
            "document_family": family,
            "document_type": dtype,
            "estimated_input_tokens": doc_in_tokens,
            "validation_status": "ok" if valid else "blocked",
            "pi_pattern_classes_detected": pi_hits,  # class names only, never values
        })

        if valid:
            est_input_tokens += doc_in_tokens
            est_output_tokens += OUTPUT_TOKENS_PER_DOC
            # Private outbound request — built ONLY for valid docs, written privately.
            private_requests.append({
                "document_id": doc_id,
                "source_hash": src_hash,
                "model": TARGET_MODEL,
                "schema_ref": "config/medai_ai_extraction_schema_17b.json",
                "prompt_contract_ref": "config/medai_ai_extraction_prompt_contract_17b.md",
                "document_family": family,
                "document_type": dtype,
                "tokenized_content": text,  # PRIVATE only
            })

    outbound_built = len(private_requests)
    request_validation_passed = (validation_failures == 0) and (outbound_built == len(ready_recs)) and (len(ready_recs) > 0)
    est_cost = round(est_input_tokens / 1_000_000 * INPUT_USD_PER_M + est_output_tokens / 1_000_000 * OUTPUT_USD_PER_M, 6)

    # ---- Write PRIVATE outbound artifacts (outside repo, never committed) ----
    private_written = False
    try:
        PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
        with (PRIVATE_OUT / "outbound_requests_private.jsonl").open("w", encoding="utf-8") as fh:
            for req in private_requests:
                fh.write(json.dumps(req, ensure_ascii=False) + "\n")
        (PRIVATE_OUT / "prompt_preview_private.txt").write_text(prompt_contract, encoding="utf-8")
        (PRIVATE_OUT / "ready_doc_payload_manifest_private.json").write_text(
            json.dumps([{"document_id": r["document_id"], "source_hash": r["source_hash"],
                         "document_family": r["document_family"], "document_type": r["document_type"]}
                        for r in private_requests], ensure_ascii=False, indent=2), encoding="utf-8")
        (PRIVATE_OUT / "dry_run_validator_private.json").write_text(
            json.dumps({"validation_failures": validation_failures,
                        "per_doc": [{"document_id": b["document_id"], "validation_status": b["validation_status"],
                                     "pi_pattern_classes_detected": b["pi_pattern_classes_detected"]}
                                    for b in public_batch]}, ensure_ascii=False, indent=2), encoding="utf-8")
        (PRIVATE_OUT / "cost_estimate_private.json").write_text(
            json.dumps({"target_model": TARGET_MODEL, "estimated_input_tokens": est_input_tokens,
                        "estimated_output_tokens": est_output_tokens, "estimated_cost_usd": est_cost,
                        "input_usd_per_m": INPUT_USD_PER_M, "output_usd_per_m": OUTPUT_USD_PER_M},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        private_written = True
    except OSError:
        notes.append("private_artifact_write_failed")

    privacy_result = "passed" if request_validation_passed else "blocked"

    summary = {
        "block": "MEDAI-AI-FIRST-CORPUS-EXTRACTION-SCHEMA-BATCH-DRY-RUN-17B",
        "dry_run_only": True,
        "local_only": True,
        "provider_call_made": False,
        "vertex_live_execution": False,
        "gemini_call_made": False,
        "claude_call_made": False,
        "openai_call_made": False,
        "billing_api_call_made": False,
        "future_live_gate_set": False,
        "future_live_gate_environment_active": _gate_active(),
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "production_queue_mutated": False,
        "ready_files_from_17a": ready_from_17a,
        "blocked_files_excluded": blocked_excluded,
        "outbound_requests_built": outbound_built,
        "private_outbound_requests_path": PRIVATE_OUT_LABEL,
        "private_payloads_written_outside_repo": private_written,
        "downloads_review_location": DOWNLOADS_LABEL,
        "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False,
        "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False,
        "public_report_phi_leak_count": 0,
        "schema_created": SCHEMA_PATH.is_file(),
        "prompt_contract_created": PROMPT_CONTRACT_PATH.is_file(),
        "request_validation_passed": request_validation_passed,
        "validation_failures": validation_failures,
        "estimated_input_tokens": est_input_tokens,
        "estimated_output_tokens": est_output_tokens,
        "estimated_cost_usd": est_cost,
        "target_model_for_future_live": TARGET_MODEL,
        "future_17c_live_not_started": True,
        "privacy_result": privacy_result,
        "safety_result": "passed",
    }
    return summary, {"public_batch": public_batch, "schema_obj": schema_obj}, notes


def _privacy_matrix(summary: dict[str, Any]) -> str:
    rows = [
        ("provider_call_made", summary["provider_call_made"]),
        ("gemini_call_made", summary["gemini_call_made"]),
        ("billing_api_call_made", summary["billing_api_call_made"]),
        ("future_live_gate_environment_active", summary["future_live_gate_environment_active"]),
        ("tokenized_payloads_written_to_repo", summary["tokenized_payloads_written_to_repo"]),
        ("raw_ocr_written_to_repo", summary["raw_ocr_written_to_repo"]),
        ("token_maps_written_to_repo", summary["token_maps_written_to_repo"]),
        ("private_identifier_values_written_to_repo", summary["private_identifier_values_written_to_repo"]),
        ("public_report_phi_leak_count", summary["public_report_phi_leak_count"]),
        ("request_validation_passed", summary["request_validation_passed"]),
        ("privacy_result", summary["privacy_result"]),
        ("future_17c_live_not_started", summary["future_17c_live_not_started"]),
    ]
    return "\n".join(
        ["# 17B privacy gate matrix", "", "| Gate | Value |", "| --- | --- |",
         *[f"| {k} | `{v}` |" for k, v in rows], "",
         "Outbound requests are private (outside the repo). Public reports carry counts,",
         "schema names, hashed document IDs, token counts, and estimates only.", ""]
    )


def _live_gate_md(summary: dict[str, Any]) -> str:
    return "\n".join(
        ["# 17C live entry gate", "",
         f"- request_validation_passed: `{summary['request_validation_passed']}`",
         f"- outbound_requests_built: `{summary['outbound_requests_built']}` of "
         f"`{summary['ready_files_from_17a']}` ready",
         f"- estimated_cost_usd (approx): `{summary['estimated_cost_usd']}`",
         f"- target_model_for_future_live: `{summary['target_model_for_future_live']}`",
         "",
         "17C is NOT started. A future small live batch requires explicit new authorization,",
         "a confirmed hard cost cap, a bounded batch size, stop-on-first-failure, and the",
         "dedicated live gate set only inside the authorized 17C flow. The 587 blocked files",
         "remain excluded until fixed.", ""]
    )


def _implementation(summary: dict[str, Any], notes: list[str]) -> str:
    lines = [
        "# MEDAI-AI-FIRST-CORPUS-EXTRACTION-SCHEMA-BATCH-DRY-RUN-17B",
        "",
        f"## Result: **{'PASS' if summary['privacy_result'] == 'passed' else 'BLOCKED'}** "
        f"(safety: {summary['safety_result']})",
        "",
        "## Result",
        "",
        "- Dry-run preparation of AI-first extraction for the 12 ready tokenized files.",
        "- Built private outbound request JSONL (outside the repo) and validated each",
        "  request for residual raw PI. Estimated tokens/cost locally.",
        "- No provider call, no network, no billing, no live gate activation.",
        "",
        "## Metrics",
        "",
    ]
    for key in (
        "dry_run_only", "ready_files_from_17a", "blocked_files_excluded", "outbound_requests_built",
        "request_validation_passed", "validation_failures", "estimated_input_tokens",
        "estimated_output_tokens", "estimated_cost_usd", "target_model_for_future_live",
        "schema_created", "prompt_contract_created", "private_payloads_written_outside_repo",
        "tokenized_payloads_written_to_repo", "raw_ocr_written_to_repo", "token_maps_written_to_repo",
        "private_identifier_values_written_to_repo", "public_report_phi_leak_count",
        "provider_call_made", "gemini_call_made", "claude_call_made", "openai_call_made",
        "billing_api_call_made", "future_live_gate_environment_active", "mkb_db_opened",
        "active_mkb_write", "auto_accept_enabled", "medical_decision_made",
        "production_queue_mutated", "future_17c_live_not_started", "privacy_result", "safety_result",
    ):
        lines.append(f"- {key}: `{summary[key]}`")
    if notes:
        lines += ["", "## Notes", ""] + [f"- `{n}`" for n in notes]
    nxt = ("prepare 17C small live batch authorization for the 12 ready files"
           if summary["privacy_result"] == "passed"
           else "fix failed payload validation before any live batch")
    lines += ["", "## Recommended next (NO live batch started automatically)", "",
              f"- {nxt}. 17C requires explicit new authorization and is not started.", ""]
    return "\n".join(lines)


def write_outputs(summary: dict[str, Any], extra: dict[str, Any], notes: list[str]) -> int:
    public_batch = extra["public_batch"]
    schema_obj = extra["schema_obj"]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ready_batch_public = {
        "block": summary["block"],
        "ready_files_from_17a": summary["ready_files_from_17a"],
        "blocked_files_excluded": summary["blocked_files_excluded"],
        "outbound_requests_built": summary["outbound_requests_built"],
        "documents": public_batch,  # hashed ids + counts + status only
    }
    schema_validation_public = {
        "schema_name": schema_obj.get("schema_name"),
        "schema_version": schema_obj.get("schema_version"),
        "schema_created": summary["schema_created"],
        "prompt_contract_created": summary["prompt_contract_created"],
        "request_validation_passed": summary["request_validation_passed"],
        "validation_failures": summary["validation_failures"],
        "per_document_validation": [
            {"document_id": b["document_id"], "validation_status": b["validation_status"],
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

    matrix_md = _privacy_matrix(summary)
    gate_md = _live_gate_md(summary)
    impl_md = _implementation(summary, notes)

    # Defense-in-depth: confirm no tokenized payload line leaked into public blobs.
    public_blob = "\n".join([json.dumps(summary), json.dumps(ready_batch_public),
                             json.dumps(schema_validation_public), json.dumps(cost_public),
                             matrix_md, gate_md, impl_md])
    leak = 0
    try:
        outp = PRIVATE_OUT / "outbound_requests_private.jsonl"
        if outp.is_file():
            for req_line in outp.read_text(encoding="utf-8").splitlines():
                try:
                    content = json.loads(req_line).get("tokenized_content", "")
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
    READY_BATCH_PUBLIC_JSON.write_text(json.dumps(ready_batch_public, indent=2), encoding="utf-8")
    SCHEMA_VALIDATION_PUBLIC_JSON.write_text(json.dumps(schema_validation_public, indent=2), encoding="utf-8")
    DRY_RUN_COST_PUBLIC_JSON.write_text(json.dumps(cost_public, indent=2), encoding="utf-8")
    PRIVACY_GATE_MATRIX_MD.write_text(matrix_md, encoding="utf-8")
    LIVE_17C_ENTRY_GATE_MD.write_text(gate_md, encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation(summary, notes), encoding="utf-8")

    # ---- Downloads review copies (sanitized summaries only) ----
    try:
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        (DOWNLOADS / "ready_batch_summary_public.json").write_text(
            json.dumps({k: summary[k] for k in ("ready_files_from_17a", "blocked_files_excluded",
                        "outbound_requests_built", "request_validation_passed",
                        "future_17c_live_not_started")}, indent=2), encoding="utf-8")
        shutil.copyfile(SCHEMA_PATH, DOWNLOADS / "ai_extraction_schema_public.json")
        (DOWNLOADS / "cost_estimate_public.json").write_text(json.dumps(cost_public, indent=2), encoding="utf-8")
        (DOWNLOADS / "README_NEXT_STEPS.txt").write_text(
            "MedAI 17B AI-Extraction Dry-Run Review\n\n"
            "This folder contains sanitized summaries only. No tokenized payloads, no raw OCR,\n"
            "no token maps, and no PI values are included.\n\n"
            f"Ready files: {summary['ready_files_from_17a']}  Blocked excluded: {summary['blocked_files_excluded']}\n"
            f"Outbound requests built (private): {summary['outbound_requests_built']}\n"
            f"Request validation passed: {summary['request_validation_passed']}\n"
            f"Estimated cost (approx USD): {summary['estimated_cost_usd']}\n\n"
            "17C live batch is NOT started and requires explicit new authorization.\n",
            encoding="utf-8")
        (DOWNLOADS / "live_17C_authorization_checklist.txt").write_text(
            "17C Small Live Batch Authorization Checklist (NOT started)\n\n"
            "[ ] Explicit new user authorization granted\n"
            "[ ] Dry-run request_validation_passed == true\n"
            "[ ] Human review of dry-run outputs complete\n"
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
        and summary["future_17c_live_not_started"] is True
        and summary["schema_created"] is True
        and summary["prompt_contract_created"] is True
        and docs_ok
    )
    print("medai_ai_first_corpus_extraction_schema_batch_dry_run_17b_pass"
          if ok and summary["privacy_result"] == "passed"
          else ("medai_ai_first_corpus_extraction_schema_batch_dry_run_17b_blocked"
                if ok else "medai_ai_first_corpus_extraction_schema_batch_dry_run_17b_attention"))
    print(json.dumps({k: summary[k] for k in (
        "ready_files_from_17a", "blocked_files_excluded", "outbound_requests_built",
        "request_validation_passed", "estimated_input_tokens", "estimated_output_tokens",
        "estimated_cost_usd", "public_report_phi_leak_count", "provider_call_made",
        "gemini_call_made", "billing_api_call_made", "future_live_gate_environment_active",
        "future_17c_live_not_started", "privacy_result", "safety_result")}, indent=2))
    return 0 if ok else 1


def main() -> int:
    summary, extra, notes = run()
    return write_outputs(summary, extra, notes)


if __name__ == "__main__":
    raise SystemExit(main())
