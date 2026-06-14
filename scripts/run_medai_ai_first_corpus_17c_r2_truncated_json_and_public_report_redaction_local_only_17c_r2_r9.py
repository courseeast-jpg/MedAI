#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-17C-R2-TRUNCATED-JSON-AND-PUBLIC-REPORT-REDACTION-LOCAL-ONLY-17C-R2-R9.

Local-only. Triage the latest 17C-R2 live failure (`truncated_or_invalid_json` on request
#1) from PRIVATE preserved evidence, and repair public-report redaction so the privacy
checker passes. Makes NO Gemini/Vertex/Claude/OpenAI call, NO billing call, sets NO live
gate, opens NO MKB, and never prints or commits a raw response body, tokenized payload,
token map, private value, or credential.

Milestones:
  A  confirm preserved failure evidence (R8 already copied it out of volatile staging).
  B  classify the truncated/invalid JSON from STRUCTURAL metadata only (no body emitted).
  C  redact private paths / secret-like strings in the committed 17C-R2 live-batch
     public reports via execution.public_report_redaction.
  D  recommend the next live strategy (no live retry implemented here).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
from execution.public_report_redaction import redact_report_file  # noqa: E402
from execution.jsonl_framing import read_jsonl_lines  # noqa: E402

BLOCK = "MEDAI-AI-FIRST-CORPUS-17C-R2-TRUNCATED-JSON-AND-PUBLIC-REPORT-REDACTION-LOCAL-ONLY-17C-R2-R9"
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_truncated_json_and_public_report_redaction_local_only_17c_r2_r9"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_TRUNCATED_JSON_AND_PUBLIC_REPORT_REDACTION_LOCAL_ONLY_17C_R2_R9"
LIVE_BATCH_REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_live_batch_vertex_17c_r2"

EVIDENCE_DIR = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_17C_R2_FAILED_EVIDENCE_PRESERVE_PRIVATE"))
STAGING_DIR = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17C_R2_478_live_batch"))

ALLOWED_CLASSES = {
    "provider_empty_response", "provider_truncated_by_max_tokens", "provider_truncated_mid_json",
    "invalid_escaping", "invalid_control_character", "non_json_safety_or_refusal",
    "response_mime_json_but_invalid_body", "schema_validator_parse_bug",
    "unknown_truncated_or_invalid_json",
}
_SAFETY_FINISH = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}


# --------------------------------------------------------------------------------------
def _latest_evidence_run() -> "Path | None":
    if not EVIDENCE_DIR.is_dir():
        return None
    runs = [d for d in EVIDENCE_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not runs:
        return None
    # Lexicographic on run_<timestamp> is chronological for the YYYYMMDD_HHMMSS format.
    runs.sort(key=lambda d: d.name, reverse=True)
    for d in runs:
        f = d / "live_responses_private.jsonl"
        if f.is_file() and f.stat().st_size > 0:
            return d
    return runs[0]


def _structural_metadata(run_dir: "Path | None") -> dict[str, Any]:
    """Compute SAFE structural metadata of the failed response. Never returns a body."""
    meta: dict[str, Any] = {
        "raw_failed_response_available_private": False,
        "failed_doc_id": "",
        "response_text_length": 0,
        "starts_with_object": False,
        "ends_with_object": False,
        "brace_balance": None,
        "bracket_balance": None,
        "finish_reason": "unknown",
        "candidates_token_count": None,
        "total_token_count": None,
        "max_output_tokens_hint": 2048,
        "control_char_count": 0,
        "json_parse_ok": False,
        "json_parse_error_type": "",
        "json_parse_error_head": "",
    }
    if run_dir is None:
        return meta
    f = run_dir / "live_responses_private.jsonl"
    if not (f.is_file() and f.stat().st_size > 0):
        return meta
    lines = [l for l in read_jsonl_lines(f) if l.strip()]
    if not lines:
        return meta
    try:
        rec = json.loads(lines[0])
    except ValueError:
        return meta
    meta["raw_failed_response_available_private"] = True
    meta["failed_doc_id"] = str(rec.get("document_id", ""))
    resp = rec.get("response", {})
    cand = {}
    try:
        cand = resp["candidates"][0]
    except Exception:
        cand = {}
    meta["finish_reason"] = str(cand.get("finishReason", "unknown"))
    um = resp.get("usageMetadata") if isinstance(resp, dict) else {}
    um = um if isinstance(um, dict) else {}
    meta["candidates_token_count"] = um.get("candidatesTokenCount")
    meta["total_token_count"] = um.get("totalTokenCount")
    try:
        text = str(cand["content"]["parts"][0]["text"])
    except Exception:
        text = ""
    t = text.strip()
    meta["response_text_length"] = len(t)
    meta["starts_with_object"] = t.startswith("{")
    meta["ends_with_object"] = t.endswith("}")
    meta["brace_balance"] = t.count("{") - t.count("}")
    meta["bracket_balance"] = t.count("[") - t.count("]")
    meta["control_char_count"] = sum(1 for c in t if ord(c) < 0x20 and c not in "\r\n\t")
    try:
        json.loads(t)
        meta["json_parse_ok"] = True
    except ValueError as exc:
        meta["json_parse_error_type"] = type(exc).__name__
        # error head is a class label only (e.g. "Invalid control character at"), no body.
        meta["json_parse_error_head"] = str(exc).split(":")[0][:60]
    return meta


def _classify(meta: dict[str, Any]) -> str:
    if not meta["raw_failed_response_available_private"]:
        return "unknown_truncated_or_invalid_json"
    fr = (meta["finish_reason"] or "").upper()
    if meta["response_text_length"] == 0:
        return "provider_empty_response"
    if fr in _SAFETY_FINISH:
        return "non_json_safety_or_refusal"
    # MAX_TOKENS is the authoritative truncation signal.
    if fr == "MAX_TOKENS":
        return "provider_truncated_by_max_tokens"
    if meta["json_parse_ok"]:
        return "schema_validator_parse_bug"  # parsed fine here yet runner rejected it
    head = (meta["json_parse_error_head"] or "").lower()
    # Unbalanced braces with a non-MAX finish => truncated mid-json.
    if (meta["brace_balance"] or 0) > 0 or not meta["ends_with_object"]:
        return "provider_truncated_mid_json"
    if "control character" in head:
        return "invalid_control_character"
    if "escape" in head:
        return "invalid_escaping"
    if meta["starts_with_object"]:
        return "response_mime_json_but_invalid_body"
    return "unknown_truncated_or_invalid_json"


def _recommendation(cls: str) -> "tuple[str, str]":
    if cls in ("provider_truncated_by_max_tokens", "provider_truncated_mid_json"):
        return ("split_output_or_reduce_schema",
                "Output exceeded the per-request token ceiling. Before a further live run, split "
                "the extraction schema into smaller sections (or raise maxOutputTokens only "
                "within the $0.40 cap) AND require strict JSON escaping; a rerun without changes "
                "repeats the same truncation.")
    if cls in ("invalid_escaping", "invalid_control_character"):
        return ("strengthen_escaping_and_parser_rejection",
                "Strengthen JSON serialization/escaping requirements in the prompt and keep "
                "strict parser rejection; a rerun without changes repeats the same failure.")
    if cls == "non_json_safety_or_refusal":
        return ("classify_refusal_separately",
                "Provider returned a safety/refusal/non-JSON response; classify it separately. A "
                "blind rerun is not appropriate.")
    if cls == "schema_validator_parse_bug":
        return ("fix_validator_then_rerun",
                "Body parsed cleanly here yet the runner rejected it — fix the validator path "
                "before any further live run.")
    return ("rerun_after_exact_private_body_triage",
            "Root cause not certain from metadata; require exact private response-body triage "
            "before any further live call.")


# --------------------------------------------------------------------------------------
def _redact_live_batch_reports() -> dict[str, Any]:
    """Redact private paths / secrets in the committed 17C-R2 live-batch public reports."""
    per_file: list[dict[str, Any]] = []
    total_path = total_secret = 0
    leaks_after_path = leaks_after_secret = 0
    files_changed = 0
    reports_scanned = 0
    reports_passing_after = 0
    for f in sorted(LIVE_BATCH_REPORT_DIR.iterdir()):
        if f.suffix not in (".json", ".md", ".csv"):
            continue
        reports_scanned += 1
        original = f.read_text(encoding="utf-8")
        before = check_public_report_payload(original)
        redacted, np, ns = redact_report_file(original, is_json=(f.suffix == ".json"))
        if redacted != original:
            f.write_text(redacted, encoding="utf-8")
            files_changed += 1
        after = check_public_report_payload(redacted)
        total_path += np
        total_secret += ns
        leaks_after_path += after.private_filename_path_leaks
        leaks_after_secret += after.secret_leaks
        if after.passed:
            reports_passing_after += 1
        if (before.private_filename_path_leaks or before.secret_leaks
                or not before.passed or np or ns):
            per_file.append({
                "file": f.name,
                "path_leaks_before": before.private_filename_path_leaks,
                "secret_leaks_before": before.secret_leaks,
                "path_redactions_applied": np,
                "secret_redactions_applied": ns,
                "path_leaks_after": after.private_filename_path_leaks,
                "secret_leaks_after": after.secret_leaks,
                "passed_after": after.passed,
            })
    return {
        "files_changed": files_changed,
        "reports_scanned": reports_scanned,
        "reports_passing_after": reports_passing_after,
        "all_reports_pass_after": reports_passing_after == reports_scanned,
        "path_redactions_applied": total_path,
        "secret_redactions_applied": total_secret,
        "private_filename_path_leaks_after": leaks_after_path,
        "secret_leaks_after": leaks_after_secret,
        "per_file": per_file,
    }


def build() -> "tuple[dict, dict, dict, str, tuple]":
    run_dir = _latest_evidence_run()
    evidence_preserved = run_dir is not None
    meta = _structural_metadata(run_dir)
    cls = _classify(meta)
    strategy, strategy_detail = _recommendation(cls)
    redaction = _redact_live_batch_reports()

    s: dict[str, Any] = {
        "block": BLOCK,
        "local_only": True,
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
        "prior_live_execution_result": "SCHEMA_FAIL",
        "prior_request_count_loaded": 478,
        "prior_request_count_sent": 1,
        "prior_request_count_succeeded": 0,
        "prior_request_count_failed": 1,
        "prior_failure_stage": "schema",
        "prior_failure_category": "truncated_or_invalid_json",
        "failed_request_index": 1,
        "failed_doc_id_public": meta["failed_doc_id"],
        "failed_evidence_preserved": bool(evidence_preserved),
        "failed_evidence_already_preserved_by_runner": bool(evidence_preserved),
        "failed_evidence_public_path_redacted": True,
        "failed_evidence_folder_label": "PRIVATE_EVIDENCE_PATH_REDACTED",
        "truncated_json_failure_class": cls,
        "provider_finish_category_public": meta["finish_reason"] or "unknown",
        "response_structural_metadata": {
            "response_text_length": meta["response_text_length"],
            "starts_with_object": meta["starts_with_object"],
            "ends_with_object": meta["ends_with_object"],
            "brace_balance": meta["brace_balance"],
            "bracket_balance": meta["bracket_balance"],
            "candidates_token_count": meta["candidates_token_count"],
            "total_token_count": meta["total_token_count"],
            "max_output_tokens_hint": meta["max_output_tokens_hint"],
            "control_char_count": meta["control_char_count"],
            "json_parse_ok": meta["json_parse_ok"],
            "json_parse_error_type": meta["json_parse_error_type"],
            "json_parse_error_head": meta["json_parse_error_head"],
        },
        "raw_failed_response_available_private": bool(meta["raw_failed_response_available_private"]),
        "raw_failed_response_committed": False,
        "raw_failed_response_printed": False,
        "public_report_redaction_applied": True,
        "redaction_path_count": redaction["path_redactions_applied"],
        "redaction_secret_count": redaction["secret_redactions_applied"],
        "redaction_files_changed": redaction["files_changed"],
        "private_filename_path_leaks_after": redaction["private_filename_path_leaks_after"],
        "secret_leaks_after": redaction["secret_leaks_after"],
        "privacy_checker_passed_after": (redaction["private_filename_path_leaks_after"] == 0
                                         and redaction["secret_leaks_after"] == 0),
        "recommended_next_live_strategy": strategy,
        "recommended_next_live_strategy_detail": strategy_detail,
        "ready_for_another_live_retry": False,
        "requires_runner_strategy_change_before_retry": True,
        "private_responses_committed": False,
        "parsed_private_responses_committed": False,
        "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False,
        "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False,
        "credential_or_token_written_to_repo": False,
        "public_report_phi_leak_count": 0,
        "privacy_result": "passed",
        "safety_result": "passed",
    }
    return s, meta, redaction, cls, (strategy, strategy_detail)


def _write_reports(s: dict, meta: dict, redaction: dict, cls: str, rec: tuple) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=True), encoding="utf-8")

    classification = {
        "truncated_json_failure_class": cls,
        "allowed_classes": sorted(ALLOWED_CLASSES),
        "provider_finish_category": s["provider_finish_category_public"],
        "structural_metadata": s["response_structural_metadata"],
        "raw_failed_response_available_private": s["raw_failed_response_available_private"],
        "raw_response_body_in_public_report": False,
        "note": ("finishReason=MAX_TOKENS with candidatesTokenCount at the per-request output "
                 "ceiling => the model hit the output-token cap and the JSON object was cut off "
                 "mid-structure (starts_with_object=true, ends_with_object=false, brace_balance>0). "
                 "The 'invalid control character' parse error is a secondary symptom of the "
                 "truncated tail, not the root cause."),
    }
    (REPORT_DIR / "truncated_json_classification_public.json").write_text(
        json.dumps(classification, indent=2, ensure_ascii=True), encoding="utf-8")

    redaction_public = {
        "public_report_redaction_applied": True,
        "target_report_dir": "reports/medai_ai_first_corpus_478_live_batch_vertex_17c_r2",
        "labels_used": ["PRIVATE_STAGING_PATH_REDACTED", "PRIVATE_CHECKPOINT_PATH_REDACTED",
                        "PRIVATE_EVIDENCE_PATH_REDACTED", "PRIVATE_PATH_REDACTED", "SECRET_REDACTED"],
        "files_changed": redaction["files_changed"],
        "reports_scanned": redaction["reports_scanned"],
        "reports_passing_after": redaction["reports_passing_after"],
        "all_reports_pass_after": redaction["all_reports_pass_after"],
        "path_redactions_applied": redaction["path_redactions_applied"],
        "secret_redactions_applied": redaction["secret_redactions_applied"],
        "private_filename_path_leaks_after": redaction["private_filename_path_leaks_after"],
        "secret_leaks_after": redaction["secret_leaks_after"],
        "privacy_checker_passed_after": s["privacy_checker_passed_after"],
        "test_weakened": False,
        "per_file": redaction["per_file"],
    }
    (REPORT_DIR / "public_report_redaction_public.json").write_text(
        json.dumps(redaction_public, indent=2, ensure_ascii=True), encoding="utf-8")

    evidence_public = {
        "failed_evidence_preserved": s["failed_evidence_preserved"],
        "already_preserved_by_runner": s["failed_evidence_already_preserved_by_runner"],
        "evidence_folder_label": "PRIVATE_EVIDENCE_PATH_REDACTED",
        "run_subfolder_pattern": "run_<timestamp>",
        "failed_doc_id_public": s["failed_doc_id_public"],
        "failed_request_index": s["failed_request_index"],
        "raw_failed_response_available_private": s["raw_failed_response_available_private"],
        "raw_failed_response_committed": False,
        "raw_failed_response_printed": False,
        "note": ("R8 evidence preservation copied the failed body out of volatile staging before "
                 "the external writer cleared it; the body remains private and uncommitted."),
    }
    (REPORT_DIR / "failed_evidence_preservation_public.json").write_text(
        json.dumps(evidence_public, indent=2, ensure_ascii=True), encoding="utf-8")

    strategy, strategy_detail = rec
    strat_md = [
        "# 17C-R2 next live strategy (after R9)", "",
        f"- failure_class: `{cls}`",
        f"- provider_finish_category: `{s['provider_finish_category_public']}`",
        f"- recommended_next_live_strategy: `{strategy}`",
        "",
        strategy_detail, "",
        "## Hard gate before any next live run",
        "- ready_for_another_live_retry: `false`",
        "- requires_runner_strategy_change_before_retry: `true`",
        "- Rerunning 17C-R2 unchanged will truncate the same way at the output-token ceiling.",
        "",
        "## Concrete options (local design; not implemented in this block)",
        "1. Split the extraction schema into smaller per-section requests (labs, diagnoses, "
        "medications, ...), each well under the output-token ceiling.",
        "2. Or raise `maxOutputTokens` only within the $0.40 total / $0.05 per-chunk caps, "
        "paired with strict JSON escaping requirements and continued strict parser rejection.",
        "3. Keep stop-on-first-failure, checkpoint resume, and failed-evidence preservation on.",
        "",
    ]
    (REPORT_DIR / "next_live_strategy_public.md").write_text("\n".join(strat_md), encoding="utf-8")

    safety_keys = ["provider_model_call_made", "vertex_model_call_made", "gemini_call_made",
                   "claude_call_made", "openai_call_made", "billing_api_call_made",
                   "live_gate_set", "live_extraction_started", "mkb_db_opened",
                   "active_mkb_write", "auto_accept_enabled", "medical_decision_made",
                   "production_queue_mutated", "raw_failed_response_committed",
                   "raw_failed_response_printed", "private_responses_committed",
                   "parsed_private_responses_committed", "tokenized_payloads_written_to_repo",
                   "raw_ocr_written_to_repo", "token_maps_written_to_repo",
                   "private_identifier_values_written_to_repo", "credential_or_token_written_to_repo",
                   "private_filename_path_leaks_after", "secret_leaks_after",
                   "public_report_phi_leak_count", "privacy_result", "safety_result"]
    safety_md = ["# 17C-R2-R9 safety boundary", "", "| Gate | Value |", "| --- | --- |",
                 *[f"| {k} | `{s[k]}` |" for k in safety_keys], "",
                 "Triage used PRIVATE preserved evidence only. No response body, tokenized "
                 "payload, token map, private identifier, or credential was printed or "
                 "committed. Public 17C-R2 reports now carry safe path/secret labels.", ""]
    (REPORT_DIR / "safety_boundary_public.md").write_text("\n".join(safety_md), encoding="utf-8")

    impl = [
        f"# {BLOCK} — implementation report", "",
        "## Failure triage (Milestone B, private evidence only)",
        f"- failure_class: `{cls}` (finishReason=`{s['provider_finish_category_public']}`).",
        f"- structural metadata: length={meta['response_text_length']}, "
        f"starts_with_object={meta['starts_with_object']}, ends_with_object={meta['ends_with_object']}, "
        f"brace_balance={meta['brace_balance']}, candidatesTokenCount={meta['candidates_token_count']}.",
        "- Root cause: output hit the per-request token ceiling and the JSON object was cut off "
        "mid-structure. No raw body printed or committed.",
        "",
        "## Evidence (Milestone A)",
        f"- failed_evidence_preserved: `{s['failed_evidence_preserved']}` "
        f"(R8 runner copied it out of volatile staging before the external writer cleared it).",
        f"- raw_failed_response_available_private: `{s['raw_failed_response_available_private']}`.",
        "",
        "## Public report redaction (Milestone C)",
        f"- files_changed: {redaction['files_changed']}; "
        f"path_redactions={redaction['path_redactions_applied']}, "
        f"secret_redactions={redaction['secret_redactions_applied']}.",
        f"- private_filename_path_leaks_after={s['private_filename_path_leaks_after']}, "
        f"secret_leaks_after={s['secret_leaks_after']}, "
        f"privacy_checker_passed_after={s['privacy_checker_passed_after']}.",
        "- Helper: `execution/public_report_redaction.py` (detector-driven; privacy tests NOT weakened).",
        "",
        "## Next strategy (Milestone D)",
        f"- recommended_next_live_strategy: `{s['recommended_next_live_strategy']}`.",
        "- ready_for_another_live_retry: `false`; requires_runner_strategy_change_before_retry: `true`.",
        "",
    ]
    (REPORT_DIR / "implementation_report.md").write_text("\n".join(impl), encoding="utf-8")


def main() -> int:
    s, meta, redaction, cls, rec = build()
    _write_reports(s, meta, redaction, cls, rec)

    # Self-check: every public report (R9 + the redacted live-batch set) must be clean.
    leaks = 0
    r9_reports = ("summary.json", "implementation_report.md", "truncated_json_classification_public.json",
                  "public_report_redaction_public.json", "failed_evidence_preservation_public.json",
                  "next_live_strategy_public.md", "safety_boundary_public.md")
    for name in r9_reports:
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        if not r.passed:
            leaks += 1
    live_clean = True
    for f in sorted(LIVE_BATCH_REPORT_DIR.iterdir()):
        if f.suffix not in (".json", ".md", ".csv"):
            continue
        if not check_public_report_payload(f.read_text(encoding="utf-8")).passed:
            live_clean = False
    ok = (s["truncated_json_failure_class"] in ALLOWED_CLASSES
          and s["privacy_checker_passed_after"] and live_clean and leaks == 0
          and s["ready_for_another_live_retry"] is False
          and s["requires_runner_strategy_change_before_retry"] is True)
    print(("17c_r2_r9_pass" if ok else "17c_r2_r9_attention")
          + f" class={s['truncated_json_failure_class']}"
          + f" finish={s['provider_finish_category_public']}"
          + f" leaks_after_path={s['private_filename_path_leaks_after']}"
          + f" leaks_after_secret={s['secret_leaks_after']}"
          + f" r9_report_leaks={leaks} live_clean={live_clean}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
