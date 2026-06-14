#!/usr/bin/env python3
"""Local-only checkpoint unblock + public-report redaction repair for 17C-R2-R12.

This script performs no provider/model call and does not set any live gate. It only:
1. archives and clears a stale zero-completed failed checkpoint when safe,
2. reapplies public-report redaction to the 17C-R2 live reports,
3. validates the local gates required before the separately authorized live run.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
from execution import live_checkpoint as lc  # noqa: E402
from execution.jsonl_framing import read_jsonl_lines  # noqa: E402
from execution.public_report_redaction import redact_report_file  # noqa: E402
import scripts.run_medai_ai_first_corpus_17c_r2_authorized_total_cap_10_update_local_only_17c_r2_r11 as r11  # noqa: E402
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live  # noqa: E402

BLOCK = "MEDAI-AI-FIRST-CORPUS-17C-R2-CHECKPOINT-UNBLOCK-REDACTION-AND-FULL-LIVE-RUN-17C-R2-R12"
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_checkpoint_unblock_redaction_and_full_live_run_17c_r2_r12"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_CHECKPOINT_UNBLOCK_REDACTION_AND_FULL_LIVE_RUN_17C_R2_R12"
LIVE_REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_live_batch_vertex_17c_r2"
ARCHIVE_ROOT = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_17C_R2_FAILED_EVIDENCE_PRESERVE_PRIVATE"))

TRIAGED_FAILURES = {"truncated_or_invalid_json", "provider_truncated_by_max_tokens", "provider_truncated_mid_json"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_canonical_order() -> tuple[str, list[str], bool]:
    if not live.CANON_BATCH.is_file():
        return "", [], False
    lines = [line for line in read_jsonl_lines(live.CANON_BATCH) if line.strip()]
    order: list[str] = []
    malformed = False
    for line in lines:
        try:
            order.append(str(json.loads(line).get("document_id") or ""))
        except ValueError:
            malformed = True
    if malformed or len(order) != live.EXPECTED_REQUESTS or len(set(order)) != live.EXPECTED_REQUESTS:
        return "", order, False
    return lc.sha256_file(live.CANON_BATCH), order, True


def _archive_checkpoint(timestamp: str) -> tuple[bool, str, int]:
    dest = ARCHIVE_ROOT / f"checkpoint_reset_archive_{timestamp}"
    copied = 0
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for name in lc.CHECKPOINT_FILES:
            src = lc.CHECKPOINT_DIR / name
            if src.is_file():
                shutil.copy2(src, dest / name)
                copied += 1
        (dest / "README_PRIVATE_DO_NOT_SHARE.txt").write_text(
            "Private checkpoint archive for 17C-R2-R12 reset. Do not commit or share.\n",
            encoding="utf-8",
        )
        return True, str(dest), copied
    except OSError:
        return False, str(dest), copied


def _checkpoint_action(batch_sha: str, order: list[str]) -> dict[str, Any]:
    state = lc.load_state()
    completed = lc.load_completed()
    failed = lc.load_failed()
    result: dict[str, Any] = {
        "checkpoint_block_detected": False,
        "checkpoint_unblock_action": "none",
        "checkpoint_archive_created": False,
        "checkpoint_archive_path_label": "",
        "checkpoint_archive_file_count": 0,
        "completed_request_count_before_unblock": len(completed),
        "failed_request_present_before_unblock": bool(failed),
        "failed_doc_hash_before_unblock": (failed or {}).get("failed_doc_id", ""),
        "failed_request_index_before_unblock": (failed or {}).get("request_index"),
        "failed_category_before_unblock": (failed or {}).get("failure_category", ""),
        "checkpoint_inconsistent": False,
        "unresolved_failed_doc_after_unblock": False,
        "checkpoint_clean_or_resumable_before_live": False,
    }
    start, blocked, reason, completed_set = lc.decide_resume(batch_sha, order, live.EXPECTED_REQUESTS)
    result["checkpoint_resume_reason_before_unblock"] = reason
    result["checkpoint_resume_start_index_before_unblock"] = start
    if not blocked:
        result["checkpoint_clean_or_resumable_before_live"] = True
        return result
    result["checkpoint_block_detected"] = True
    if reason != "failed_doc_unresolved_requires_triage_or_reset":
        result["checkpoint_inconsistent"] = True
        result["checkpoint_unblock_action"] = f"blocked:{reason}"
        return result
    if len(completed_set) > 0 or len(completed) > 0:
        result["checkpoint_unblock_action"] = "blocked:completed_ids_present"
        return result
    failure_category = str((failed or {}).get("failure_category") or "")
    if failure_category not in TRIAGED_FAILURES:
        result["checkpoint_unblock_action"] = f"blocked:untriaged_failure:{failure_category}"
        return result
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    ok, archive_path, copied = _archive_checkpoint(ts)
    result["checkpoint_archive_created"] = ok
    result["checkpoint_archive_path_label"] = "PRIVATE_CHECKPOINT_ARCHIVE_PATH_REDACTED" if ok else ""
    result["checkpoint_archive_file_count"] = copied
    if not ok:
        result["checkpoint_unblock_action"] = "blocked:archive_failed"
        return result
    lc.reset_checkpoint()
    start_after, blocked_after, reason_after, completed_after = lc.decide_resume(batch_sha, order, live.EXPECTED_REQUESTS)
    result["checkpoint_unblock_action"] = "archived_and_reset_stale_zero_completed_failed_checkpoint"
    result["checkpoint_resume_reason_after_unblock"] = reason_after
    result["checkpoint_resume_start_index_after_unblock"] = start_after
    result["completed_request_count_after_unblock"] = len(completed_after)
    result["unresolved_failed_doc_after_unblock"] = bool(lc.load_failed())
    result["checkpoint_clean_or_resumable_before_live"] = not blocked_after and not result["unresolved_failed_doc_after_unblock"]
    return result


def _redact_live_reports() -> dict[str, Any]:
    scanned = changed = path_redactions = secret_redactions = path_after = secret_after = phi_after = 0
    per_file: list[dict[str, Any]] = []
    for path in sorted(LIVE_REPORT_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".csv"}:
            continue
        scanned += 1
        original = path.read_text(encoding="utf-8", errors="ignore")
        redacted, np, ns = redact_report_file(original, is_json=path.suffix.lower() == ".json")
        if redacted != original:
            path.write_text(redacted, encoding="utf-8")
            changed += 1
        check = check_public_report_payload(redacted)
        path_after += check.private_filename_path_leaks
        secret_after += check.secret_leaks
        phi_after += 0 if check.raw_phi_logged_in_public_reports is False else 1
        path_redactions += np
        secret_redactions += ns
        per_file.append(
            {
                "file": path.name,
                "path_redactions_applied": np,
                "secret_redactions_applied": ns,
                "private_path_leaks_after": check.private_filename_path_leaks,
                "secret_leaks_after": check.secret_leaks,
                "passed_after": check.passed,
            }
        )
    return {
        "checkpoint_branch_redaction_applied": changed > 0 or (path_after == 0 and secret_after == 0),
        "reports_scanned": scanned,
        "reports_changed": changed,
        "path_redactions_applied": path_redactions,
        "secret_redactions_applied": secret_redactions,
        "private_path_leaks_after": path_after,
        "secret_leaks_after": secret_after,
        "public_report_phi_leak_count": phi_after,
        "per_file": per_file,
    }


def _write_docs(summary: dict[str, Any]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "MEDAI_17C_R2_R12_CHECKPOINT_UNBLOCK_REDACTION_AND_LIVE_RUN.md").write_text(
        "# MEDAI 17C-R2 R12 Checkpoint Unblock, Redaction, And Full Live Run\n\n"
        "R12 first performs a local checkpoint unblock and public report redaction repair. "
        "Only after those gates pass may the already authorized full 478-request live run be executed once. "
        "No MKB write, auto-accept, or medical decision is authorized.\n",
        encoding="utf-8",
    )
    (DOC_DIR / "MEDAI_17C_R2_R12_LOCAL_SAFETY_GATE.md").write_text(
        "# MEDAI 17C-R2 R12 Local Safety Gate\n\n"
        f"- checkpoint_clean_or_resumable_before_live: `{summary['checkpoint_clean_or_resumable_before_live']}`\n"
        f"- ready_for_full_live_run: `{summary['ready_for_full_live_run']}`\n"
        f"- private_path_leaks_after: `{summary['private_path_leaks_after']}`\n"
        f"- secret_leaks_after: `{summary['secret_leaks_after']}`\n",
        encoding="utf-8",
    )


def _write_reports(summary: dict[str, Any], checkpoint: dict[str, Any], redaction: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(REPORT_DIR / "summary.json", summary)
    _write_json(REPORT_DIR / "checkpoint_unblock_public.json", checkpoint)
    _write_json(REPORT_DIR / "redaction_repair_public.json", redaction)
    (REPORT_DIR / "implementation_report.md").write_text(
        f"# {BLOCK}\n\n"
        f"- checkpoint_block_detected: `{summary['checkpoint_block_detected']}`\n"
        f"- checkpoint_unblock_action: `{summary['checkpoint_unblock_action']}`\n"
        f"- checkpoint_archive_created: `{summary['checkpoint_archive_created']}`\n"
        f"- unresolved_failed_doc_after_unblock: `{summary['unresolved_failed_doc_after_unblock']}`\n"
        f"- ready_for_full_live_run: `{summary['ready_for_full_live_run']}`\n"
        f"- private_path_leaks_after: `{summary['private_path_leaks_after']}`\n"
        f"- secret_leaks_after: `{summary['secret_leaks_after']}`\n"
        "- No provider/model call was made by this local repair script.\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "live_run_gate_public.md").write_text(
        "# 17C-R2 R12 live run gate\n\n"
        f"- ready_for_full_live_run: `{summary['ready_for_full_live_run']}`\n"
        f"- canonical_batch_valid: `{summary['canonical_batch_valid']}`\n"
        f"- credential_preflight_passed: `{summary['credential_preflight_passed']}`\n"
        f"- total_cap_usd: `{summary['total_cap_usd']}`\n"
        f"- per_chunk_cap_usd: `{summary['per_chunk_cap_usd']}`\n"
        f"- max_output_tokens: `{summary['max_output_tokens']}`\n",
        encoding="utf-8",
    )


def run() -> dict[str, Any]:
    batch_sha, order, canonical_valid = _load_canonical_order()
    checkpoint = _checkpoint_action(batch_sha, order) if canonical_valid else {
        "checkpoint_block_detected": False,
        "checkpoint_unblock_action": "blocked:canonical_batch_invalid",
        "checkpoint_archive_created": False,
        "unresolved_failed_doc_after_unblock": False,
        "checkpoint_clean_or_resumable_before_live": False,
        "checkpoint_inconsistent": False,
    }
    redaction = _redact_live_reports()
    r11_summary = r11.build_summary()
    ready = bool(
        canonical_valid
        and checkpoint["checkpoint_clean_or_resumable_before_live"]
        and not checkpoint["checkpoint_inconsistent"]
        and not checkpoint["unresolved_failed_doc_after_unblock"]
        and redaction["private_path_leaks_after"] == 0
        and redaction["secret_leaks_after"] == 0
        and redaction["public_report_phi_leak_count"] == 0
        and r11_summary["ready_to_resume_17c_r2_live"]
        and float(live.CAP_TOTAL) == 10.00
        and float(live.CAP_PER_CHUNK) == 0.05
        and int(live.MAX_OUTPUT_TOKENS) == 8192
    )
    summary = {
        "block": BLOCK,
        "local_only_repair": True,
        "provider_model_call_made": False,
        "vertex_model_call_made": False,
        "gemini_call_made": False,
        "billing_api_call_made": False,
        "live_gate_set": False,
        "live_extraction_started": False,
        "checkpoint_block_detected": checkpoint["checkpoint_block_detected"],
        "checkpoint_unblock_action": checkpoint["checkpoint_unblock_action"],
        "checkpoint_archive_created": checkpoint["checkpoint_archive_created"],
        "checkpoint_archive_path_label": checkpoint.get("checkpoint_archive_path_label", ""),
        "checkpoint_archive_file_count": checkpoint.get("checkpoint_archive_file_count", 0),
        "unresolved_failed_doc_after_unblock": checkpoint["unresolved_failed_doc_after_unblock"],
        "checkpoint_clean_or_resumable_before_live": checkpoint["checkpoint_clean_or_resumable_before_live"],
        "checkpoint_inconsistent": checkpoint["checkpoint_inconsistent"],
        "checkpoint_branch_redaction_applied": redaction["checkpoint_branch_redaction_applied"],
        "private_path_leaks_after": redaction["private_path_leaks_after"],
        "secret_leaks_after": redaction["secret_leaks_after"],
        "public_report_phi_leak_count": redaction["public_report_phi_leak_count"],
        "canonical_batch_valid": canonical_valid,
        "request_count_total": len(order),
        "credential_preflight_passed": r11_summary["credential_preflight_passed"],
        "total_cap_usd": live.CAP_TOTAL,
        "per_chunk_cap_usd": live.CAP_PER_CHUNK,
        "max_output_tokens": live.MAX_OUTPUT_TOKENS,
        "selected_chunk_size": r11_summary["selected_chunk_size"],
        "estimated_total_cost_usd": r11_summary["estimated_total_cost_with_new_output_ceiling_usd"],
        "estimated_per_chunk_cost_usd": r11_summary["estimated_per_chunk_cost_with_selected_chunk_size_usd"],
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "production_queue_mutated": False,
        "ready_for_full_live_run": ready,
        "private_checkpoint_evidence_responses_committed": False,
        "raw_ai_response_committed": False,
        "tokenized_payloads_committed": False,
        "credentials_tokens_committed": False,
        "privacy_result": "passed" if redaction["private_path_leaks_after"] == 0 and redaction["secret_leaks_after"] == 0 else "blocked",
        "safety_result": "passed" if ready else "blocked",
    }
    _write_docs(summary)
    _write_reports(summary, checkpoint, redaction)
    return summary


def main() -> int:
    summary = run()
    result = "PASS" if summary["ready_for_full_live_run"] else "BLOCKED"
    print(f"{BLOCK}_{result}")
    print(json.dumps({
        "checkpoint_block_detected": summary["checkpoint_block_detected"],
        "checkpoint_unblock_action": summary["checkpoint_unblock_action"],
        "checkpoint_archive_created": summary["checkpoint_archive_created"],
        "unresolved_failed_doc_after_unblock": summary["unresolved_failed_doc_after_unblock"],
        "checkpoint_clean_or_resumable_before_live": summary["checkpoint_clean_or_resumable_before_live"],
        "private_path_leaks_after": summary["private_path_leaks_after"],
        "secret_leaks_after": summary["secret_leaks_after"],
        "credential_preflight_passed": summary["credential_preflight_passed"],
        "ready_for_full_live_run": summary["ready_for_full_live_run"],
    }, indent=2, sort_keys=True))
    return 0 if summary["ready_for_full_live_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
