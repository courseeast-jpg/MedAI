#!/usr/bin/env python3
"""Local-only 17C-R2-R11 total cost-cap authorization update.

This script records and validates the user-authorized total cap increase from
$0.40 to $10.00 for the canonical 478-request 17C-R2 live runner. It performs
only local preflight/reporting. It does not run live extraction, set live gates,
call a model, call billing APIs, open MKB, or write private responses.
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
import scripts.run_medai_ai_first_corpus_17c_r2_max_output_strategy_local_only_17c_r2_r10 as r10  # noqa: E402
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live  # noqa: E402

BLOCK = "MEDAI-AI-FIRST-CORPUS-17C-R2-AUTHORIZED-TOTAL-CAP-10-UPDATE-LOCAL-ONLY-17C-R2-R11"
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_authorized_total_cap_10_update_local_only_17c_r2_r11"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_AUTHORIZED_TOTAL_CAP_10_UPDATE_LOCAL_ONLY_17C_R2_R11"
R10_REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_max_output_strategy_local_only_17c_r2_r10"
LIVE_REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_live_batch_vertex_17c_r2"

PREVIOUS_TOTAL_CAP = 0.40
AUTHORIZED_TOTAL_CAP = 10.00
PER_CHUNK_CAP = 0.05
MAX_OUTPUT_TOKENS = 8192
REQUEST_COUNT_TOTAL = 478

DOC_NAMES = (
    "MEDAI_17C_R2_AUTHORIZED_TOTAL_CAP_10_UPDATE_LOCAL_ONLY_17C_R2_R11.md",
    "MEDAI_17C_R2_COST_AUTHORIZATION_POLICY_R11.md",
    "MEDAI_17C_R2_FULL_CORPUS_LIVE_ENTRY_GATE_AFTER_R11.md",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _public_report_leak_counts() -> tuple[int, int, int]:
    phi = path_leaks = secret_leaks = 0
    for directory in (R10_REPORT_DIR, LIVE_REPORT_DIR):
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".csv"}:
                continue
            result = check_public_report_payload(path.read_text(encoding="utf-8", errors="ignore"))
            phi += 0 if result.raw_phi_logged_in_public_reports is False else 1
            path_leaks += result.private_filename_path_leaks
            secret_leaks += result.secret_leaks
    return phi, path_leaks, secret_leaks


def _live_runner_flags() -> dict[str, bool]:
    src = inspect.getsource(live)
    return {
        "checkpoint_resume_preserved": "lc.decide_resume(" in src and "lc.init_checkpoint(" in src,
        "failed_evidence_preservation_preserved": "lc.preserve_failed_evidence(" in src,
        "public_report_redaction_preserved": "PRIVATE_PATH_REDACTED" in src or "PRIVATE_STAGING_PATH_REDACTED" in src,
        "adaptive_chunk_size_enabled": "planner.select_chunk_size(" in src,
        "stop_on_first_failure_preserved": "stopped_on_first_failure" in src and "mark_failed" in src,
        "live_gate_scoped_per_chunk_preserved": (
            "os.environ" in src and "[LIVE_GATE]" in src and "os.environ.pop(LIVE_GATE" in src
        ),
    }


def build_summary() -> dict[str, Any]:
    r10_summary = _read_json(R10_REPORT_DIR / "summary.json")
    plan = r10._cost_plan()
    flags = _live_runner_flags()
    credential_ok, credential_category = live._credential_preflight()
    phi_leaks, private_path_leaks, secret_leaks = _public_report_leak_counts()

    total_estimate = float(plan.get("estimated_total_cost_with_new_output_ceiling_usd") or 0)
    selected_chunk = int(plan.get("selected_chunk_size") or 0)
    per_chunk_estimate = float(plan.get("estimated_per_chunk_cost_with_selected_chunk_size_usd") or 0)
    canonical_batch_valid = bool(plan.get("canonical_batch_valid")) and int(plan.get("request_count") or 0) == REQUEST_COUNT_TOTAL
    total_within = total_estimate <= AUTHORIZED_TOTAL_CAP
    chunks_within = selected_chunk > 0 and per_chunk_estimate <= PER_CHUNK_CAP
    live_constants_ok = (
        float(live.CAP_TOTAL) == AUTHORIZED_TOTAL_CAP
        and float(live.PREVIOUS_CAP_TOTAL) == PREVIOUS_TOTAL_CAP
        and float(live.CAP_PER_CHUNK) == PER_CHUNK_CAP
        and int(live.MAX_OUTPUT_TOKENS) == MAX_OUTPUT_TOKENS
    )
    redaction_ok = private_path_leaks == 0 and secret_leaks == 0
    ready = bool(
        canonical_batch_valid
        and credential_ok
        and total_within
        and chunks_within
        and flags["checkpoint_resume_preserved"]
        and flags["failed_evidence_preservation_preserved"]
        and redaction_ok
        and live_constants_ok
    )
    safety_passed = bool(
        live_constants_ok
        and flags["adaptive_chunk_size_enabled"]
        and flags["checkpoint_resume_preserved"]
        and flags["failed_evidence_preservation_preserved"]
        and flags["stop_on_first_failure_preserved"]
        and flags["live_gate_scoped_per_chunk_preserved"]
    )

    return {
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
        "target_model": live.TARGET_MODEL,
        "max_output_tokens": live.MAX_OUTPUT_TOKENS,
        "previous_authorized_hard_cost_cap_total_usd": PREVIOUS_TOTAL_CAP,
        "authorized_hard_cost_cap_total_usd": live.CAP_TOTAL,
        "hard_cost_cap_per_chunk_usd": live.CAP_PER_CHUNK,
        "estimated_total_cost_with_new_output_ceiling_usd": round(total_estimate, 6),
        "estimated_total_within_authorized_cap": total_within,
        "estimated_per_chunk_cost_with_selected_chunk_size_usd": round(per_chunk_estimate, 6),
        "estimated_chunks_within_per_chunk_cap": chunks_within,
        "selected_chunk_size": selected_chunk,
        "adaptive_chunk_size_enabled": flags["adaptive_chunk_size_enabled"],
        "canonical_batch_valid": canonical_batch_valid,
        "request_count_total": int(plan.get("request_count") or 0),
        "credential_preflight_passed": bool(credential_ok),
        "credential_status_category": credential_category,
        "checkpoint_resume_preserved": flags["checkpoint_resume_preserved"],
        "failed_evidence_preservation_preserved": flags["failed_evidence_preservation_preserved"],
        "public_report_redaction_preserved": redaction_ok,
        "ready_to_resume_17c_r2_live": ready,
        "requires_new_cost_authorization": False,
        "private_responses_committed": False,
        "parsed_private_responses_committed": False,
        "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False,
        "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False,
        "credential_or_token_written_to_repo": False,
        "private_paths_redacted_from_public_reports": private_path_leaks == 0,
        "public_report_phi_leak_count": phi_leaks,
        "private_filename_path_leaks_after": private_path_leaks,
        "secret_leaks_after": secret_leaks,
        "r10_ready_to_resume_before_cap_update": bool(r10_summary.get("ready_to_resume_17c_r2_live")),
        "r10_requires_new_cost_authorization_before_cap_update": bool(r10_summary.get("requires_new_cost_authorization")),
        "r10_estimated_total_within_old_cap": bool(r10_summary.get("estimated_total_within_authorized_cap")),
        "live_runner_constants_ok": live_constants_ok,
        "stop_on_first_failure_preserved": flags["stop_on_first_failure_preserved"],
        "live_gate_scoped_per_chunk_preserved": flags["live_gate_scoped_per_chunk_preserved"],
        "privacy_result": "passed" if phi_leaks == 0 and private_path_leaks == 0 and secret_leaks == 0 else "blocked",
        "safety_result": "passed" if safety_passed else "blocked",
    }


def _write_docs(summary: dict[str, Any]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / DOC_NAMES[0]).write_text(
        "# MEDAI 17C-R2 Authorized Total Cap $10 Update Local Only R11\n\n"
        "R11 records the explicit total hard-cost cap update from $0.40 to $10.00 for the full canonical 478-request corpus extraction. "
        "This block is local/config/preflight only. It does not run live extraction, send provider requests, call billing APIs, set a live gate, open MKB, write MKB records, auto-accept, or make a medical decision.\n",
        encoding="utf-8",
    )
    (DOC_DIR / DOC_NAMES[1]).write_text(
        "# MEDAI 17C-R2 Cost Authorization Policy R11\n\n"
        f"- Previous total cap: `${summary['previous_authorized_hard_cost_cap_total_usd']}`\n"
        f"- New authorized total cap: `${summary['authorized_hard_cost_cap_total_usd']}`\n"
        f"- Per-chunk cap remains: `${summary['hard_cost_cap_per_chunk_usd']}`\n"
        f"- Estimated total at maxOutputTokens {summary['max_output_tokens']}: `${summary['estimated_total_cost_with_new_output_ceiling_usd']}`\n"
        f"- Estimated total within new cap: `{summary['estimated_total_within_authorized_cap']}`\n"
        "No billing API was called; this is a local estimate and authorization record.\n",
        encoding="utf-8",
    )
    (DOC_DIR / DOC_NAMES[2]).write_text(
        "# MEDAI 17C-R2 Full Corpus Live Entry Gate After R11\n\n"
        f"- ready_to_resume_17c_r2_live: `{summary['ready_to_resume_17c_r2_live']}`\n"
        f"- canonical_batch_valid: `{summary['canonical_batch_valid']}`\n"
        f"- credential_preflight_passed: `{summary['credential_preflight_passed']}`\n"
        f"- checkpoint_resume_preserved: `{summary['checkpoint_resume_preserved']}`\n"
        f"- failed_evidence_preservation_preserved: `{summary['failed_evidence_preservation_preserved']}`\n"
        f"- public_report_redaction_preserved: `{summary['public_report_redaction_preserved']}`\n"
        "Future live extraction still requires a separate operator command. R11 does not start 17C live execution.\n",
        encoding="utf-8",
    )


def _write_reports(summary: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(REPORT_DIR / "summary.json", summary)
    _write_json(
        REPORT_DIR / "cost_cap_update_public.json",
        {
            "previous_authorized_hard_cost_cap_total_usd": summary["previous_authorized_hard_cost_cap_total_usd"],
            "authorized_hard_cost_cap_total_usd": summary["authorized_hard_cost_cap_total_usd"],
            "hard_cost_cap_per_chunk_usd": summary["hard_cost_cap_per_chunk_usd"],
            "estimated_total_cost_with_new_output_ceiling_usd": summary["estimated_total_cost_with_new_output_ceiling_usd"],
            "estimated_total_within_authorized_cap": summary["estimated_total_within_authorized_cap"],
            "estimated_per_chunk_cost_with_selected_chunk_size_usd": summary["estimated_per_chunk_cost_with_selected_chunk_size_usd"],
            "estimated_chunks_within_per_chunk_cap": summary["estimated_chunks_within_per_chunk_cap"],
            "requires_new_cost_authorization": summary["requires_new_cost_authorization"],
            "billing_api_call_made": False,
        },
    )
    (REPORT_DIR / "live_entry_gate_public.md").write_text(
        "# 17C-R2 R11 live entry gate\n\n"
        f"- ready_to_resume_17c_r2_live: `{summary['ready_to_resume_17c_r2_live']}`\n"
        f"- canonical_batch_valid: `{summary['canonical_batch_valid']}`\n"
        f"- request_count_total: `{summary['request_count_total']}`\n"
        f"- credential_preflight_passed: `{summary['credential_preflight_passed']}`\n"
        f"- estimated_total_within_authorized_cap: `{summary['estimated_total_within_authorized_cap']}`\n"
        f"- estimated_chunks_within_per_chunk_cap: `{summary['estimated_chunks_within_per_chunk_cap']}`\n"
        "- live_extraction_started: `false`\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "safety_boundary_public.md").write_text(
        "# 17C-R2 R11 safety boundary\n\n"
        "- provider_model_call_made: `false`\n"
        "- billing_api_call_made: `false`\n"
        "- live_gate_set: `false`\n"
        "- live_extraction_started: `false`\n"
        "- active_mkb_write: `false`\n"
        "- auto_accept_enabled: `false`\n"
        "- medical_decision_made: `false`\n"
        "- private responses/tokenized payloads/token maps/private identifiers committed: `false`\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "implementation_report.md").write_text(
        f"# {BLOCK}\n\n"
        f"- Result: `{summary['privacy_result']}` / `{summary['safety_result']}`\n"
        f"- Previous total cap: `{summary['previous_authorized_hard_cost_cap_total_usd']}`\n"
        f"- New authorized total cap: `{summary['authorized_hard_cost_cap_total_usd']}`\n"
        f"- Per-chunk cap: `{summary['hard_cost_cap_per_chunk_usd']}`\n"
        f"- maxOutputTokens: `{summary['max_output_tokens']}`\n"
        f"- Estimated total: `{summary['estimated_total_cost_with_new_output_ceiling_usd']}`\n"
        f"- Selected chunk size: `{summary['selected_chunk_size']}`\n"
        f"- Credential preflight passed: `{summary['credential_preflight_passed']}`\n"
        f"- Ready to resume 17C-R2 live: `{summary['ready_to_resume_17c_r2_live']}`\n"
        "- No live extraction was started in this block.\n",
        encoding="utf-8",
    )


def run() -> dict[str, Any]:
    summary = build_summary()
    _write_docs(summary)
    _write_reports(summary)
    return summary


def main() -> int:
    summary = run()
    result = "PASS" if summary["privacy_result"] == "passed" and summary["safety_result"] == "passed" else "BLOCKED"
    print(f"{BLOCK}_{result}")
    print(
        json.dumps(
            {
                "authorized_hard_cost_cap_total_usd": summary["authorized_hard_cost_cap_total_usd"],
                "hard_cost_cap_per_chunk_usd": summary["hard_cost_cap_per_chunk_usd"],
                "estimated_total_cost_with_new_output_ceiling_usd": summary["estimated_total_cost_with_new_output_ceiling_usd"],
                "estimated_total_within_authorized_cap": summary["estimated_total_within_authorized_cap"],
                "estimated_per_chunk_cost_with_selected_chunk_size_usd": summary["estimated_per_chunk_cost_with_selected_chunk_size_usd"],
                "selected_chunk_size": summary["selected_chunk_size"],
                "credential_preflight_passed": summary["credential_preflight_passed"],
                "ready_to_resume_17c_r2_live": summary["ready_to_resume_17c_r2_live"],
                "provider_model_call_made": summary["provider_model_call_made"],
                "billing_api_call_made": summary["billing_api_call_made"],
                "live_extraction_started": summary["live_extraction_started"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
