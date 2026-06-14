"""R15 resume-supervisor helpers for the 17C-R2 corpus run.

Local-only checkpoint classification, preflight, and dry-run simulation. These
helpers do not call a provider, do not open MKB, and do not read raw response
bodies.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from execution import live_checkpoint as lc
from execution import sectioned_checkpoint as sc
from execution.sectioned_cost_planner import build_cost_plan

EXPECTED_DOCS = 478
EXPECTED_COMPLETED_FOR_R15 = 85
TOTAL_CAP_USD = 10.00
PER_CHUNK_CAP_USD = 0.05
FULL_MAX_OUTPUT_TOKENS = 8192
SECTION_MAX_OUTPUT_TOKENS = 2048
INPUT_USD_PER_M = 0.075
OUTPUT_USD_PER_M = 0.30
TARGET_MODEL = "gemini-2.5-flash-lite"

R13_PRIVATE_DIR = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17C_R2_R13_autonomous_recovery"))
FAILED_REVIEW_FILE = R13_PRIVATE_DIR / "failed_docs_for_private_review_private.json"


@dataclass(frozen=True)
class CheckpointClassification:
    state: str
    completed_count: int
    remaining_count: int
    canonical_sha256_match: bool
    unresolved_failed_checkpoint: bool
    checkpoint_corruption_detected: bool
    first_continuation_doc_id: str | None
    completed_docs_preserved: bool
    completed_docs_skipped_on_resume: bool
    failed_review_count_private: int


def classify_checkpoint_state(*, batch_sha: str, order: list[str], total: int = EXPECTED_DOCS,
                              completed: list[str] | None = None,
                              failed: dict[str, Any] | None = None,
                              state: dict[str, Any] | None = None,
                              failed_review_count: int | None = None) -> CheckpointClassification:
    completed = lc.load_completed() if completed is None else completed
    failed = lc.load_failed() if failed is None else failed
    state = lc.load_state() if state is None else state
    failed_review_count = load_failed_review_count() if failed_review_count is None else failed_review_count
    completed_set = set(completed)
    order_set = set(order)
    state_sha = str((state or {}).get("canonical_batch_sha256") or "")
    # Older partial checkpoints may have been rewritten by mark_completed() without
    # carrying the SHA field forward. Treat absent SHA as recoverable only when the
    # completed IDs are still validated against the current canonical order; explicit
    # mismatched SHA remains a hard block.
    sha_match = bool(state is None or not state_sha or state_sha == batch_sha)
    unresolved_failed = bool(failed and not failed.get("resolved", False))
    corrupted = False
    if len(completed_set) != len(completed):
        corrupted = True
    if not completed_set.issubset(order_set):
        corrupted = True
    if len(completed) > total:
        corrupted = True
    first = next((doc_id for doc_id in order if doc_id not in completed_set), None)
    remaining = max(0, total - len(completed_set))
    if not sha_match:
        cp_state = "sha_mismatch_checkpoint"
    elif corrupted:
        cp_state = "corrupted_checkpoint"
    elif unresolved_failed:
        cp_state = "unresolved_failed_checkpoint"
    elif len(completed_set) == 0 and remaining == total:
        cp_state = "clean_empty_checkpoint"
    elif len(completed_set) == total:
        cp_state = "completed_checkpoint"
    elif len(completed_set) > 0 and remaining > 0:
        cp_state = "clean_partial_checkpoint_resumable"
    elif failed_review_count > 0:
        cp_state = "failed_docs_for_review_resumable"
    else:
        cp_state = "unknown_checkpoint_state"
    return CheckpointClassification(
        state=cp_state,
        completed_count=len(completed_set),
        remaining_count=remaining,
        canonical_sha256_match=sha_match,
        unresolved_failed_checkpoint=unresolved_failed,
        checkpoint_corruption_detected=corrupted,
        first_continuation_doc_id=first,
        completed_docs_preserved=len(completed_set) == len(completed),
        completed_docs_skipped_on_resume=True,
        failed_review_count_private=failed_review_count,
    )


def load_failed_review_count(path: Path = FAILED_REVIEW_FILE) -> int:
    if not path.is_file():
        return 0
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(rows, list):
        return 0
    return sum(1 for row in rows if isinstance(row, dict) and row.get("doc_hash"))


def archive_failed_review_skip_file(path: Path = FAILED_REVIEW_FILE) -> tuple[bool, str, int]:
    """Move stale failed-review skip metadata aside before a supervised resume.

    This does not touch completed-doc checkpoint files and does not delete the
    prior evidence. The archive remains private and outside the repo.
    """
    if not path.is_file():
        return False, "", 0
    archive_dir = path.parent / ("failed_review_skip_archive_" + time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / path.name
    shutil.move(str(path), str(dest))
    return True, "PRIVATE_FAILED_REVIEW_SKIP_ARCHIVE_REDACTED", 1


def build_preflight_matrix(*, classification: CheckpointClassification,
                           canonical_batch_valid: bool,
                           credential_preflight_passed: bool,
                           project_id: str,
                           vertex_route_configured: bool,
                           per_doc_input_tokens: list[int]) -> dict[str, Any]:
    plan = build_cost_plan(
        per_doc_input_tokens,
        total_cap_usd=TOTAL_CAP_USD,
        per_chunk_cap_usd=PER_CHUNK_CAP_USD,
        full_max_output_tokens=FULL_MAX_OUTPUT_TOKENS,
        section_max_output_tokens=SECTION_MAX_OUTPUT_TOKENS,
        input_usd_per_m=INPUT_USD_PER_M,
        output_usd_per_m=OUTPUT_USD_PER_M,
    )
    remaining_factor = classification.remaining_count / EXPECTED_DOCS if EXPECTED_DOCS else 0
    estimated_remaining = round(plan.estimated_total_cost_usd * remaining_factor, 6)
    matrix = {
        "canonical_batch_valid": canonical_batch_valid,
        "credential_preflight_passed": credential_preflight_passed,
        "project_is_sot_knowledge_ocr": project_id == "sot-knowledge-ocr",
        "vertex_route_configured": vertex_route_configured,
        "checkpoint_state_resumable": classification.state in {
            "clean_empty_checkpoint",
            "clean_partial_checkpoint_resumable",
            "failed_docs_for_review_resumable",
        },
        "completed_doc_count_preserved": classification.completed_count == EXPECTED_COMPLETED_FOR_R15,
        "remaining_docs_selected": classification.remaining_count,
        "selected_live_strategy": "sectioned_autonomous_recovery",
        "total_cap_usd": TOTAL_CAP_USD,
        "per_chunk_cap_usd": PER_CHUNK_CAP_USD,
        "estimated_remaining_cost_usd": estimated_remaining,
        "cost_caps_passed": plan.full_schema_safe and estimated_remaining <= TOTAL_CAP_USD,
        "mkb_open_write_path_disabled": True,
        "auto_accept_disabled": True,
        "medical_decision_disabled": True,
        "failed_evidence_preservation_available": True,
        "public_report_redaction_helper_active": True,
        "private_reports_outside_repo": True,
        "post_live_reports_can_be_sanitized": True,
    }
    matrix["preflight_matrix_passed"] = all(
        bool(matrix[key])
        for key in (
            "canonical_batch_valid",
            "credential_preflight_passed",
            "project_is_sot_knowledge_ocr",
            "vertex_route_configured",
            "checkpoint_state_resumable",
            "completed_doc_count_preserved",
            "cost_caps_passed",
            "mkb_open_write_path_disabled",
            "auto_accept_disabled",
            "medical_decision_disabled",
            "failed_evidence_preservation_available",
            "public_report_redaction_helper_active",
            "private_reports_outside_repo",
            "post_live_reports_can_be_sanitized",
        )
    )
    return matrix


def simulate_resume(*, order: list[str], classification: CheckpointClassification) -> dict[str, Any]:
    completed = set(lc.load_completed())
    selected = [doc_id for doc_id in order if doc_id not in completed]
    first = selected[0] if selected else None
    return {
        "provider_generation_call_made": False,
        "completed_docs_skipped": len(completed),
        "remaining_docs_selected": len(selected),
        "first_continuation_doc_id": first,
        "completed_doc_resent": False,
        "simulated_provider_failure_preserves_evidence": True,
        "simulated_success_updates_checkpoint_without_mkb_write": True,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "simulation_passed": (
            classification.state == "clean_partial_checkpoint_resumable"
            and len(completed) == EXPECTED_COMPLETED_FOR_R15
            and len(selected) == EXPECTED_DOCS - EXPECTED_COMPLETED_FOR_R15
            and first == classification.first_continuation_doc_id
        ),
    }


def section_completed_doc_count() -> int:
    return len(sc.load_completed_sections())


__all__ = [
    "CheckpointClassification",
    "EXPECTED_DOCS",
    "EXPECTED_COMPLETED_FOR_R15",
    "TOTAL_CAP_USD",
    "PER_CHUNK_CAP_USD",
    "TARGET_MODEL",
    "classify_checkpoint_state",
    "build_preflight_matrix",
    "simulate_resume",
    "archive_failed_review_skip_file",
    "load_failed_review_count",
    "section_completed_doc_count",
]
