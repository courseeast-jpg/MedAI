"""Durable private checkpoint + failed-evidence preservation for the 17C-R2 live batch
(17C-R2-R8).

Local-only. This module performs NO provider/network/billing/model calls. Every durable
artifact lives OUTSIDE the repository:

  - checkpoint state under  %LOCALAPPDATA%\\MedAI_Private\\ai_extraction_17C_R2_478_live_checkpoint
  - preserved failure evidence under  %USERPROFILE%\\Downloads\\MedAI_17C_R2_FAILED_EVIDENCE_PRESERVE_PRIVATE

Raw/parsed response bodies are copied ONLY into the private folders above; they are never
returned to the caller for public reporting. Public reports may carry only status,
counts, hashes, failure categories, and the preservation path.

Design goals (R8):
  1. Per-request durable checkpoint so a live retry resumes from the next *unsent*
     request and never re-sends an already-completed document.
  2. Immediate copy of failed-response evidence out of the volatile live-staging folder
     (an external writer has cleared MedAI_Private staging twice), so failed bodies
     survive for later local triage.
  3. A strict resume policy that BLOCKS on canonical-batch SHA256 mismatch, on an
     unresolved failed document, or on an inconsistent checkpoint.

All JSON is written with ensure_ascii=True; JSONL uses physical "\\n" framing only
(consistent with execution/jsonl_framing.py), to avoid Unicode line-separator artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

# ---- Durable locations (outside the repo) --------------------------------------------
CHECKPOINT_DIR = Path(os.path.expandvars(
    r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17C_R2_478_live_checkpoint"))
EVIDENCE_DIR = Path(os.path.expandvars(
    r"%USERPROFILE%\Downloads\MedAI_17C_R2_FAILED_EVIDENCE_PRESERVE_PRIVATE"))
LIVE_STAGING_DIR = Path(os.path.expandvars(
    r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17C_R2_478_live_batch"))

CHECKPOINT_DIR_LABEL = r"C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17C_R2_478_live_checkpoint"
EVIDENCE_DIR_LABEL = r"C:\Users\S1\Downloads\MedAI_17C_R2_FAILED_EVIDENCE_PRESERVE_PRIVATE"

# Checkpoint file names (all *_private; none ever committed).
STATE_FILE = "checkpoint_state_private.json"
COMPLETED_FILE = "checkpoint_completed_doc_ids_private.json"
FAILED_FILE = "checkpoint_failed_doc_private.json"
CHUNK_STATUS_FILE = "checkpoint_chunk_status_private.jsonl"
PROVIDER_TRACE_FILE = "checkpoint_provider_trace_private.jsonl"
SCHEMA_VALIDATION_FILE = "checkpoint_schema_validation_private.jsonl"

CHECKPOINT_FILES = (STATE_FILE, COMPLETED_FILE, FAILED_FILE, CHUNK_STATUS_FILE,
                    PROVIDER_TRACE_FILE, SCHEMA_VALIDATION_FILE)

# Private staging artifacts copied into the evidence folder on failure.
STAGING_EVIDENCE_FILES = (
    "live_responses_private.jsonl",       # failed raw provider response (+ prior raw)
    "parsed_responses_private.jsonl",     # parsed response if available
    "schema_validation_private.json",     # failed doc metadata + per-doc validation
    "provider_trace_private.json",        # provider trace
    "stopped_on_failure_private.json",    # failure stage/category
)

_README_NAME = "README_PRIVATE_DO_NOT_SHARE.txt"


# ---- Low-level helpers ----------------------------------------------------------------
def _base(base: "str | Path | None" = None) -> Path:
    return Path(base) if base is not None else CHECKPOINT_DIR


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=True) + "\n")


def sha256_file(path: "str | Path") -> str:
    """Stream a SHA256 over the canonical batch file (no full read into memory)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_outside_repo(path: "str | Path", repo_root: "str | Path") -> bool:
    """True iff `path` is neither the repo root nor any descendant of it."""
    p = Path(os.path.expandvars(str(path)))
    rr = Path(repo_root)
    try:
        p = p.resolve()
        rr = rr.resolve()
    except OSError:
        pass
    return rr != p and rr not in p.parents


# ---- Checkpoint readers ---------------------------------------------------------------
def load_state(base: "str | Path | None" = None) -> "dict | None":
    f = _base(base) / STATE_FILE
    if not f.is_file():
        return None
    try:
        obj = json.loads(f.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {"_inconsistent": True}
    except (OSError, ValueError):
        return {"_inconsistent": True}


def load_completed(base: "str | Path | None" = None) -> list[str]:
    f = _base(base) / COMPLETED_FILE
    if not f.is_file():
        return []
    try:
        obj = json.loads(f.read_text(encoding="utf-8"))
        return [str(x) for x in obj] if isinstance(obj, list) else []
    except (OSError, ValueError):
        return []


def load_failed(base: "str | Path | None" = None) -> "dict | None":
    f = _base(base) / FAILED_FILE
    if not f.is_file():
        return None
    try:
        obj = json.loads(f.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


# ---- Checkpoint lifecycle -------------------------------------------------------------
def init_checkpoint(run_id: str, batch_sha: str, model: str, cap_total: float,
                    cap_chunk: float, total: int,
                    base: "str | Path | None" = None) -> dict:
    """Create a fresh checkpoint when none exists for this batch SHA. If a state already
    exists for the SAME batch SHA, it is preserved (resume) and returned unchanged."""
    b = _base(base)
    b.mkdir(parents=True, exist_ok=True)
    existing = load_state(base)
    if existing and not existing.get("_inconsistent") and \
            existing.get("canonical_batch_sha256") == batch_sha:
        return existing
    state = {
        "run_id": run_id,
        "canonical_batch_sha256": batch_sha,
        "model": model,
        "cap_total_usd": cap_total,
        "cap_per_chunk_usd": cap_chunk,
        "request_count_total": total,
        "request_index": 0,
        "sent_count": 0,
        "succeeded_count": 0,
        "failed_count": 0,
        "completed_doc_count": 0,
        "failed_doc_id": None,
        "chunk_index": -1,
        "chunk_status": "not_started",
        "status": "initialized",
    }
    _write_json(b / STATE_FILE, state)
    if not (b / COMPLETED_FILE).is_file():
        _write_json(b / COMPLETED_FILE, [])
    return state


def _first_unsent_index(order: "list[str]", completed: "set[str]") -> int:
    for i, doc_id in enumerate(order):
        if doc_id not in completed:
            return i
    return len(order)


def decide_resume(batch_sha: str, order: "list[str]", total: int,
                  base: "str | Path | None" = None) -> "tuple[int, bool, str, set[str]]":
    """Resume policy. Returns (start_index, blocked, reason, completed_set).

    - No checkpoint           -> (0, False, "no_checkpoint_start_from_first")
    - State unreadable        -> blocked "checkpoint_state_unreadable"
    - Batch SHA256 mismatch   -> blocked "canonical_batch_sha256_mismatch"
    - Inconsistent checkpoint -> blocked "checkpoint_inconsistent_*"
    - Unresolved failed doc   -> blocked "failed_doc_unresolved_requires_triage_or_reset"
                                 (start_index still points at the next unsent request)
    - Otherwise               -> (first_unsent, False, "resume_from_next_unsent_request")
    """
    state = load_state(base)
    if state is None:
        return 0, False, "no_checkpoint_start_from_first", set()
    if state.get("_inconsistent"):
        return 0, True, "checkpoint_state_unreadable", set()
    if state.get("canonical_batch_sha256") != batch_sha:
        return 0, True, "canonical_batch_sha256_mismatch", set()

    completed = load_completed(base)
    completed_set = set(completed)
    order_set = set(order)
    if len(completed_set) != len(completed):
        return 0, True, "checkpoint_inconsistent_duplicate_completed", completed_set
    if not completed_set.issubset(order_set):
        return 0, True, "checkpoint_inconsistent_unknown_completed_doc", completed_set
    if len(completed) > total:
        return 0, True, "checkpoint_inconsistent_completed_exceeds_total", completed_set

    start = _first_unsent_index(order, completed_set)
    failed = load_failed(base)
    if failed and not failed.get("resolved", False):
        return start, True, "failed_doc_unresolved_requires_triage_or_reset", completed_set
    return start, False, "resume_from_next_unsent_request", completed_set


def mark_completed(doc_id: str, index: int, base: "str | Path | None" = None) -> None:
    b = _base(base)
    b.mkdir(parents=True, exist_ok=True)
    completed = load_completed(base)
    if doc_id not in completed:
        completed.append(doc_id)
    _write_json(b / COMPLETED_FILE, completed)
    state = load_state(base) or {}
    state["request_index"] = index + 1
    state["completed_doc_count"] = len(completed)
    state["sent_count"] = int(state.get("sent_count", 0)) + 1
    state["succeeded_count"] = int(state.get("succeeded_count", 0)) + 1
    state["status"] = "in_progress"
    _write_json(b / STATE_FILE, state)
    _append_jsonl(b / SCHEMA_VALIDATION_FILE,
                  {"doc_id": doc_id, "index": index, "schema_valid": True, "reason": "ok"})


def mark_failed(doc_id: str, index: int, category: str,
                base: "str | Path | None" = None) -> None:
    b = _base(base)
    b.mkdir(parents=True, exist_ok=True)
    _write_json(b / FAILED_FILE, {
        "failed_doc_id": doc_id,
        "request_index": index,
        "failure_category": category,
        "resolved": False,
    })
    state = load_state(base) or {}
    state["failed_doc_id"] = doc_id
    state["request_index"] = index
    state["sent_count"] = int(state.get("sent_count", 0)) + 1
    state["failed_count"] = int(state.get("failed_count", 0)) + 1
    state["status"] = "blocked_failed_doc"
    _write_json(b / STATE_FILE, state)
    _append_jsonl(b / SCHEMA_VALIDATION_FILE,
                  {"doc_id": doc_id, "index": index, "schema_valid": False, "reason": category})


def record_chunk_status(chunk_index: int, sent: int, succeeded: int, status: str,
                        base: "str | Path | None" = None) -> None:
    _append_jsonl(_base(base) / CHUNK_STATUS_FILE,
                  {"chunk": chunk_index, "sent": sent, "succeeded": succeeded, "status": status})


def record_provider_trace(category: str, index: int = -1,
                          base: "str | Path | None" = None) -> None:
    """Provider STATUS category only (never a body, token, or credential)."""
    _append_jsonl(_base(base) / PROVIDER_TRACE_FILE,
                  {"index": index, "provider_status_category": category})


def resolve_failed(base: "str | Path | None" = None) -> bool:
    """Operator-explicit clearance of a triaged failed doc (unblocks resume)."""
    failed = load_failed(base)
    if not failed:
        return False
    failed["resolved"] = True
    _write_json(_base(base) / FAILED_FILE, failed)
    state = load_state(base) or {}
    if state:
        state["status"] = "in_progress"
        _write_json(_base(base) / STATE_FILE, state)
    return True


def reset_checkpoint(base: "str | Path | None" = None) -> None:
    """Operator-explicit checkpoint reset (removes all checkpoint files)."""
    b = _base(base)
    for name in CHECKPOINT_FILES:
        try:
            (b / name).unlink()
        except OSError:
            pass


# ---- Failed-evidence preservation -----------------------------------------------------
def preserve_failed_evidence(run_ts: str,
                             staging_dir: "str | Path | None" = None,
                             base: "str | Path | None" = None,
                             evidence_base: "str | Path | None" = None
                             ) -> "tuple[bool, str, int]":
    """Copy private failure evidence into an isolated run folder under the evidence dir.

    Copies (when present): failed raw provider response, parsed response, failed doc
    metadata, schema validation detail, provider trace, and the checkpoint state — then
    writes a sanitized README. The destination is PRIVATE and must never be committed.

    Returns (preserved, run_dir_path, files_copied). `preserved` is True once the run
    folder exists, even if some sources were already cleared by the external writer
    (files_copied then reflects how much survived).
    """
    ev = Path(evidence_base) if evidence_base is not None else EVIDENCE_DIR
    run_dir = ev / f"run_{run_ts}"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False, str(run_dir), 0

    copied = 0
    stg = Path(staging_dir) if staging_dir is not None else LIVE_STAGING_DIR
    for name in STAGING_EVIDENCE_FILES:
        src = stg / name
        if src.is_file():
            try:
                shutil.copy2(src, run_dir / name)
                copied += 1
            except OSError:
                pass
    cb = _base(base)
    for name in CHECKPOINT_FILES:
        src = cb / name
        if src.is_file():
            try:
                shutil.copy2(src, run_dir / ("checkpoint_copy_" + name))
                copied += 1
            except OSError:
                pass

    readme = (
        "MedAI 17C-R2 FAILED-RESPONSE EVIDENCE — PRIVATE, DO NOT SHARE\n"
        "=============================================================\n\n"
        "This folder preserves PRIVATE failure evidence copied out of the volatile live\n"
        "staging area immediately on a 17C-R2 live-batch failure (R8 hardening).\n\n"
        "It MAY contain raw/parsed AI response bodies and tokenized payload fragments.\n"
        "Treat as PRIVATE clinical material:\n"
        "  - Do NOT commit to git.\n"
        "  - Do NOT paste into chat, tickets, or public reports.\n"
        "  - Do NOT email or upload to any external service.\n\n"
        "Public reports reference only: evidence_preserved, preservation_path, failed\n"
        "doc hash, and failure category — never a body, token map, or credential.\n"
    )
    try:
        (run_dir / _README_NAME).write_text(readme, encoding="utf-8")
    except OSError:
        pass
    return True, str(run_dir), copied


__all__ = [
    "CHECKPOINT_DIR", "EVIDENCE_DIR", "LIVE_STAGING_DIR",
    "CHECKPOINT_DIR_LABEL", "EVIDENCE_DIR_LABEL", "CHECKPOINT_FILES",
    "sha256_file", "is_outside_repo",
    "load_state", "load_completed", "load_failed",
    "init_checkpoint", "decide_resume", "mark_completed", "mark_failed",
    "record_chunk_status", "record_provider_trace",
    "resolve_failed", "reset_checkpoint", "preserve_failed_evidence",
]
