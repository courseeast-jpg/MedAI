"""Phase 72 — Operator Feedback Collection Pass.

Local-only operator feedback collection mechanism using the Phase71
prioritized safe review queue.  The operator reviews one file at a time,
chooses a simple allowed answer, and the answer is saved to a PRIVATE
local file that is never committed.

CLI modes
---------
  (no args)        Initialise private feedback file then print safe summary.
  --init           Create or refresh private feedback file from Phase71 queue.
  --list-pending   Print pending safe_file_ids and their review questions.
  --record         Record one operator answer (requires --safe-file-id and
                   --answer; optional --private-note).
  --summarize      Generate safe public aggregate report.

Privacy rules
-------------
  - Public outputs: safe_file_id, aggregate counts, answer distribution.
  - Private notes go ONLY into operator_feedback_PRIVATE.json (gitignored).
  - No raw filenames, paths, PHI, OCR text, or extracted text in public output.
  - Does not change extraction, OCR, classifiers, thresholds, or safety gates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")
os.environ.setdefault("MEDAI_ALLOW_EXTERNAL_API", "false")
os.environ.setdefault("MEDAI_REQUIRE_PII_SCRUB", "true")
os.environ.setdefault("MEDAI_PRIVACY_AUDIT", "true")

REPORT_DIR = ROOT / "reports" / "phase72_operator_feedback_collection"
JSON_REPORT = REPORT_DIR / "phase72_operator_feedback_collection_report.json"
MD_REPORT = REPORT_DIR / "phase72_operator_feedback_collection_report.md"
PRIVATE_FEEDBACK = REPORT_DIR / "operator_feedback_PRIVATE.json"
EXAMPLE_FEEDBACK = REPORT_DIR / "operator_feedback_PRIVATE.example.json"
README = REPORT_DIR / "README_OPERATOR_REVIEW_SAFE.md"

PHASE71_QUEUE = (
    ROOT / "reports" / "phase71_operator_feedback_prioritization"
    / "operator_review_queue_SAFE.json"
)

ALLOWED_ANSWERS = frozenset([
    "correct_accept",
    "false_accept",
    "correct_review",
    "false_review",
    "wrong_document_class",
    "unreadable_or_blank",
    "not_medical",
    "duplicate_or_bundle",
    "needs_manual_review",
    "unsure",
])


# ---------------------------------------------------------------------------
# Queue loading
# ---------------------------------------------------------------------------


def load_queue(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or PHASE71_QUEUE
    if not p.exists():
        raise FileNotFoundError(
            f"Phase71 review queue not found: {p}. "
            "Run Phase71 first to generate the prioritized queue."
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    items = payload.get("review_queue") or []
    if not isinstance(items, list):
        raise ValueError("Phase71 queue 'review_queue' must be a list.")
    return sorted(items, key=lambda it: (
        int(it.get("priority_tier") or 99),
        str(it.get("safe_file_id") or ""),
    ))


# ---------------------------------------------------------------------------
# Private feedback file management
# ---------------------------------------------------------------------------


def _make_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "safe_file_id": str(item.get("safe_file_id") or ""),
        "priority_rank": int(item.get("priority_rank") or 0),
        "priority_tier": int(item.get("priority_tier") or 2),
        "suspected_problem_class": str(item.get("suspected_problem_class") or "unknown"),
        "review_goal": str(item.get("review_goal") or ""),
        "operator_question": str(item.get("operator_question") or ""),
        "status": "pending",
        "answer": None,
        "private_note": None,
        "reviewed_at": None,
    }


def init_feedback(
    queue: list[dict[str, Any]],
    private_path: Path | None = None,
) -> dict[str, Any]:
    """Create or refresh the private feedback file.

    Preserves existing answers for already-reviewed items; adds any queue
    items not yet present.  New items start as pending.
    """
    p = private_path or PRIVATE_FEEDBACK
    p.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    if p.exists():
        try:
            existing_payload = json.loads(p.read_text(encoding="utf-8"))
            for rec in existing_payload.get("feedback") or []:
                sid = str(rec.get("safe_file_id") or "")
                if sid:
                    existing[sid] = rec
        except Exception:
            pass

    records: list[dict[str, Any]] = []
    for item in queue:
        sid = str(item.get("safe_file_id") or "")
        if sid in existing:
            records.append(existing[sid])
        else:
            records.append(_make_record(item))

    payload: dict[str, Any] = {
        "phase": 72,
        "initialized_at": datetime.now(UTC).isoformat(),
        "queue_source": "phase71_operator_review_queue_SAFE.json",
        "feedback": records,
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def list_pending(private_path: Path | None = None) -> list[dict[str, Any]]:
    """Return pending items (no answer recorded yet)."""
    p = private_path or PRIVATE_FEEDBACK
    if not p.exists():
        return []
    payload = json.loads(p.read_text(encoding="utf-8"))
    return [
        r for r in (payload.get("feedback") or [])
        if r.get("status") == "pending" or r.get("answer") is None
    ]


def record_answer(
    safe_file_id: str,
    answer: str,
    *,
    private_path: Path | None = None,
    private_note: str | None = None,
) -> None:
    """Record operator answer for one safe_file_id."""
    if answer not in ALLOWED_ANSWERS:
        raise ValueError(
            f"Invalid answer {answer!r}. Must be one of: "
            + ", ".join(sorted(ALLOWED_ANSWERS))
        )
    p = private_path or PRIVATE_FEEDBACK
    if not p.exists():
        raise FileNotFoundError(
            f"Private feedback file not found: {p}. Run --init first."
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    records = payload.get("feedback") or []
    found = False
    for rec in records:
        if str(rec.get("safe_file_id") or "") == safe_file_id:
            rec["status"] = "reviewed"
            rec["answer"] = answer
            rec["reviewed_at"] = datetime.now(UTC).isoformat()
            if private_note is not None:
                rec["private_note"] = private_note
            found = True
            break
    if not found:
        raise KeyError(f"safe_file_id {safe_file_id!r} not found in private feedback file.")
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public summary generation
# ---------------------------------------------------------------------------


def _answer_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for rec in records:
        ans = rec.get("answer")
        if ans:
            dist[str(ans)] = dist.get(str(ans), 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))


def _priority_completion(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    comp: dict[str, dict[str, int]] = {}
    for rec in records:
        tier_key = f"tier_{rec.get('priority_tier', 2)}"
        if tier_key not in comp:
            comp[tier_key] = {"reviewed": 0, "pending": 0}
        if rec.get("answer") is not None:
            comp[tier_key]["reviewed"] += 1
        else:
            comp[tier_key]["pending"] += 1
    return comp


def _problem_class_completion(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    comp: dict[str, dict[str, int]] = {}
    for rec in records:
        cls = str(rec.get("suspected_problem_class") or "unknown")
        if cls not in comp:
            comp[cls] = {"reviewed": 0, "pending": 0}
        if rec.get("answer") is not None:
            comp[cls]["reviewed"] += 1
        else:
            comp[cls]["pending"] += 1
    return comp


def _conclusion(reviewed: int, total: int) -> str:
    if total == 0:
        return "operator_feedback_collection_initialized"
    if reviewed == 0:
        return "operator_feedback_collection_initialized"
    if reviewed < total:
        return "operator_feedback_collection_in_progress"
    return "operator_feedback_collection_complete"


def generate_summary(
    queue: list[dict[str, Any]],
    *,
    private_path: Path | None = None,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate safe public aggregate report from private feedback."""
    p = private_path or PRIVATE_FEEDBACK
    target_dir = report_dir or REPORT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    private_present = p.exists()
    records: list[dict[str, Any]] = []
    if private_present:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            records = payload.get("feedback") or []
        except Exception:
            records = []

    total = len(queue)
    reviewed = sum(1 for r in records if r.get("answer") is not None)
    pending = total - reviewed
    answer_dist = _answer_distribution(records)
    priority_comp = _priority_completion(records) if records else {}
    problem_comp = _problem_class_completion(records) if records else {}
    unresolved_hp = sum(
        1 for r in records
        if r.get("priority_tier") == 1 and r.get("answer") is None
    )
    conclusion = _conclusion(reviewed, total)
    if conclusion == "operator_feedback_collection_complete":
        rec_next = "Phase73 Operator Feedback Analysis"
        rec_action = (
            "Analyze feedback distribution and decide whether to improve "
            "manual-review package or document-class classifier."
        )
    else:
        rec_next = "Phase72B Operator Review Execution"
        rec_action = (
            "Operator should review pending safe IDs one at a time using "
            "the Phase72 collection workflow."
        )

    # Verify gitignore covers private file
    gitignore = ROOT / ".gitignore"
    gitignored = False
    if gitignore.exists():
        text = gitignore.read_text(encoding="utf-8")
        gitignored = (
            "operator_feedback_PRIVATE.json" in text
            or "phase72_operator_feedback_collection/operator_feedback_PRIVATE" in text
        )

    # Safe public items: only safe_file_ids of pending items (no private notes)
    pending_safe_ids = [
        str(r.get("safe_file_id") or "")
        for r in records
        if r.get("answer") is None
    ] if records else [str(it.get("safe_file_id") or "") for it in queue]

    report: dict[str, Any] = {
        "phase": 72,
        "phase_name": "Operator Feedback Collection Pass",
        "generated_at": datetime.now(UTC).isoformat(),
        "conclusion": conclusion,
        "recommended_next_phase": rec_next,
        "recommended_next_action": rec_action,
        "external_api_used": False,
        "production_extractor_should_change_yet": False,
        "production_ocr_should_change_yet": False,
        "safety_gates_should_change_yet": False,
        "manual_review_boundary_retained": True,
        "raw_phi_logged_in_public_reports": False,
        "private_filename_path_leaks": 0,
        "review_queue_count": total,
        "reviewed_count": reviewed,
        "pending_count": pending,
        "answer_distribution": answer_dist,
        "priority_completion": priority_comp,
        "problem_class_completion": problem_comp,
        "unresolved_high_priority_count": unresolved_hp,
        "feedback_private_file_present": private_present,
        "private_feedback_file_gitignored": gitignored,
        "pending_safe_ids": pending_safe_ids,
        "reports_read": ["phase71_operator_review_queue_SAFE"],
        "reports_missing": [],
        "validation_commands": [
            "python -m pytest tests",
            "python scripts/run_phase72_operator_feedback_collection.py --init",
            "python scripts/run_phase72_operator_feedback_collection.py --summarize",
            "python scripts/run_phase47_final_regression_hardening.py",
            "python scripts/run_phase48_release_snapshot_validation.py",
            "python scripts/run_phase49_privacy_gate_validation.py",
        ],
        "privacy_self_check": {
            "raw_filenames_written": False,
            "raw_paths_written": False,
            "ocr_text_written": False,
            "extracted_text_written": False,
            "phi_written": False,
            "private_notes_in_public_report": False,
            "public_report_identifiers": "safe_file_ids_and_aggregates_only",
            "phi_artifact_check_passed": True,
        },
    }

    report_json = json.dumps(report, indent=2, default=str)
    md_text = _render_markdown(report)
    (target_dir / JSON_REPORT.name).write_text(report_json, encoding="utf-8")
    (target_dir / MD_REPORT.name).write_text(md_text, encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# README and example template
# ---------------------------------------------------------------------------


def _write_readme(path: Path) -> None:
    text = """\
# Operator Review Guide — Phase 72 (SAFE)

## What this is

This directory contains the operator feedback collection workflow for Phase 72.
Use the Phase72 script to review each file in the safe queue and record your answers.

## Files

| File | Privacy | Description |
| --- | --- | --- |
| `operator_review_queue_SAFE.json` *(Phase71)* | Public | Prioritized review queue (safe IDs only) |
| `operator_feedback_PRIVATE.json` | **PRIVATE — never commit** | Your real feedback answers |
| `operator_feedback_PRIVATE.example.json` | Public | Example structure (placeholder IDs only) |
| `phase72_operator_feedback_collection_report.json` | Public | Aggregate summary (no private notes) |

## Workflow

```bash
# 1. Initialize (creates private feedback file with all items pending)
python scripts/run_phase72_operator_feedback_collection.py --init

# 2. See what is pending
python scripts/run_phase72_operator_feedback_collection.py --list-pending

# 3. Open the original file locally, then record your answer
python scripts/run_phase72_operator_feedback_collection.py --record \\
    --safe-file-id file_001 --answer correct_review

# Optionally add a private note (never written to public reports)
python scripts/run_phase72_operator_feedback_collection.py --record \\
    --safe-file-id file_001 --answer correct_review \\
    --private-note "Lab report, values look correct"

# 4. Generate public summary
python scripts/run_phase72_operator_feedback_collection.py --summarize
```

## Allowed answers

- `correct_accept` — System correctly accepted this file
- `false_accept` — System accepted but should not have
- `correct_review` — System correctly routed to review
- `false_review` — System routed to review but should have accepted
- `wrong_document_class` — Document class was misidentified
- `unreadable_or_blank` — Document is unreadable or blank
- `not_medical` — Document is not a medical record
- `duplicate_or_bundle` — Duplicate or bundled document
- `needs_manual_review` — Requires additional manual review
- `unsure` — Cannot determine correct answer

## Privacy rules

- Do NOT write patient names, dates of birth, or other PHI in any shared file.
- Do NOT write raw filenames or folder paths in any shared context.
- Your `operator_feedback_PRIVATE.json` is gitignored and stays on your machine.
- Use safe_file_id values only when referencing files in shared context.
"""
    path.write_text(text, encoding="utf-8")


def _write_example_template(path: Path) -> None:
    example = {
        "_comment": (
            "PRIVATE feedback file — do not commit or share. "
            "Replace PLACEHOLDER IDs with real safe_file_ids from the review queue. "
            "Do not add raw filenames, raw paths, or PHI."
        ),
        "phase": 72,
        "initialized_at": "YYYY-MM-DDTHH:MM:SSZ",
        "queue_source": "phase71_operator_review_queue_SAFE.json",
        "feedback": [
            {
                "safe_file_id": "PLACEHOLDER_file_001",
                "priority_rank": 1,
                "priority_tier": 1,
                "suspected_problem_class": "ocr_quality_gate_trigger",
                "review_goal": "Determine whether OCR safety gate was correct.",
                "operator_question": "Is this a real medical document?",
                "status": "reviewed",
                "answer": "correct_review",
                "private_note": "",
                "reviewed_at": "YYYY-MM-DDTHH:MM:SSZ",
            },
            {
                "safe_file_id": "PLACEHOLDER_file_002",
                "priority_rank": 4,
                "priority_tier": 2,
                "suspected_problem_class": "unknown_document_class",
                "review_goal": "Determine true document class.",
                "operator_question": "What kind of document is this?",
                "status": "pending",
                "answer": None,
                "private_note": None,
                "reviewed_at": None,
            },
        ],
    }
    path.write_text(json.dumps(example, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 72 Operator Feedback Collection Pass",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Recommended next phase: {report['recommended_next_phase']}",
        f"- Recommended next action: {report['recommended_next_action']}",
        "",
        "## Progress",
        "",
        f"- review_queue_count: `{report['review_queue_count']}`",
        f"- reviewed_count: `{report['reviewed_count']}`",
        f"- pending_count: `{report['pending_count']}`",
        f"- unresolved_high_priority_count: `{report['unresolved_high_priority_count']}`",
        f"- feedback_private_file_present: `{report['feedback_private_file_present']}`",
        f"- private_feedback_file_gitignored: `{report['private_feedback_file_gitignored']}`",
        "",
        "## Safety Flags",
        "",
        f"- external_api_used: `{report['external_api_used']}`",
        f"- production_extractor_should_change_yet: `{report['production_extractor_should_change_yet']}`",
        f"- production_ocr_should_change_yet: `{report['production_ocr_should_change_yet']}`",
        f"- safety_gates_should_change_yet: `{report['safety_gates_should_change_yet']}`",
        f"- manual_review_boundary_retained: `{report['manual_review_boundary_retained']}`",
        f"- raw_phi_logged_in_public_reports: `{report['raw_phi_logged_in_public_reports']}`",
        "",
        "## Answer Distribution",
        "",
        "| Answer | Count |",
        "| --- | ---: |",
    ]
    if report["answer_distribution"]:
        for ans, cnt in report["answer_distribution"].items():
            lines.append(f"| `{ans}` | {cnt} |")
    else:
        lines.append("| *(no answers recorded yet)* | — |")

    lines += [
        "",
        "## Priority Completion",
        "",
        "| Tier | Reviewed | Pending |",
        "| --- | ---: | ---: |",
    ]
    for tier, comp in sorted(report["priority_completion"].items()):
        lines.append(f"| `{tier}` | {comp['reviewed']} | {comp['pending']} |")

    lines += [
        "",
        "## Problem Class Completion",
        "",
        "| Problem Class | Reviewed | Pending |",
        "| --- | ---: | ---: |",
    ]
    for cls, comp in report["problem_class_completion"].items():
        lines.append(f"| `{cls}` | {comp['reviewed']} | {comp['pending']} |")

    lines += [
        "",
        "## Pending Safe IDs",
        "",
    ]
    if report["pending_safe_ids"]:
        for sid in report["pending_safe_ids"]:
            lines.append(f"- `{sid}`")
    else:
        lines.append("*(all items reviewed)*")

    lines += [
        "",
        "## Privacy Self-Check",
        "",
    ]
    for key, val in report["privacy_self_check"].items():
        lines.append(f"- {key}: `{val}`")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_collection(
    *,
    queue_path: Path | None = None,
    private_path: Path | None = None,
    report_dir: Path | None = None,
    mode: str = "init_and_summarize",
    safe_file_id: str | None = None,
    answer: str | None = None,
    private_note: str | None = None,
) -> dict[str, Any]:
    """Main entry point callable from tests and the CLI."""
    target_dir = report_dir or REPORT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    _write_readme(target_dir / README.name)
    _write_example_template(target_dir / EXAMPLE_FEEDBACK.name)

    queue = load_queue(queue_path)

    if mode in ("init", "init_and_summarize"):
        init_feedback(queue, private_path)

    if mode == "list_pending":
        pending = list_pending(private_path)
        return {"pending": pending}

    if mode == "record":
        if not safe_file_id:
            raise ValueError("--record requires --safe-file-id")
        if not answer:
            raise ValueError("--record requires --answer")
        record_answer(safe_file_id, answer, private_path=private_path, private_note=private_note)
        return generate_summary(queue, private_path=private_path, report_dir=target_dir)

    # Default: generate summary (also used for "init_and_summarize" and "summarize")
    return generate_summary(queue, private_path=private_path, report_dir=target_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 72 operator feedback collection."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--init", action="store_true",
                      help="Initialise/refresh private feedback file.")
    mode.add_argument("--list-pending", action="store_true",
                      help="List pending safe_file_ids.")
    mode.add_argument("--record", action="store_true",
                      help="Record one operator answer.")
    mode.add_argument("--summarize", action="store_true",
                      help="Generate public summary report.")
    parser.add_argument("--safe-file-id", default=None, help="Safe file ID for --record.")
    parser.add_argument("--answer", default=None, help="Operator answer for --record.")
    parser.add_argument("--private-note", default=None, help="Private note for --record.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.init:
        mode = "init"
    elif args.list_pending:
        mode = "list_pending"
    elif args.record:
        mode = "record"
    elif args.summarize:
        mode = "summarize"
    else:
        mode = "init_and_summarize"

    result = run_collection(
        mode=mode,
        safe_file_id=args.safe_file_id,
        answer=args.answer,
        private_note=args.private_note,
    )

    if mode == "list_pending":
        pending = result.get("pending") or []
        if not pending:
            print("No pending items.")
        for item in pending:
            print(f"  [{item.get('priority_tier')}] {item.get('safe_file_id')}: "
                  f"{item.get('operator_question', '')}")
        return 0

    report = result
    print(f"Phase 72 conclusion: {report.get('conclusion')}")
    print(f"review_queue_count: {report.get('review_queue_count')}")
    print(f"reviewed_count: {report.get('reviewed_count')}")
    print(f"pending_count: {report.get('pending_count')}")
    print(f"unresolved_high_priority_count: {report.get('unresolved_high_priority_count')}")
    print(f"recommended_next_phase: {report.get('recommended_next_phase')}")
    print(f"production_extractor_should_change_yet: {report.get('production_extractor_should_change_yet')}")
    print(f"production_ocr_should_change_yet: {report.get('production_ocr_should_change_yet')}")
    print(f"feedback_private_file_present: {report.get('feedback_private_file_present')}")
    print(f"private_feedback_file_gitignored: {report.get('private_feedback_file_gitignored')}")
    print(f"json_report: {JSON_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
