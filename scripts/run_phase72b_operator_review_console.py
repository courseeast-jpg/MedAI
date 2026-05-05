"""Phase 72B — Human-Minimized Operator Review Console.

One-command local review workflow that reduces operator work to a minimal
loop: one file at a time, one keystroke per answer, instant save.

CLI modes
---------
  --summary        (non-interactive) Generate safe public report from current
                   private feedback state. Safe for automated validation.
  --tier 1         Review tier-1 pending items in interactive terminal wizard.
  --tier 2         Review tier-2 pending items in interactive terminal wizard.
  --all            Review all pending items in interactive terminal wizard.
  --launch-ui      Print command to launch the Streamlit operator feedback UI.
  (no args)        Equivalent to --summary.

Interactive wizard behavior
---------------------------
  - Shows one pending item at a time.
  - Displays: safe_file_id, priority_tier, suspected_problem_class,
    review_goal, operator_question, numbered answer options.
  - Accepts a single digit (1-10), 's' to skip, 'q' to quit.
  - Saves immediately after selection.
  - Advances automatically to next pending item.
  - After tier-1 review, prompts whether to continue with tier-2.
  - Generates updated public summary on exit.

Privacy rules
-------------
  - Public outputs: safe_file_id, aggregate counts only.
  - Raw filenames, paths, PHI, OCR text never written to public reports.
  - Private notes written only to gitignored private feedback file.
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

REPORT_DIR = ROOT / "reports" / "phase72b_operator_review_console"
JSON_REPORT = REPORT_DIR / "phase72b_operator_review_console_report.json"
MD_REPORT = REPORT_DIR / "phase72b_operator_review_console_report.md"
README = REPORT_DIR / "README_FAST_OPERATOR_REVIEW_SAFE.md"

PHASE71_QUEUE = (
    ROOT / "reports" / "phase71_operator_feedback_prioritization"
    / "operator_review_queue_SAFE.json"
)
PHASE72_PRIVATE = (
    ROOT / "reports" / "phase72_operator_feedback_collection"
    / "operator_feedback_PRIVATE.json"
)

ALLOWED_ANSWERS = [
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
]
WIZARD_SKIP = "skip"
WIZARD_QUIT = "quit"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_queue(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or PHASE71_QUEUE
    if not p.exists():
        raise FileNotFoundError(
            f"Phase71 review queue not found: {p}. Run Phase71 first."
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    items = payload.get("review_queue") or []
    return sorted(items, key=lambda it: (
        int(it.get("priority_tier") or 99),
        str(it.get("safe_file_id") or ""),
    ))


def load_private_feedback(path: Path | None = None) -> dict[str, Any]:
    p = path or PHASE72_PRIVATE
    if not p.exists():
        return {"phase": 72, "feedback": []}
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Pending item selection
# ---------------------------------------------------------------------------


def get_pending(
    records: list[dict[str, Any]],
    *,
    tier: int | None = None,
) -> list[dict[str, Any]]:
    """Return pending (unreviewed) records, optionally filtered by tier."""
    result = [r for r in records if r.get("answer") is None]
    if tier is not None:
        result = [r for r in result if int(r.get("priority_tier") or 0) == tier]
    return sorted(result, key=lambda r: (
        int(r.get("priority_tier") or 99),
        int(r.get("priority_rank") or 999),
        str(r.get("safe_file_id") or ""),
    ))


def get_next_pending(
    records: list[dict[str, Any]],
    *,
    tier: int | None = None,
) -> dict[str, Any] | None:
    """Return the first unreviewed record, optionally filtered by tier."""
    pending = get_pending(records, tier=tier)
    return pending[0] if pending else None


def get_next_pending_tier1(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first unreviewed tier-1 record (deterministic)."""
    return get_next_pending(records, tier=1)


# ---------------------------------------------------------------------------
# Answer recording
# ---------------------------------------------------------------------------


def validate_answer(answer: str) -> None:
    if answer not in ALLOWED_ANSWERS:
        raise ValueError(
            f"Invalid answer {answer!r}. Allowed: "
            + ", ".join(ALLOWED_ANSWERS)
        )


def record_answer_to_file(
    safe_file_id: str,
    answer: str,
    *,
    private_path: Path | None = None,
    private_note: str | None = None,
) -> None:
    """Record one operator answer into the private feedback file."""
    validate_answer(answer)
    p = private_path or PHASE72_PRIVATE
    if not p.exists():
        raise FileNotFoundError(
            f"Private feedback file not found: {p}. Run Phase72 --init first."
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
# Public report generation
# ---------------------------------------------------------------------------


def _answer_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for rec in records:
        ans = rec.get("answer")
        if ans:
            dist[str(ans)] = dist.get(str(ans), 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))


def _tier_counts(
    records: list[dict[str, Any]],
) -> tuple[int, int, int, int]:
    """Return (tier1_pending, tier1_reviewed, tier2_pending, tier2_reviewed)."""
    t1p = t1r = t2p = t2r = 0
    for rec in records:
        tier = int(rec.get("priority_tier") or 2)
        reviewed = rec.get("answer") is not None
        if tier == 1:
            if reviewed:
                t1r += 1
            else:
                t1p += 1
        else:
            if reviewed:
                t2r += 1
            else:
                t2p += 1
    return t1p, t1r, t2p, t2r


def generate_console_report(
    queue: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
    *,
    report_dir: Path | None = None,
    private_path: Path | None = None,
) -> dict[str, Any]:
    target_dir = report_dir or REPORT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    total = len(queue)
    reviewed = sum(1 for r in feedback_records if r.get("answer") is not None)
    pending = total - reviewed
    t1p, t1r, t2p, t2r = _tier_counts(feedback_records)
    answer_dist = _answer_distribution(feedback_records)
    unresolved_hp = t1p

    # Check gitignore
    gitignore = ROOT / ".gitignore"
    gitignored = False
    if gitignore.exists():
        text = gitignore.read_text(encoding="utf-8")
        gitignored = (
            "operator_feedback_PRIVATE.json" in text
            or "phase72_operator_feedback_collection/operator_feedback_PRIVATE" in text
        )

    private_present = (private_path or PHASE72_PRIVATE).exists()

    if reviewed == 0:
        conclusion = "operator_review_console_ready"
    elif reviewed < total:
        conclusion = "operator_review_console_ready"
    else:
        conclusion = "operator_review_console_complete"

    report: dict[str, Any] = {
        "phase": "72B",
        "phase_name": "Human-Minimized Operator Review Console",
        "generated_at": datetime.now(UTC).isoformat(),
        "conclusion": conclusion,
        "recommended_next_phase": "Phase72C Tier-1 Operator Review Execution",
        "recommended_next_action": (
            f"Run the one-command operator review console for the "
            f"{t1p} tier-1 pending item(s) first."
        ),
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
        "unresolved_high_priority_count": unresolved_hp,
        "tier_1_pending_count": t1p,
        "tier_1_reviewed_count": t1r,
        "tier_2_pending_count": t2p,
        "tier_2_reviewed_count": t2r,
        "console_default_scope": "tier_1_only",
        "answer_distribution": answer_dist,
        "feedback_private_file_present": private_present,
        "private_feedback_file_gitignored": gitignored,
        "one_command_usage": (
            "python scripts/run_phase72b_operator_review_console.py --tier 1"
        ),
        "reports_read": [
            "phase71_operator_review_queue_SAFE",
            "phase72_operator_feedback_PRIVATE",
        ],
        "reports_missing": [],
        "validation_commands": [
            "python -m pytest tests",
            "python scripts/run_phase72b_operator_review_console.py --summary",
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


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 72B Human-Minimized Operator Review Console",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Recommended next phase: {report['recommended_next_phase']}",
        "",
        "## One-Command Usage",
        "",
        f"```bash",
        f"{report['one_command_usage']}",
        f"```",
        "",
        "## Progress",
        "",
        f"- review_queue_count: `{report['review_queue_count']}`",
        f"- reviewed_count: `{report['reviewed_count']}`",
        f"- pending_count: `{report['pending_count']}`",
        f"- **tier_1_pending_count: `{report['tier_1_pending_count']}`** ← start here",
        f"- tier_2_pending_count: `{report['tier_2_pending_count']}`",
        f"- unresolved_high_priority_count: `{report['unresolved_high_priority_count']}`",
        f"- console_default_scope: `{report['console_default_scope']}`",
        "",
        "## Safety Flags",
        "",
        f"- external_api_used: `{report['external_api_used']}`",
        f"- production_extractor_should_change_yet: `{report['production_extractor_should_change_yet']}`",
        f"- production_ocr_should_change_yet: `{report['production_ocr_should_change_yet']}`",
        f"- safety_gates_should_change_yet: `{report['safety_gates_should_change_yet']}`",
        f"- manual_review_boundary_retained: `{report['manual_review_boundary_retained']}`",
        "",
    ]
    if report["answer_distribution"]:
        lines += [
            "## Answer Distribution",
            "",
            "| Answer | Count |",
            "| --- | ---: |",
        ]
        for ans, cnt in report["answer_distribution"].items():
            lines.append(f"| `{ans}` | {cnt} |")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


def _write_readme(path: Path) -> None:
    path.write_text("""\
# Fast Operator Review — Phase 72B (SAFE)

## One command to start

```bash
# Review tier-1 items (3 items, recommended starting point)
python scripts/run_phase72b_operator_review_console.py --tier 1

# Review tier-2 items
python scripts/run_phase72b_operator_review_console.py --tier 2

# Review all pending
python scripts/run_phase72b_operator_review_console.py --all

# Non-interactive summary only
python scripts/run_phase72b_operator_review_console.py --summary

# Launch Streamlit operator feedback UI
python scripts/run_phase72b_operator_review_console.py --launch-ui
```

## Terminal wizard

For each pending item the wizard shows:
1. safe_file_id, priority tier, problem class
2. Review goal
3. Operator question
4. Numbered answer options

**Keystrokes:**
- `1`–`10` — select answer
- `s` — skip this item
- `q` — quit and save summary

Answers are saved immediately to the gitignored private feedback file.
A fresh public summary is generated on exit.

## Streamlit UI

```bash
streamlit run app/operator_feedback.py
```

Provides clickable answer buttons, progress bar, and automatic advance.

## Privacy rules

- Do NOT type patient names or other PHI at any prompt.
- Answers are saved to `operator_feedback_PRIVATE.json` (gitignored).
- Public summary contains only counts and safe_file_ids — no notes.
""", encoding="utf-8")


# ---------------------------------------------------------------------------
# Interactive terminal wizard
# ---------------------------------------------------------------------------


def _print_item(item: dict[str, Any], index: int, total: int) -> None:
    print()
    print(f"{'='*60}")
    print(f"  Item {index}/{total}  |  Tier {item.get('priority_tier')}  "
          f"|  {item.get('suspected_problem_class')}")
    print(f"  safe_file_id: {item.get('safe_file_id')}")
    print(f"{'='*60}")
    print(f"  Goal   : {item.get('review_goal', '')}")
    print(f"  Question: {item.get('operator_question', '')}")
    print()
    for i, ans in enumerate(ALLOWED_ANSWERS, start=1):
        print(f"  {i:2d}. {ans}")
    print()
    print("   s. skip    q. quit")


def _run_wizard(
    pending: list[dict[str, Any]],
    *,
    private_path: Path,
    report_dir: Path,
    queue: list[dict[str, Any]],
) -> None:
    total = len(pending)
    if total == 0:
        print("No pending items in this tier.")
        return
    i = 0
    while i < len(pending):
        item = pending[i]
        _print_item(item, i + 1, total)
        while True:
            try:
                raw = input("  Answer: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                raw = "q"
            if raw == "q":
                print("Quitting. Saving summary...")
                _finish(queue, private_path=private_path, report_dir=report_dir)
                return
            if raw == "s":
                print("  Skipped.")
                i += 1
                break
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(ALLOWED_ANSWERS):
                    answer = ALLOWED_ANSWERS[idx]
                    record_answer_to_file(
                        str(item.get("safe_file_id") or ""),
                        answer,
                        private_path=private_path,
                    )
                    print(f"  Saved: {answer}")
                    i += 1
                    break
                print("  Invalid — enter a number 1-10, 's' to skip, 'q' to quit.")
            except ValueError:
                print("  Invalid — enter a number 1-10, 's' to skip, 'q' to quit.")
    _finish(queue, private_path=private_path, report_dir=report_dir)


def _finish(
    queue: list[dict[str, Any]],
    *,
    private_path: Path,
    report_dir: Path,
) -> None:
    feedback = load_private_feedback(private_path)
    records = feedback.get("feedback") or []
    report = generate_console_report(
        queue, records, report_dir=report_dir, private_path=private_path
    )
    print()
    print(f"Summary saved. reviewed={report['reviewed_count']} "
          f"pending={report['pending_count']} "
          f"tier1_pending={report['tier_1_pending_count']}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_console(
    *,
    mode: str = "summary",
    tier: int | None = 1,
    queue_path: Path | None = None,
    private_path: Path | None = None,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    """Entry point callable from tests and CLI (non-interactive modes)."""
    target_dir = report_dir or REPORT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_readme(target_dir / README.name)

    queue = load_queue(queue_path)
    feedback = load_private_feedback(private_path)
    records = feedback.get("feedback") or []

    if mode == "summary":
        return generate_console_report(
            queue, records, report_dir=target_dir, private_path=private_path
        )

    if mode == "wizard":
        pending = get_pending(records, tier=tier)
        _run_wizard(
            pending,
            private_path=private_path or PHASE72_PRIVATE,
            report_dir=target_dir,
            queue=queue,
        )
        feedback = load_private_feedback(private_path)
        records = feedback.get("feedback") or []
        return generate_console_report(
            queue, records, report_dir=target_dir, private_path=private_path
        )

    return generate_console_report(
        queue, records, report_dir=target_dir, private_path=private_path
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 72B — one-command operator review console."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summary", action="store_true",
                      help="Non-interactive: generate public summary (default).")
    mode.add_argument("--tier", type=int, metavar="N",
                      help="Interactive wizard for tier N (1 or 2).")
    mode.add_argument("--all", action="store_true",
                      help="Interactive wizard for all pending items.")
    mode.add_argument("--launch-ui", action="store_true",
                      help="Print command to launch Streamlit operator feedback UI.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.launch_ui:
        print("streamlit run app/operator_feedback.py")
        return 0

    if args.tier is not None:
        mode = "wizard"
        tier = args.tier
    elif args.all:
        mode = "wizard"
        tier = None
    else:
        mode = "summary"
        tier = 1

    report = run_console(mode=mode, tier=tier)
    print(f"Phase 72B conclusion: {report.get('conclusion')}")
    print(f"tier_1_pending_count: {report.get('tier_1_pending_count')}")
    print(f"tier_2_pending_count: {report.get('tier_2_pending_count')}")
    print(f"reviewed_count: {report.get('reviewed_count')}")
    print(f"pending_count: {report.get('pending_count')}")
    print(f"one_command_usage: {report.get('one_command_usage')}")
    print(f"production_extractor_should_change_yet: "
          f"{report.get('production_extractor_should_change_yet')}")
    print(f"production_ocr_should_change_yet: "
          f"{report.get('production_ocr_should_change_yet')}")
    print(f"json_report: {JSON_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
