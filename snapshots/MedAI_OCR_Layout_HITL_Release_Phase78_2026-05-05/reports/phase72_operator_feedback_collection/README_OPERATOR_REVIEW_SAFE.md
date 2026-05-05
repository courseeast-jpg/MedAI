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
python scripts/run_phase72_operator_feedback_collection.py --record \
    --safe-file-id file_001 --answer correct_review

# Optionally add a private note (never written to public reports)
python scripts/run_phase72_operator_feedback_collection.py --record \
    --safe-file-id file_001 --answer correct_review \
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
