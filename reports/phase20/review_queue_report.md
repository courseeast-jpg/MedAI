# Phase 20 Review Queue + Operator Audit Loop

## Goal

Phase 20 adds a deterministic operator-review layer for documents and records that should remain observable but should not be auto-finalized.

## Queue Entry Conditions

- Extraction validation returns `needs_review`.
- Extraction validation returns `rejected`.
- Truth resolution quarantines a record for operator inspection.
- Medication safety requires review or blocks a medication record.
- Phase 12 quota-safe execution classifies an external API issue as `external_quota_blocked`.

## Captured Fields

Each review queue entry records:

- `run_id`
- `document_id`
- `source_filename`
- `reason`
- `confidence`
- `extractor_route`
- `extractor_actual`
- `timestamp`
- `recommended_action`
- `raw_evidence_path` when available

Additional context remains attached when relevant, including validation errors, retry visibility, DDI findings, and truth-resolution metadata.

## Auditability

- Operator follow-up is now durable in a single JSONL queue.
- External quota blocks remain visible without being promoted to hard failures.
- Phase 12 summaries expose the review queue path and queue size.
- Phase 18 full-cycle reports surface the same review queue observability in the top-level run summary.

## `queued_for_review` vs `review_queue_items`

- `queued_for_review` is a document-level Phase 12 outcome counter. It increments only when a processed document ends the run as `queued_for_review`.
- `review_queue_items` is the count of operator-audit rows written to the review queue JSONL for the run.

These values measure different things, so they do not need to match.

In the current baseline-compatible Phase 20 run:

- `queued_for_review = 0` because no document finished with the terminal outcome `queued_for_review`.
- `review_queue_items = 30` because the operator queue still captured reviewable events that did not change the document outcome to `queued_for_review`.

Most of those 30 queue rows come from two sources:

- truth-resolution quarantine events for medication dose conflicts inside documents that still completed as `written_with_review`
- external quota block events recorded for the 4 documents classified as `external_quota_blocked`

This distinction is intentional. `queued_for_review` preserves the Phase 19 document-outcome baseline, while `review_queue_items` adds Phase 20 audit visibility over all operator-touch review events.

## Baseline Protection

- Phase 19 quota-safe behavior is preserved: external quota blocks do not count as hard failures.
- The review queue adds observability only; it does not weaken deterministic routing, validation thresholds, or existing pass/fail semantics.
