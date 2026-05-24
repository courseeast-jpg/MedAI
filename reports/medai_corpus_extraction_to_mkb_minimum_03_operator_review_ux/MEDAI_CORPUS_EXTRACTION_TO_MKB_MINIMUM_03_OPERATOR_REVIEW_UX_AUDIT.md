# MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-03 — Milestone A Baseline Audit

Audit-only inspection. No runtime code changed in this milestone.

## Branch / Head

- Branch: `clinical-knowledge-architecture`
- HEAD short before audit: `c7c5f13`

## Files inspected (read-only)

- `mkb/sqlite_store.py`
- `execution/pipeline.py`
- `execution/mkb_writer.py`
- `app/main.py`
- `app/extracted_information_preview.py`
- `app/test_launcher.py`
- `app/schemas.py`
- `app/operator_feedback.py`, `app/operator_control_panel.py`, `app/operator_safety.py`

## Current review-bound record shape

| Field | Value when review-bound |
| --- | --- |
| `fact_type` | `test_result` |
| `tier` | `quarantined` |
| `status` | `active` (default) |
| `requires_review` | **true** |
| structured carries | `test_name`, `value`, `unit`, `reference_range`, `flag`, `parser_name`, `language_hint`, `source_line_hash`, `requires_human_review`, `auto_accept_allowed`, `provenance` |

## Update paths already available

| Capability | API |
| --- | --- |
| Insert/replace record by id | `SQLiteStore.write_record` |
| Update status (+ optional tier) | `SQLiteStore.update_status` |
| Append ledger audit event | `SQLiteStore.write_ledger(LedgerEvent)` |
| Fetch single record | `SQLiteStore.get_record` |
| Fetch review-bound records | `SQLiteStore.get_records_requiring_review` |

The existing `LedgerEvent` (`app/schemas.py`) already carries
`event_type`, `record_id`, `source_type`, `previous_value`, `details`,
`timestamp`, `session_id`. **Reusable** as the operator-action log. No
new private JSONL log is required.

## Planned operator-action target table

| Action | new `tier` | new `status` | new `requires_review` |
| --- | --- | --- | :-: |
| `accept_after_source_comparison` | `active` | `active` | **false** |
| `reject_extracted_fact` | `superseded` | `rejected_after_operator_review` | **false** |
| `defer_extracted_fact` | `quarantined` | `deferred_by_operator` | **true** |

## Action guards already implied by schema

- **Missing record** — `get_record` returns `None` → action fails safely.
- **Already-active record** — `tier == active` AND
  `requires_review == False` → accept path declines (already accepted).
- **Medication fact** — `fact_type == "medication"` → accept declines
  via `medication_safety_workflow_required` (the operator-driven lab
  review path **must not** override the medication safety gate).
- **DDI-blocked record** — `ddi_status` in
  `{"high_blocked", "pending_ddi_check", "pending_medication_review", "pending_ddi"}`
  → accept declines.

## UI entry point

`app/main.py:render_run_result_card` already invokes
`app/extracted_information_preview.build_extracted_information_preview_plan`
to draw the preview table. The new action workflow will:

1. extend the Streamlit-free render plan with per-row action affordances
   (action keys, disclaimer, button labels) — no Streamlit import added
   to the helper;
2. attach `st.button` calls in `app/main.py` after the existing table,
   wrapped in a `try/except` block so any Streamlit-side failure cannot
   block the rest of the card.

## Privacy observations

- The audit inspected zero real documents, zero raw OCR text, zero raw
  filenames, zero private paths.
- The audit emits counts and safe IDs only.
- No external API was used.

## Conclusion

Ready to proceed to Milestone B (operator action model). Existing
SQLite ledger + update paths are reused; no fallback private JSONL log
is needed. The operator-action map preserves the review-bound default
and never auto-accepts.
