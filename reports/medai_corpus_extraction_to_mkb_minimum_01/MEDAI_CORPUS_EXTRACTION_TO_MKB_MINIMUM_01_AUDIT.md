# MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-01 — Milestone A Baseline Audit

Audit-only inspection. No runtime code changed in this milestone.

## Branch / Head

- Branch: `clinical-knowledge-architecture`
- HEAD short before audit: `ef42cb8`

## Files inspected (read-only)

- `execution/pipeline.py`
- `execution/jobs.py`
- `execution/mkb_writer.py`
- `execution/validation.py`
- `extractors/spacy_extractor.py`
- `mkb/sqlite_store.py`
- `app/main.py`
- `app/test_launcher.py`
- `app/schemas.py`

## Required questions

| Question | Answer |
| --- | --- |
| Are entities extracted today? | yes, when text matches existing spaCy regexes or structured lab patterns |
| Are records created today? | yes, via `ExecutionPipeline._entities_to_records` |
| Are records written to SQLite today? | yes, via `MKBWriter.write` calling `SQLiteStore.write_record` |
| Are review-bound records persisted today? | yes, via `MKBWriter.write` when `requires_review` is true or `tier='quarantined'` |
| Are records visible in MKB Explorer today? | yes, via `SQLiteStore.get_by_specialty` and raw SQL paging |
| Are records returned by MKB retrieval today? | yes, via `get_by_specialty`, `get_active_medications`, `get_active_diagnoses`, `get_records_requiring_review` |
| Are structured facts visible in the Run & Review card today? | **no** |

## Root cause: where extracted information is dropped or hidden

The data is in memory and in SQLite. The hand-off layer to the Run &
Review card has no field for it.

Specific gaps:

1. `TestFileResult` dataclass at `app/test_launcher.py:42-71` has no
   `extracted_medical_facts_preview_safe` field, no
   `extracted_medical_fact_count`, no `extraction_to_mkb_candidate_count`,
   no `extraction_to_mkb_written_count`, no
   `extraction_to_mkb_review_count`.
2. `_process_one_file` at `app/test_launcher.py:344-423` reads
   `extractor_result` and `audit` but does not copy entities,
   structured facts, or written-record summaries into TestFileResult.
3. `render_run_result_card` at `app/main.py:1133+` has no "Extracted
   information preview" section. The current prose
   "The lab values have not been checked or accepted" is hard-coded in
   `operator_result_explanation` at `app/main.py:869+` regardless of
   whether facts were actually extracted.
4. `extractor_result` lacks an explicit
   `extracted_medical_facts_preview_safe` field that the UI helper
   layer can consume safely.
5. No UI render-plan helper module exists that builds a public-safe
   extracted-fact table from a TestFileResult-shaped dict without
   importing Streamlit.

## Downstream persistence already works

| Aspect | Today |
| --- | :-: |
| `MKBWriter` persists accepted records | yes |
| `MKBWriter` persists review-bound records to SQLite | yes |
| `MKBWriter` persists quarantined records to SQLite | yes |
| `SQLiteStore.get_records_requiring_review` returns them | yes |
| MKB Explorer can filter by specialty and tier | yes |
| MKB Explorer can filter by `fact_type` | no |

## Minimum safe path proposed (Milestones B..G)

- Add `execution/extracted_medical_facts.py` deterministic local-only
  lab adapter.
- Merge adapter output into `extracted["entities"]` with clear
  provenance, only when document family is lab-style and existing
  entities are empty or sparse.
- Expose preview + counts on `extractor_result`.
- Carry preview through `TestFileResult`.
- Add `app/extracted_information_preview.py` Streamlit-free render-plan
  helper.
- Wire the render plan into `render_run_result_card` before the
  "What MedAI did not do" section.
- Adjust the absolute prose
  "The lab values have not been checked or accepted" to a precise
  statement that reflects the new preview section.
- Augment MKB Explorer with a `fact_type` filter only if strictly
  needed for visibility of `test_result` records.
- Preserve every safety gate, threshold, OCR routing rule, DDI
  behavior, privacy gate, and medical decision rule. The adapter is
  conservative (fewer correct facts beats unsafe facts).
- Preserve review-bound default and the
  `auto_accept_allowed=false` default for every new fact.

## Privacy observations

- The current Run & Review card does not print raw text, raw OCR text,
  or raw filenames in its public path.
- The current `TestFileResult` carries `file_name`, but the card
  already treats it with care.
- Public reports continue to pass
  `clinical_knowledge.privacy.check_public_report_payload`.
- This audit inspected no real document, no raw OCR text, no private
  path. No external API was used.

## Conclusion

Ready to proceed to Milestone B (safe structured fact model / adapter).
No code has changed in Milestone A. No external API used. No private
data accessed.
