# MEDAI-CORPUS2-P2-PRIVACY-UNBLOCK-AND-LIVE-EXTRACTION-01 — implementation report

## A1 over-tokenization unblock
- supplemental private values added: `3` by class `{'DATE': 2, 'PROVIDER': 2}`.
- vault values before/after: `8` / `11`.
- bounded rules: provider/person names near provider/signature/referring/ordering cues; facility names near facility types/labels; all spelled-out dates. Clinical terms, lab names, and allowlist phrases are excluded.

## A2 gate
- high/provider-facility/spelled-date raw leaks after re-tokenization: `0` / `0` / `0`.
- live_entry_gate_passed: `True`; run_result: `LIVE_FAIL`.

## Boundaries
No MKB, no auto-accept, no medical decision, shared $50 cap, no Corpus 1 access.
