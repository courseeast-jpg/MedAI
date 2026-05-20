# MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01

## Why This Exists
This block wires safe aggregate terminology match metadata into Advanced technical details only.

## Wiring Summary
- Helper env var: `MEDAI_TERMINOLOGY_LOOKUP_ENABLED`.
- UI env var: `MEDAI_TERMINOLOGY_LOOKUP_UI_ENABLED`.
- Both env vars must be truthy before anything renders.
- The UI renders only already-emitted aggregate helper metadata.
- No adapter is created by the UI and no terminology rows are read.

## Default-Off Proof
- Neither env truthy renders: `False`.
- Helper-only env truthy renders: `False`.
- UI-only env truthy renders: `False`.
- Both env truthy renders: `True`.

## Read-Only Render Surface
- Render location: Advanced technical details only.
- Allowed Streamlit calls: `['st.markdown', 'st.caption']`.
- Forbidden Streamlit call count: `0`.
- Rendered fields: match family, terminology system family, matches count, review-required / auto-accept summary, and disclaimer.

## License And Privacy Proof
- Licensed terminology rows read/rendered/publicly reported: false.
- Row code, display text, synonyms, definitions, raw text, OCR text, document text, filenames, private paths, PHI, and secrets are not rendered.
- Public report privacy clean: `True`.

## What Was Not Changed
- Extraction, OCR, classifier, thresholds, scoring, cue packs, DDI, clinical inference, auto-accept, and external API behavior were not changed.
- Cue expansion remains NOT recommended.

## Validation Evidence
- Focused CKA-TERM-INTEGRATION-WIRING-NEXT-01 tests: passed, 13/13.
- Direct report script: passed.
- Public report privacy checks: passed, 3/3 wiring reports privacy-clean.
- Final CKA MVP validation: passed, 12/12 validation cases and 693 total tests passed; external API used: false.
- B07 term01 validation: passed, 6/6 cases; external API used: false.
- ROUTE-FIX validation: passed, `medai_route_fix01_ready`; external API used: false.
- UI ops validation: passed, `medai_ui_ops_panel_ready`.
- UI boot validation: passed, `medai_ui_boot_fix_startup_resilience_ready`.
- Staged safety check: passed; only scoped terminology wiring files were staged.

## Recommended Next Step
`CKA-TERM-INTEGRATION-WIRING-UAT-01 or ROADMAP-05`.
