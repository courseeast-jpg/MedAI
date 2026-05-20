# MEDAI-CKA-TERM-INTEGRATION-NEXT-01 — Default-Off Terminology Match Hypothesis Helper

First concrete implementation block of the terminology / coding
integration track after the v1 freeze. Adds a single default-off,
fail-closed, aggregate-only, review-bound terminology match hypothesis
helper plus a focused 26-test module. No runtime / Streamlit / launcher
/ preflight / config change. No tags created. FREEZE pair and
PARK-20..23 tag pairs untouched. Cue expansion remains explicitly
**NOT** recommended.

## 1. Why this block exists

`CKA-TERM-INTEGRATION-PLAN-01` (commit `30502bd`) approved opening the
terminology track with a tightly-scoped first implementation. This
block delivers that helper under the SPEC's invariants:

- `default_off`: true
- `local_only`: true
- `no_licensed_rows_committed`: true
- `no_clinical_auto_accept`: true
- `no_diagnosis_or_treatment_inference`: true
- `no_ddi_behavior_change_unless_explicitly_scoped`: true
- `aggregate_only_public_reports`: true
- `review_bound_outputs`: true
- `tests_before_integration`: true (26/26 focused tests pass)

## 2. What helper was added

**Module:** `clinical_knowledge/terminology/term_match_hypothesis.py`

**Public function:**

```text
derive_terminology_match_hypothesis(
    record,
    *,
    enabled=None,
    env=None,
    lookup_adapter=None,
) -> dict | None
```

**Env var:** `MEDAI_TERMINOLOGY_LOOKUP_ENABLED`

**Adapter protocol:** `TerminologyLookupAdapter` (duck-typed; the
existing `SyntheticReadOnlyTerminologyAdapter` from
`clinical_knowledge.terminology.term05_read_only_adapter` satisfies
it).

**Positive signature:** record must carry a non-empty, operator-curated
candidate-text field from a fixed allowlist
(`terminology_candidate_text`, `candidate_text`, `candidate_query`).
The helper **explicitly does not** accept raw document fields
(`ocr_text`, `raw_text`, `native_text`, `filename`, etc.), so it
cannot be misused as a free-text terminology lookup over private
content.

**Output:** a fresh dict with controlled-vocabulary tokens, explicit
invariant flags, and aggregate counts only. **No row content** from
the underlying terminology source ever escapes the helper.

| Output field | Value / meaning |
| --- | --- |
| `terminology_match_hypothesis` | True |
| `source_phase` | `MEDAI-CKA-TERM-INTEGRATION-NEXT-01` |
| `enabled` | True |
| `env_var` | `MEDAI_TERMINOLOGY_LOOKUP_ENABLED` |
| `match_family` | `exact_terminology_match` / `ambiguous_terminology_match` / `unmapped_terminology_candidate` |
| `terminology_system_family` | `rxnorm` / `loinc` / `internal_public_reference` |
| `matches_count` | aggregate count (no row content) |
| `license_class` | `aggregate_public_report_only` |
| `review_required` | True |
| `auto_accept_allowed` | False |
| `clinical_interpretation_performed` / `diagnosis_inference_performed` / `treatment_inference_performed` / `medication_inference_performed` / `ddi_behavior_changed` / `abbreviation_expanded` / `lab_value_parsed` | False |
| `licensed_row_content_included` / `raw_text_emitted` / `raw_ocr_text_emitted` / `raw_document_text_emitted` / `raw_filename_emitted` / `private_path_emitted` / `phi_emitted` / `secret_emitted` | False |
| `public_report_safe` | True |
| `ocr_routing_changed` / `ocr_engine_behavior_changed` / `pdf_text_extraction_behavior_changed` / `layout_extraction_behavior_changed` / `table_extraction_behavior_changed` / `classifier_behavior_changed` / `thresholds_or_scoring_changed` / `cue_packs_added` / `external_api_used` | False |
| `frozen_operator_release_preserved` | True |
| `freeze_tags_touched` / `park_20_tags_touched` / `park_21_tags_touched` / `park_22_tags_touched` / `park_23_tags_touched` | False |
| `disclaimer` | controlled review disclaimer |

## 3. Default-off proof

Five independent default-off paths, any one of which suffices:

1. `enabled=False` always returns `None`, regardless of env.
2. `enabled=None` AND env var unset or falsy returns `None` (verified
   against `_TRUTHY = {"1", "true", "yes", "on", "enabled"}` and every
   member of `{"0", "false", "no", "off", "disabled", ""}` in the
   focused tests).
3. `lookup_adapter=None` **fails closed** — returns `None` even when
   the env var is truthy. Production callers must explicitly inject an
   audited adapter.
4. The record fails the positive signature (missing /
   whitespace-only / wrong field) returns `None`.
5. The adapter returns no candidate text — the helper still emits
   aggregate counts of zero but never row content.

`is_terminology_lookup_default_disabled(env={})` returns `True`;
`is_terminology_lookup_enabled(env={})` returns `False`.

## 4. License / privacy proof

- The helper does not open any file in `terminology_data/`,
  `data/terminology/`, `config/terminology_sources.local.json`, or
  `LICENSE_ACK_PRIVATE.json`. It does not import any module that does
  so by default.
- The adapter protocol is duck-typed and exposes only `.lookup()`.
- The helper extracts only an aggregate count plus controlled-
  vocabulary family labels from the adapter's result; it never embeds
  match codes, display strings, system identifiers beyond a
  controlled-vocab family tag, definitions, RxCUIs, LOINC numbers, or
  synonyms.
- A focused test (`test_no_licensed_row_content_in_output`)
  enumerates every output key and confirms none of them contains a
  row-content substring (`code`, `display`, `rxcui`, `loinc_num`,
  `synonym`, `concept`, `definition`).
- A focused test (`test_no_raw_text_filenames_paths_phi_secrets_in_output`)
  confirms all 7 raw-content / privacy-leak flags are False.
- The output `license_class` is set to `aggregate_public_report_only`;
  the underlying source license class is tracked separately by the
  `PLAN-01` license-class table.
- The helper's source file contains no Streamlit symbol
  (`streamlit`, `import streamlit`, `st.`, `st_button`, `st_form`).

## 5. Synthetic / mocked terminology testing method

All 26 focused tests use the existing
`SyntheticReadOnlyTerminologyAdapter` from `CKA-TERM-05`. This adapter:

- Uses **only synthetic in-memory fixtures** (
  `build_synthetic_qa_store()` from `cka_term_qa_golden`).
- Does **not** access the private TERM-02 store.
- Does **not** access `terminology_data/` or `data/terminology/`.
- Advertises `read_only=True` and `external_api_used=False`.
- Marks every match as `synthetic=True`.

No licensed terminology rows are required by, read by, or produced by
the tests. The synthetic adapter is the canonical test fixture for any
subsequent CKA-TERM-* block.

## 6. What was NOT changed

- `app/main.py` — unchanged. No Streamlit wiring. No import of the
  new helper. A focused test asserts the marker
  `MEDAI-CKA-TERM-INTEGRATION-NEXT-01`, the function name
  `derive_terminology_match_hypothesis`, and the module name
  `term_match_hypothesis` are absent from `app/main.py`.
- Launchers (`Start_MedAI_UI.bat`, `Start_MedAI_UI_Silent.vbs`,
  `Start_MedAI_Test_UI.bat`, `Start_MedAI_UI_Encrypted.bat`) —
  unchanged.
- `app/startup_preflight.py` — unchanged.
- `app/config.py` — unchanged.
- OCR routing, extraction (PDF text / layout / table), classifier,
  thresholds, scoring, cue packs — unchanged.
- DDI, diagnosis, medication, treatment, abbreviation logic —
  unchanged.
- External APIs — disabled by default; the helper never calls one.
- `terminology_data/`, `data/terminology/`,
  `LICENSE_ACK_PRIVATE.json` — never read or staged.
- Tags — none created, moved, or deleted. PARK-20..23 pairs still
  resolve to their original commits. FREEZE pair still resolves to
  `7ef8ffd`.

## 7. Public-report no-row-output rule

The helper's output is itself the canonical "aggregate-only" record:

- `license_class = aggregate_public_report_only`
- `licensed_row_content_included = False`
- `public_report_safe = True`

Any consuming public report (this one, future wiring reports) must
follow the same rule: no row content; only family labels, counts, and
invariant flags. The privacy scanner
(`clinical_knowledge.privacy.check_public_report_payload`) gates
commits.

## 8. Validation evidence

| Validation | Result |
| --- | --- |
| Focused CKA-TERM-INTEGRATION-NEXT-01 tests | **26/26 PASS** |
| Public-report privacy checks (3 PACKAGING reports) | PASS |
| Final CKA MVP validation | PASS (`cka_mvp_release_package_ready`, 693 tests, `external_api_used: false`) |
| B07 term01 opt-in integration | PASS (`cases_failed: 0`, `external_api_used: false`) |
| ROUTE-FIX 01 | PASS (`medai_route_fix01_ready`, `passed: true`) |
| UI ops panel | PASS (`medai_ui_ops_panel_ready`) |
| UI boot fix | PASS (`medai_ui_boot_fix_startup_resilience_ready`) |
| Staged safety check | PASS — only CKA-TERM-INTEGRATION-NEXT-01 scoped files (1 helper + 1 test + 3 reports) staged into the implementation commit; validation receipt churn isolated to a separate receipt-refresh commit. |
| Full pytest | Not run. Reports/implementation block; the focused test module passes 26/26. Known Streamlit-import sandbox limitation persists but is unrelated to this block. |

## 9. Recommended next step

`MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01` — only after operator
review of this helper. The wiring block must:

- Remain default-off; check `MEDAI_TERMINOLOGY_LOOKUP_ENABLED` (plus a
  second UI env var if desired, mirroring DIAG-17/18).
- Be read-only (no buttons, forms, callbacks, actions, state
  mutations, data-layer writes, document_type mutations).
- Live inside the Advanced technical details expander only.
- Emit only safe `st.markdown` / `st.caption` calls.
- Embed **no row content** from the underlying terminology source.
- Be guarded by `try/except` so non-Streamlit environments stay
  importable.
- Not touch PARK-20..23 tag pairs or the FREEZE tag pair.

If wiring is not desired now, proceed to `MEDAI-ROADMAP-05` to
re-evaluate the next strategic phase after this first terminology
implementation lands.

## 10. Cue expansion remains NOT recommended

Reaffirmed. The helper's output explicitly sets `cue_packs_added=False`
and the focused test
`test_cue_expansion_remains_not_recommended` enforces it. Cue
expansion would re-open the classifier surface, expand the regression
surface, create new licensing / privacy responsibilities, and resolve
no currently failing operator outcome.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Whole MedAI project | ~94.1% done / ~5.9% remaining | ~94.3% done / ~5.7% remaining |
