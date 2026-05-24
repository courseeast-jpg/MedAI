# MEDAI-UI-STARTUP-RESILIENCE-09 — Milestone A Baseline Audit

Audit-only. No runtime code changed in this milestone.

## Branch / Head

- Branch: `clinical-knowledge-architecture`
- HEAD short before audit: `8f49ab6`

## Files inspected

- `app/startup_preflight.py`
- `app/main.py`
- `execution/pipeline.py`
- `reports/medai_ui_startup_resilience_08/`

## Operator-observed state after pulling 08

| Field | Value |
| --- | --- |
| `sqlite_store_initialized` | **true** |
| `vector_store_initialized` | **true** |
| `db_startup_category` | `sqlcipher_encrypted_metadata_probe_unreadable_but_store_ok` |
| `exception_class` | `null` |
| UI banner seen by operator | "MKB initialization failed. MedAI started in diagnostics-only mode." |

## Question 1 — Why does UI still block when SQLite is OK?

`app/startup_preflight.initialize_startup_state` still computes:

```python
overall_ok = sqlite_ok and pipeline_ok
```

When `ExecutionPipeline` construction raises locally (likely from an
optional dependency reached during PIIStripper / OCR-gate / router
build), `pipeline_ok` is `False`, so `overall_ok` is `False`. The
wrapper drops the components dict and `main()` falls through to
`render_degraded_startup_panel` which prints the legacy
"MKB initialization failed" message — **even though the SQLiteStore
is healthy**.

## Question 2 — Which local component is failing?

The audit cannot inspect the operator's runtime state, but the
component class buckets the new startup model will surface include:

- PIIStripper / Presidio / spaCy / model dependency raised at
  ExecutionPipeline construction;
- `extraction.pii_stripper` regex fallback raised during config;
- `ingestion.cyrillic_ocr_gate` or extractors path raised on a
  Windows-specific edge;
- any optional dependency exception swallowed by
  `ExecutionPipeline.__init__` chain via `_build_router` or PII
  stripper construction.

Diagnostic strategy: per-component `(name, exception_class)` is already
captured in `component_status.component_errors` by the 08 refactor.
The new banner will surface that list to the operator (class names only
— never raw exception messages).

## Question 3 — Which tabs can safely remain available with SQLite-only startup?

| Tab | Requires | Available when SQLite-only |
| --- | --- | :-: |
| MKB Explorer | `components["sql"]` | yes |
| Conflict Review | `components["sql"]` | yes |
| Operator Control Panel | none (subprocess only) | yes |
| Startup Diagnostics | `StartupDiagnostics` | yes |
| Run & Review | `components["execution"]` | **degraded** (banner + safe message) |
| Upload | `components["execution"]` | **degraded** (banner + safe message) |
| Decision Engine / Query | `components["engine"]` | unavailable |

## Planned status categories

- `app_startup_ok`
- `app_startup_ok_with_degraded_vector`
- `app_startup_ok_with_degraded_pipeline` (new)
- `app_startup_ok_sqlite_only` (new)
- `app_startup_failed`

## Planned ok rule

`StartupState.ok = sqlite_store_initialized`. ExecutionPipeline failure
no longer blocks UI startup; only the tabs that require it are
degraded.

## Planned repair command surfaced in the banner

```
python scripts/run_medai_local_self_healing_validation_07.py
```

## Privacy observations

- Audit inspected zero real documents, zero raw OCR text, zero raw
  filenames, zero private paths.
- Component error reporting will continue to emit `(name, class_name)`
  tuples only — never exception message strings, never path data.
- No `DB_ENCRYPTION_KEY` value, env var, or full path will appear in
  any new report payload.

## Conclusion

Ready to proceed to Milestone B (startup status model + ok rule fix).
Existing review-bound default, medication / DDI gates, OCR routing,
classifier rules, and confidence thresholds remain untouched.
