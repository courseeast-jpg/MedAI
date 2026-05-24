# MEDAI-UI-STARTUP-RESILIENCE-08 — Milestone A Baseline Audit

Audit-only inspection. No runtime code changed in this milestone.

## Branch / Head

- Branch: `clinical-knowledge-architecture`
- HEAD short before audit: `bcf12a4`

## Files inspected (read-only)

- `app/main.py`
- `app/startup_preflight.py`
- `mkb/sqlite_store.py`
- `mkb/vector_store.py`
- `app/config.py`

## Question 1 — Which components does `load_system()` initialize?

The factory at `app/main.py:237-265` instantiates ten components in
sequence, all in a single try-protected scope from
`app/startup_preflight.initialize_startup_state`:

1. `SQLiteStore`
2. `VectorStore` (Chroma)
3. `QualityGate` (requires SQLite + Vector)
4. connector registry
5. `MedicationSafetyGate`
6. `ResponseScorer`
7. `ClaudeSynthesizer`
8. `DecisionEngine`
9. `ExecutionPipeline`
10. `EnrichmentEngine`

If any of those raises, the wrapper at
`app/startup_preflight.initialize_startup_state` catches **any**
`Exception` and the UI renders the diagnostics panel with the message
"MKB initialization failed. MedAI started in diagnostics-only mode."

## Question 2 — Which exceptions can be misreported as DB init failure?

- `pydantic.ConfigValidationError` raised from `chromadb`'s
  `VectorStore` constructor.
- `chromadb` dependency / model incompatibility.
- Any `OSError` raised when `CHROMA_PATH` cannot be created.
- Any optional-dependency failure that surfaces through
  `connectors_registry`, the `PIIStripper`, or the `DecisionEngine`
  construction path.

All of these are surfaced today as **MKB initialization failed**, even
though SQLite may have initialized cleanly seconds earlier.

## Question 3 — Is SQLCipher's encrypted-DB diagnostic probe expected to fail standard sqlite3?

Yes. `mkb/sqlite_store.py` uses `sqlcipher3` when available, so the
actual store can read an encrypted DB. The diagnostics probe in
`startup_preflight.sqlite_metadata_probe` opens the DB through the
standard `sqlite3` module without the key, so it cannot read the
encrypted bytes. The header probe returns `encrypted_or_unknown` and
the metadata probe returns `connect_failed_database_error`. The
combined category `encrypted_or_wrong_key_or_unreadable` is **correct
as a metadata observation** but misleading as a startup verdict — the
actual `SQLiteStore` may already have initialized successfully.

## Question 4 — Can the UI safely start without VectorStore / Chroma?

Yes. The existing 02/03/04 chain proves that:

- `ExecutionPipeline` accepts `vector_store=None` and
  `quality_gate=None`.
- `MKBWriter` with `quality_gate=None` treats records as approved and
  preserves `requires_review` / `tier` semantics, so review-bound
  default holds.
- Local-only and external-API-blocked defaults are unaffected.
- Semantic / vector search becomes unavailable; the UI must signal this
  rather than silently break or pretend the DB failed.

## Componentization plan

Each step in `load_system()` becomes its own try/except with a clear
component status. The wrapper returns a `StartupState` with:

| Step | Component | Failure mode |
| -: | --- | --- |
| 1 | `SQLiteStore` | **critical** — diagnostics-only mode |
| 2 | `VectorStore` | optional — degraded_vector_mode |
| 3 | `QualityGate` | depends on SQLite + Vector; degraded → `None` |
| 4 | connector registry | optional; degraded → empty |
| 5 | `MedicationSafetyGate` | depends on SQLite; degraded when SQLite missing |
| 6-8 | scorer / synthesizer / engine | optional |
| 9 | `ExecutionPipeline` | built whenever SQLite is OK |
| 10 | `EnrichmentEngine` | degraded when SQLite or Vector missing |

## New startup categories

- `sqlite_store_init_failed`
- `sqlcipher_encrypted_metadata_probe_unreadable_but_store_ok`
- `vector_store_init_failed`
- `quality_gate_degraded`
- `app_startup_ok_with_degraded_vector`
- `app_startup_ok`
- `app_startup_failed`

## New UI banners

| State | Banner |
| --- | --- |
| SQLite OK + Vector OK | (no banner — normal) |
| SQLite OK + Vector failed | "Vector/semantic index unavailable. SQLite MKB and local review workflow remain available." |
| SQLite failed | "Diagnostics-only mode. No clinical processing started." |

## Privacy observations

- The audit inspected zero real documents, zero raw OCR text, zero raw
  filenames, zero private paths.
- All planned categories carry safe public-summary fields only.
- No `DB_ENCRYPTION_KEY` value, env var, or full path appears in any
  proposed report payload.

## Conclusion

Ready to proceed to Milestone B (componentized startup). Existing
review-bound default, medication / DDI gates, OCR routing, classifier
rules, and confidence thresholds remain untouched.
