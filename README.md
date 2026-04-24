# MedAI Platform v1.1

Personal multi-specialty medical AI - local-first, self-enriching, privacy-preserving.

> Decision support only. Not a medical device. All outputs require clinical verification.

---

## Prerequisites (one-time system installs)

```bash
# Ubuntu / Debian
sudo apt-get install -y tesseract-ocr sqlcipher python3.11

# macOS
brew install tesseract sqlcipher python@3.11
```

---

## Setup (one command)

```bash
# 1. Clone
git clone https://github.com/[user]/medai && cd medai

# 2. Add API key
cp .env.example .env
# Edit .env - set ANTHROPIC_API_KEY, GEMINI_API_KEY, and DB_ENCRYPTION_KEY

# 3. Bootstrap
make install
```

`make install` handles: pip deps, spaCy model, SQLite init, ChromaDB init, embedding model download.

`.env` is local-only and must not be committed.

### Phase 1.1 Gemini Activation

- `GEMINI_API_KEY` must be stored in the local repo `.env`.
- `.env` must not be committed.
- Real Gemini validation succeeds only when:
  - extractor used = `gemini`
  - no `gemini_route_legacy_fallback=rules_based` note is present

Validated Phase 1.1 Gemini activation:

- extractor: `gemini`
- entity count: `9`
- confidence: `0.8`
- outcome: `written`
- notes: `pii_method=presidio`

### Phase 2 Extraction Validation

Every execution job now performs deterministic extraction validation after routing and before MKB write:

- Required entity fields are checked (`type`, `text`)
- Confidence thresholds are enforced
- Entity schema shape is validated
- `extractor_route` and `extractor_actual` must match
- Each extraction is classified as `accepted`, `needs_review`, or `rejected`

Phase 2 thresholds:

- `accepted`: confidence `>= 0.70` and no validation errors
- `needs_review`: confidence `>= 0.50` and `< 0.70` with no fatal schema errors
- `rejected`: confidence `< 0.50` or any fatal validation error

Phase 2 outputs:

- `ExecutionResult.validation_status`
- `ExecutionResult.validation_errors` as machine-readable error objects
- Durable review artifact at `data/review/review_queue.jsonl`
- Extended metrics snapshot with validation counters and `avg_confidence_by_status`

`review_queue.jsonl` contains only `needs_review` and `rejected` extraction items, each with structured reasons.

---

## Run

```bash
make run
# Opens http://localhost:8501
```

---

## Test

```bash
make test           # Full test suite
make test-golden    # Golden test set only (8 critical cases)
```

---

## Architecture

```
Ingestion (docling PDF + OCR validation + Presidio PII strip + Claude extraction)
    ->
Deduplication Engine (exact / semantic / time-series / conflict / implausibility)
    ->
Conflict Resolver (quarantines irreconcilable facts for user review)
    ->
Truth Resolution Engine (7 priority rules)
    ->
Medication Safety Gate Layer 2 (DDI hard block)
    ->
MKB Write (SQLite + ChromaDB, encrypted, local only)
    ->
Decision Engine (classify -> retrieve -> execute -> score -> consensus)
    ->  DDI Layer 1 fires here (evidence modifier)
Response Scoring (4-dimension formula)
    ->
Controlled Enrichment (hypothesis tier only)
    ->
UI (5 safety surfaces + Conflict Review tab)
```

### Deduplication Strategies

| Strategy | When it fires | Action |
|-------------------|------------------------------------------------------|------------------|
| Exact | same entity + value + date within 7 days | merge + provenance |
| Semantic | alias (HTN=hypertension) or embedding similarity | link canonical term |
| Time-series | same entity, different date | append timeline |
| Conflict | value / type / drug / temporal / implausible change | quarantine both |

See `mkb/deduplication_engine.py` for the full rule set and
`ARCHITECTURE.md` for all 18 edge cases.

### Conflict Review Workflow

1. Deduplication engine detects conflict -> `ConflictResolver.quarantine_conflict(...)`.
2. Both facts move to `tier='quarantined'`, a conflict row is inserted.
3. User opens **Conflict Review** tab in the Streamlit UI.
4. User chooses `fact1 / fact2 / both / merge / neither` and leaves reasoning.
5. `resolve_conflict(...)` flips record tiers, writes an audit log entry, and
   optionally creates a merged record.

### Phase 2 Acceptance Criteria

The Phase 2 implementation is considered valid when all of the following hold:

1. Existing Phase 1 tests still pass.
2. Phase 2 validation tests pass for:
   valid spaCy extraction, valid Gemini extraction, missing required fields,
   low confidence, Gemini fallback violation, and malformed payloads.
3. Pipeline runs return extraction output plus validation status.
4. `data/review/review_queue.jsonl` contains only `needs_review` or `rejected` items with reasons.
5. Metrics snapshots include:
   `accepted_count`, `review_count`, `rejected_count`,
   `validation_error_count`, and `avg_confidence_by_status`.

## Feature Flags (config.py)

| Flag | Default | Purpose |
|------|---------|---------|
| `ENABLE_GRAPH` | False | Knowledge graph (Phase 3) |
| `ENRICH_PROMOTE` | False | Auto-promote hypothesis->active (Phase 2) |
| `ACTIVE_CONNECTORS` | `["dxgpt"]` | Expand in Phase 2 |
| `ENABLE_ENRICHMENT` | True | AI response enrichment |
| `ENABLE_WEB_INGESTION` | True | Web fetch pipeline |

## MVP Exit Criteria

All 8 golden test cases must pass before running against real medical data:

1. Real PDF -> structured SQLite records within 60s
2. PII strip verified: no names/dates/facilities in outbound payload
3. Conflict: two contradicting facts -> single resolved record + ledger entry
4. DDI HIGH block: incompatible drugs -> blocked + user confirmation UI
5. Claude offline -> safe mode banner + MKB-only response
6. Low score response -> discarded + ledger entry
7. PII strip: patient name/DOB/facility not present in extracted text
8. Trust hierarchy: clinical doc beats AI fact, confidence=0.95

```bash
make test-golden
# All 8 must pass before using with real data
```

---

## Specialties

| Specialty | Status | Primary connector |
|-----------|--------|-------------------|
| Neurology | Active | DxGPT |
| Epilepsy | Active | SAGE (stub->activate Phase 2) |
| Gastroenterology | Stub | DxGPT (Phase 2) |
| Urology | Stub | DxGPT (Phase 2) |

Adding a new specialty: create `specialties/[name]/config.yaml` + `plugin.py`. Zero core changes.
