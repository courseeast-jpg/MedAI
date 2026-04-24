### System Status

- ExecutionPipeline active
- Routing: spaCy (<3000 chars) / Gemini (>=3000 or noisy)
- PII stripping: Presidio
- Storage: SQLite + VectorStore
- Outcome states: written / queued_for_review / blocked_ddi

### Verified Behavior

- spaCy path working
- Gemini path working (no fallback)
- Routing correct
- MKB writes verified
- Duplicate/review inflation fixed

### Metrics (example baseline)

- avg_latency ~= 800-1200 ms
- review_rate < 20%
- gemini usage ~10-20%

### Environment Requirements

- GEMINI_API_KEY in local .env
- .env not committed
- AI Studio key (not GCP)

### Observability

- extractor_route
- extractor_actual
- metrics snapshot available

### Known Limitations

- OCR not implemented
- Russian/multilingual not validated
- SQLCipher disabled
- Gemini dependency external

### Recovery Instructions

- git pull main
- ensure .env exists with GEMINI_API_KEY
- run pytest
- run pipeline on test_data
