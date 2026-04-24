# MedAI Continuation Snapshot

## Current State

* main contains Phase 1 execution pipeline
* Phase 1.1 Gemini activation complete
* Phase 1.2 hardening + observability complete
* Phase 1 completion snapshot exists

## Current Architecture

* ExecutionPipeline is active
* Routing:

  * short/clean text -> spaCy
  * long/noisy text -> Gemini
* extractor_route and extractor_actual are logged
* Gemini fallback is guarded
* PII stripping uses Presidio
* MKB write path uses SQLiteStore + VectorStore
* outcomes: written / queued_for_review / blocked_ddi

## Verified Metrics

* 50-document validation passed
* success rate: 100%
* review count after optimization: 4/50
* repeat ingestion review count: 0
* Gemini real activation validated:

  * extractor: gemini
  * entity count: 9
  * confidence: 0.8
  * outcome: written
  * no rules_based fallback

## Environment Requirements

* GEMINI_API_KEY must exist in local .env
* .env must never be committed
* recommended key source: Google AI Studio
* GCP billing project is not required for Phase 1

## Current Branch / Repo Status

* work is merged to main
* latest completed phase: Phase 1.2
* tests passing: test_execution.py, 7 passed

## Known Limitations

* OCR/scanned PDFs not implemented
* Russian/multilingual path not implemented
* SQLCipher not active
* Docling unavailable or not part of Phase 1 proof
* Gemini remains an external dependency

## Next Recommended Phase

Phase 2: OCR / scanned PDF support

Recommended order:

1. Add OCR capability
2. Detect scanned/image-only PDFs
3. Route scanned PDFs to OCR before extraction
4. Validate English scanned PDFs first
5. Russian OCR only after English OCR path is stable

## Do Not Do Next

* do not add UMLS yet
* do not add Russian support before OCR baseline
* do not redesign ExecutionPipeline
* do not change routing thresholds without metrics

## Recovery Procedure

1. git pull main
2. ensure .env exists with GEMINI_API_KEY
3. run:
   python -m pytest test_execution.py
4. run one Gemini-routed validation document
5. confirm no gemini_route_legacy_fallback=rules_based note
