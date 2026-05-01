# Final Validation Report

## System Info

- latest_commit_hash: `b5d5f93`
- total tests passed: `292`
- timestamp: `2026-05-01T10:38:46.3641988-04:00`

## Controlled Batch (Phase 32 Baseline)

- total_files: `15`
- accepted: `11`
- review: `4`
- empty: `1`
- accepted %: `73.33%`

## Holdout Validation (Phase 36)

- total_files: `8`
- accepted: `2`
- review_ocr_quality: `6`
- empty: `0`
- accepted %: `25.00%`

## Comparison

- accepted drop: `48.33` percentage points
- review increase: `48.33` percentage points
- explanation: Holdout acceptance dropped because the unseen documents contain OCR and multilingual noise. Phase 36 correctly routes low-quality OCR documents to `review_ocr_quality` instead of treating them as extraction failures or unsafe accepts.

## Failure Classification

- ocr_noise_remaining: `6`
- model/extraction failures: `0`

## System Capabilities

- Handles structured English lab reports reliably.
- Handles urinalysis, culture, and cytology documents via deterministic rule packs.
- Fallback system works without Gemini.
- Confidence scoring is functional and auditable.

## Known Limitations

- OCR noise remains the primary blocker.
- Foreign-language reports require stronger normalization or multilingual parsing.
- Low text coverage after scanning reduces confidence.
- System performance depends on text quality.

## Safety Guarantees

- No unsafe auto-accept on low-quality OCR.
- Empty extraction is prevented by deterministic fallback and review gating.
- Fallback behavior is deterministic when Gemini is unavailable or quota-limited.
- Review gating works for low-confidence and low-quality inputs.

## Operating Guidelines

- Use for digital PDFs or clean scans.
- Route noisy scans to manual review.
- Do not rely on the system for raw OCR documents without manual review.

## Release Decision

status = `VALIDATION_READY`

not `PRODUCTION_AUTONOMOUS`

Rationale: The system is validated for assisted use with review gating. It is not ready for fully autonomous production operation because noisy OCR and multilingual scanned documents still require manual review.

## Next Roadmap

- OCR improvement pipeline using external OCR/layout tooling.
- Language models or rules for multilingual medical parsing.
- Image-based extraction in a future phase.
