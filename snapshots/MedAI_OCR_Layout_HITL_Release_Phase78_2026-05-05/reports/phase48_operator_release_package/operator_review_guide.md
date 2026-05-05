# Operator Review Guide

This is not a medical device and not clinical diagnosis. Use this release only as a document intake and review assistant.

## Status Meanings

- `accepted`: The item passed existing extraction confidence and safety gates. Operators may use it as system-accepted intake output, but clinical use still requires appropriate human responsibility.
- `review`: The item is not accepted and requires human review. This is the top-level non-accepted bucket.
- `review_ocr_quality`: A subset of `review` where OCR/input quality or legacy OCR diagnostics remain the primary concern.
- `empty`: A subset flag under `review` indicating empty or near-empty extraction. Empty extraction must never be trusted.

Count convention: `total == accepted + review`. `review_ocr_quality` and `empty` are subsets of `review`, so they are not added to the top-level total.

## Operator Actions

- For `accepted`: verify the context before downstream use. Do not treat accepted as a clinical diagnosis.
- For `review`: inspect the source document, reason codes, extracted entities, confidence score, and audit trail.
- For `review_ocr_quality`: inspect scan quality first. Re-scan, improve OCR, or manually transcribe if needed.
- For `empty`: treat as no usable extraction. Do not rely on the output.

## Special Review Signals

- `lab_table_recovered`: deterministic lab row recovery found enough structure to move OCR-quality review to ordinary review. It still requires human review and never means accepted.
- `cyrillic_non_lab_document_review`: OCR quality was good enough to identify a Cyrillic non-lab document such as a prescription. Review content manually; do not auto-trust.
- `review_ocr_quality`: prioritize image/OCR quality and source legibility.
- `empty extraction`: verify the source file, OCR availability, and whether the document is blank, unsupported, or too degraded.
- `runtime drift warning`: treat as benchmark drift, not release improvement. Re-run the Phase46/47 validation commands and compare against the frozen baseline.

## Never Auto-Trust

Never auto-trust low-confidence extraction, poor OCR, empty extraction, review-only lab recovery, Cyrillic non-lab reconciliation, or any output produced during runtime drift. These are review signals, not acceptance signals.
