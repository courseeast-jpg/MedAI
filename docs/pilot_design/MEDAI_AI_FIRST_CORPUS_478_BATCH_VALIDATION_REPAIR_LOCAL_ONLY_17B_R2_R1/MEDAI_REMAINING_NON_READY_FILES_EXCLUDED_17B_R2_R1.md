# MEDAI Remaining Non-Ready Files Excluded 17B-R2-R1

This block processes exactly the 478 ready tokenized documents. All non-ready files are
excluded:

- Duplicates.
- Unsupported file types.
- Remaining `extraction_unavailable` files.
- Any file not marked ready by 17A-R2 / 17B-R2.

No excluded file is uploaded, tokenized for outbound, or sent to any provider. No
original source files (PDF/image/OCR) are processed. Temporary technical debt is accepted
for UI and normalization work, not for privacy. Privacy gates are never deferred.
