# MEDAI Remaining Non-Ready Files Not Included In 17C-R2

17C-R2 processes exactly the canonical 478 validated tokenized requests. All non-ready
files are excluded and are not uploaded, tokenized for outbound, or sent to any provider:

- Blocked files.
- Duplicate files.
- Remaining extraction_unavailable files.
- Unsupported file types.

No original source files (PDF/image/OCR) are processed. Temporary technical debt is
accepted for UI and normalization work, not for privacy. Privacy gates are never
deferred.
