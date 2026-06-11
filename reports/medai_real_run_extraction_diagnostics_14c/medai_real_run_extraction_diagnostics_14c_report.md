# MEDAI-REAL-RUN-EXTRACTION-DIAGNOSTICS-14C

- Privacy result: `passed`
- External API used: `False`
- Auto-accept: `False`
- Real runtime diagnostics added: `True`
- OCR recovered counter present: `True`
- OCR text shape buckets present: `True`
- Extractor dispatch counter present: `True`
- Extraction candidate counter present: `True`
- Drop reason counts present: `True`
- Records written counter present: `True`
- Records deduped counter present: `True`
- Review-bound records written counter present: `True`
- Category propagated: `True`
- Specialty propagated: `True`
- Source modality preserved: `True`
- Raw OCR text in report: `False`
- Private paths in report: `False`

## Runtime Counters

- Files processed: `1`
- OCR attempted: `1`
- OCR available: `1`
- OCR recovered: `1`
- Extractor dispatch attempted: `1`
- Extraction candidates created: `7`
- Extraction candidates after filter: `7`
- Extraction candidates dropped: `0`
- Drop reason counts: `{}`
- Records written: `6`
- Records deduped: `0`
- Review-bound records written: `6`
- Next block recommendation if candidates zero: `Not required by this validation; extraction candidates were created.`

## Limitations

- Synthetic OCR validation only; no private images, OCR dumps, PDFs, or runtime databases are committed.
- OCR text shape is reported only as buckets and booleans; raw OCR text, filenames, and paths are excluded.
- This block adds diagnostics and minimal runtime wiring only; it does not add parsers or interpret clinical content.
