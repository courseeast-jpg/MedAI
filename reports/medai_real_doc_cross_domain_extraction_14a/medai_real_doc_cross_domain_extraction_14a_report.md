# MEDAI-REAL-DOC-CROSS-DOMAIN-EXTRACTION-14A

- Privacy result: `passed`
- External API used: `False`
- Auto-accept: `False`
- Table extraction primitive added: `True`
- Key-value extraction primitive added: `True`
- Card extraction primitive added: `True`
- Narrative section extraction primitive added: `True`
- Lab adapter added: `True`
- Portal-card adapter added: `True`
- Pathology/cytology adapter added: `True`
- Imaging narrative adapter added: `True`
- Consult/treatment section adapter added: `True`
- Synthetic table rows extracted: `3`
- Synthetic key-value pairs extracted: `4`
- Synthetic card rows extracted: `2`
- Synthetic narrative sections extracted: `4`
- Review-bound records created: `21`
- Accepted records: `0`
- Category propagated: `True`
- Specialty propagated: `True`
- Raw OCR text in report: `False`
- Private paths in report: `False`

## Limitations

- Synthetic validation only; no private source documents, images, PDFs, or OCR dumps are used.
- The primitives capture visible observations and source sections only; clinical meaning is not inferred from diagnosis, treatment, imaging, medication, or recommendation headings.
- Coverage is deterministic and conservative; real-world OCR quality and layout variation may still require operator correction.
