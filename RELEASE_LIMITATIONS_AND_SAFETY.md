# MedAI — Limitations and Safety Constraints

**Release:** MedAI OCR/Layout HITL Release  
**Status:** FROZEN HITL RELEASE

---

## This system is NOT a medical device

> MedAI does not provide clinical diagnosis, medical advice, or treatment recommendations.
> All extracted data must be reviewed by a qualified human before any downstream clinical use.

This system is **not production-autonomous**. Human review is required before any extracted
fact is used in a clinical, administrative, or research workflow.

---

## Local-only operation

- All document processing runs locally on your machine.
- No patient data, document content, or extraction results are sent to external APIs by default.
- `MEDAI_LOCAL_ONLY=true` and `MEDAI_ALLOW_EXTERNAL_API=false` are enforced by default.
- Private feedback files and filename mappings are gitignored and never committed.

---

## Manual-review boundary is retained

Multiple diagnostic phases (Phase 62, 67, 69, 70) investigated OCR quality and layout
improvements. None produced evidence sufficient to change production OCR routing,
extraction logic, confidence thresholds, or safety gates.

**The manual-review boundary is retained in this release.**

This means:
- Documents with borderline OCR quality remain routed to `review_ocr_quality`
- Empty extractions remain routed to `empty`
- Unknown document classes remain routed to `review`
- No automatic acceptance of borderline cases

---

## What the system cannot do

- Cannot guarantee extraction accuracy for all document types or layouts.
- Cannot handle DOCX, MSG, audio files, or other unsupported formats.
- Cannot extract from documents with severe OCR degradation.
- Cannot auto-classify documents with ambiguous or unknown document classes.
- Cannot produce clinical-grade output without human validation.

---

## Review Package (Phase 74/75)

The automated Review Package groups all pending review cases by type.
It provides plain-language explanations without requiring document-by-document review.

> No manual document-by-document truth labeling is required to continue using this system.

The Review Package is available at:
```
reports/phase74_manual_review_package_auto_improvement/manual_review_package_SAFE.md
```
Or in the **Review Package** tab of the main UI (`streamlit run app/main.py`).

---

## Operator feedback (deferred)

Operator feedback collection (Phase 72/72B/73) is deferred by user decision.
- Operator feedback is optional — not required to continue.
- No labels have been fabricated.
- Private feedback files are gitignored.

---

## Known test suite status

- Full suite: 782+ tests, 0 failures (as of Phase 75 commit adae069)
- Phase47/48/49 validations: all `ready`

---

## Data safety

Public reports in this repository contain:
- Safe file IDs only (e.g., `file_001`, `corpus_file_000004`)
- Aggregate counts
- Branch names and diagnostic conclusions

Public reports do NOT contain:
- Patient names, dates of birth, or other PHI
- Raw lab values or clinical text
- Raw filenames or file paths
- OCR output text
- Extracted medical facts
