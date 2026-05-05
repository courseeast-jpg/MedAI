# MedAI OCR/Layout HITL Release — Operator Guide

**Release:** MedAI OCR/Layout HITL Release  
**Status:** FROZEN HITL RELEASE — local-only, human-in-the-loop, not production-autonomous

---

## What do I do next?

You do not need to review individual documents to use this system.
The automated Review Package groups all pending review cases by type and provides
plain-language explanations of what the system knows and what it does not.

**Start here:**

```bash
# View the Review Package in the Streamlit UI
streamlit run app/main.py

# Or view the standalone Review Package panel
streamlit run app/review_package_viewer.py

# Or run the one-click final validation
python scripts/run_phase76_one_click_final_validation.py
```

---

## What this system does

MedAI extracts structured lab data from local medical PDF documents.
It routes each document to one of: `accepted`, `review`, `review_ocr_quality`, or `empty`.

- **accepted** — extraction passed confidence and safety gates. Spot-check before use.
- **review** — manual reconciliation required before relying on output.
- **review_ocr_quality** — OCR quality was insufficient. Do not trust extraction.
- **empty** — no usable extraction. Check document quality or format.

---

## Privacy status

- **Local-only:** All processing runs locally. No data is sent to external APIs by default.
- **PHI protection:** Public reports contain only safe IDs and aggregate counts — no raw filenames, paths, or patient data.
- **Private files are gitignored:** `operator_feedback_PRIVATE.json` and private filename mappings are never committed.

---

## Review Package

The Review Package (Phase 74/75) groups all pending review items into 6 buckets:

| Priority | Bucket | What it means |
|---|---|---|
| 1 | OCR Quality Review | OCR gate triggered — defer until operator reviews |
| 2 | Empty Extraction Review | Extraction produced zero results — root cause varies |
| 3 | Unknown Document Class | Classifier below threshold — no change to thresholds |
| 4 | Possible Multi-Document PDF | Complex layout or bundle — do not split without validated examples |
| 5 | Unsupported/Deferred Format | .docx/.msg not yet supported — no production change needed |
| 6 | Completed Boundary Branches | Fully investigated and closed — no action required |

No manual document-by-document review is required to continue.

---

## What the system will NOT do

- Will not automatically accept review-routed documents.
- Will not change OCR routing, extraction logic, thresholds, or safety gates.
- Will not send data to external services.
- Will not provide clinical diagnosis or medical recommendations.

---

## Safety statements

> **This is not a medical device and does not provide clinical diagnosis.**
>
> This system is not production-autonomous. Human review is required before any extracted fact is used downstream.
>
> The manual-review boundary is retained. No diagnostic evidence justifies changing OCR or extraction behavior in this release.
>
> Local-only mode is the default and enforced setting.

---

## Safe report locations

| Report | Path |
|---|---|
| Review Package JSON | `reports/phase74_manual_review_package_auto_improvement/manual_review_package_SAFE.json` |
| Review Package MD | `reports/phase74_manual_review_package_auto_improvement/manual_review_package_SAFE.md` |
| Phase76 Final Validation | `reports/phase76_one_click_final_validation/phase76_one_click_final_validation_report.json` |
| Phase78 Release Snapshot | `reports/phase78_final_hitl_release_freeze/` |
