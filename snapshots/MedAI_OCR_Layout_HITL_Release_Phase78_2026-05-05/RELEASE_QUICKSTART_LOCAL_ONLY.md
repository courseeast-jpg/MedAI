# MedAI Quick Start — Local-Only HITL Mode

**This system is local-only. No data leaves your machine by default.**

---

## 1. Start the UI

```bash
streamlit run app/main.py
```

Opens at http://localhost:8501

Tabs available:
- **Current Run** — upload PDFs and run extraction
- **Blind Audit** — run the blind generalization audit
- **Report Archive** — browse previous safe reports
- **Review Package** — view the Phase 74 auto-grouped review buckets

---

## 2. Run a document through the pipeline

1. Open **Current Run** tab
2. Upload a PDF or TXT file
3. Review the result: `accepted` / `review` / `review_ocr_quality` / `empty`
4. If `accepted`: spot-check values against source before downstream use
5. If `review` or `review_ocr_quality`: human review is required

---

## 3. View the Review Package

The Review Package explains all pending review cases without requiring
you to open individual documents.

```bash
# Standalone Review Package panel
streamlit run app/review_package_viewer.py

# Or use the Review Package tab in the main UI
streamlit run app/main.py
```

No manual document-by-document truth labeling is required to continue.

---

## 4. Run final validation

```bash
python scripts/run_phase76_one_click_final_validation.py
```

This verifies:
- All Phase 47/48/49/75 readiness checks pass
- No tracked medical artifacts or private mappings
- Public reports contain no PHI or raw paths

---

## 5. Run the test suite

```bash
python -m pytest tests
```

Expected: 782+ passing, 0 failing.

---

## Safety reminders

- **Local-only:** MEDAI_LOCAL_ONLY=true is enforced by default.
- **Human review is required** before any extracted fact is used downstream.
- **This is not a medical device** and does not provide clinical diagnosis.
- **Not production-autonomous** — the system requires operator oversight.
- The **manual-review boundary is retained** for all OCR and extraction decisions.
