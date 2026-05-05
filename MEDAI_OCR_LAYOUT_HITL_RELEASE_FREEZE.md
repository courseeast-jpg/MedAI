# MedAI OCR/Layout HITL Release — FROZEN

**Release Name:** MedAI OCR/Layout HITL Release
**Release Status:** FROZEN_HITL_RELEASE
**Snapshot ID:** MedAI_OCR_Layout_HITL_Release_Phase78_2026-05-05
**Snapshot Date:** 2026-05-05
**Commit Hash:** ``

---

## Release is FROZEN

This release snapshot is frozen and ready for local HITL use.
No further changes to OCR routing, extraction logic, thresholds,
or safety gates are included in this snapshot.

## What is included

- Full pipeline code (`extraction/`, `execution/`, `ingestion/`, `monitoring/`)
- Streamlit UI (`app/`)
- All test suites (`tests/`)
- Safe public phase reports (`reports/phase*/`)
- Release docs (`RELEASE_*.md`, `MEDAI_*.md`)
- Phase 74/75 Review Package
- Phase 76 one-click final validation
- Phase 77 operator release polish

## What is excluded

- Medical PDFs and images
- Private operator feedback and filename mappings
- `.env` / secrets
- Local input folders with real patient data
- OCR text dumps or extracted medical text

## Safety statements

- **This is not a medical device and does not provide clinical diagnosis.**
- **Not production-autonomous** — human review is required before downstream use.
- **Local-only** — `MEDAI_LOCAL_ONLY=true` is enforced by default.
- **Manual-review boundary is retained** — no diagnostic evidence justified
  changing OCR routing, extraction, or safety gates in this release.
- **Operator feedback is deferred** — no labels were fabricated.

## Quick start

```bash
streamlit run app/main.py
python scripts/run_phase76_one_click_final_validation.py
python -m pytest tests
```
