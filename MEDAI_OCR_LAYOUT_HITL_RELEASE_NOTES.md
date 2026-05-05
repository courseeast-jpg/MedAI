# MedAI OCR/Layout HITL Release Notes

**Date:** 2026-05-05
**Commit:** ``
**Test count:** 782 passing

---

## Release summary

This release closes the OCR/Layout HITL diagnostic branch.
It improves the manual-review workflow without changing production behavior.

## What was improved

| Phase | Change |
|---|---|
| Phase 71 | Operator feedback prioritization — 15 items, 3 tier-1 |
| Phase 72/72B | Operator feedback collection console (CLI + Streamlit) |
| Phase 73 | Operator feedback bypass — deferred_by_user, no labels fabricated |
| Phase 74 | Manual review package auto-improvement — 6 buckets from diagnostics |
| Phase 75 | Review Package tab added to Streamlit UI |
| Phase 76 | One-click final validation script |
| Phase 77 | Operator-facing release docs (guide, quickstart, limitations) |
| Phase 78 | Final release snapshot and freeze |

## What did NOT change

- Production OCR routing
- Extraction logic
- Confidence thresholds
- Safety gates
- Privacy gates
- Acceptance behavior

## Diagnostic branches investigated and closed

| Branch | Outcome |
|---|---|
| PDF geometry header inference (Phase 62) | Closed — insufficient signal |
| PDF OCR preprocessing (Phase 67) | Closed — manual-review boundary retained |
| Image OCR preprocessing (Phase 69) | Closed — manual-review boundary retained |
| RTF local text parser (Phase 64/65) | Completed — no safety regression |
| DOCX triage (Phase 63) | Deferred — no priority evidence |

## Known limitations

See `RELEASE_LIMITATIONS_AND_SAFETY.md` for full details.
