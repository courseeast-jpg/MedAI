# Review Package UI — Phase 75 (SAFE)

## One command to view

```bash
# Standalone review package panel
streamlit run app/review_package_viewer.py

# Full app with Review Package tab
streamlit run app/main.py
```

## What you will see

- Phase74 conclusion and package summary.
- 6 review buckets, sorted by priority.
- Per-bucket explanation: why files are in review, what the system knows,
  what is unknown, safest next action.
- Safe IDs sample (no PHI, no raw filenames, no raw paths).
- Plain-language status: no manual review required to continue.

## Safety rules

- No production OCR/extractor changes are displayed or recommended.
- No private files are opened or displayed.
- All content comes from the Phase74 safe public package only.
