# MEDAI-AI-PACKAGE-QUALITY-RUN-REVIEW-WIRING-15O-C

- Baseline commit: `c675fed6d5dd`
- Report generation commit: `c675fed6d5dd`
- Privacy result: `passed`
- Run & Review package preview entry point added: `True`
- Package families previewed: `['cytology_pathology_narrative', 'urinalysis_table_like_lab', 'portal_result_cards', 'mixed_narrative_numeric_result']`
- Source visible body present count: `4`
- Evidence anchor present count: `4`
- Candidate facts separated count: `4`
- Unknown values explicit count: `4`
- Uncertainty flags visible count: `4`
- Under 1 minute compare preserved count: `4`
- Hallucinated field count: `0`
- Live call made: `False`
- External API used: `False`
- Active written count: `0`
- Auto-accept: `False`
- Review required: `True`
- Billing check pending: `True`
- Push status: `pending`

## Scope

- Adds deterministic fake/local AI package review preview wiring to the Run & Review tab.
- Does not change OCR routing, thresholds, provider execution, medical decision logic, or MKB write logic.
- Does not require or read MEDAI_VERTEX_LIVE_SMOKE_ALLOWED.
