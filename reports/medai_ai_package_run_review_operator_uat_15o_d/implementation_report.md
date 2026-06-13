# MEDAI-AI-PACKAGE-RUN-REVIEW-OPERATOR-UAT-15O-D

- UAT method used: `bounded_source_reachability_plus_deterministic_view_model_markdown_uat`
- Streamlit live launch used: `False`
- Preview entry point reachable count: `4`
- Package families checked: `['cytology_pathology_narrative', 'urinalysis_table_like_lab', 'portal_result_cards', 'mixed_narrative_numeric_result']`
- Source visible body present count: `4`
- Evidence anchor present count: `4`
- Candidate facts separated count: `4`
- Unknown values explicit count: `4`
- Uncertainty flags visible count: `4`
- No-live indicator visible count: `4`
- Active written count indicator visible count: `4`
- Auto-accept false indicator visible count: `4`
- Under 1 minute compare preserved count: `4`
- Hallucinated field count: `0`
- Live call made: `False`
- External API used: `False`
- Active written count: `0`
- Auto-accept: `False`
- Review required: `True`
- Privacy result: `passed`
- Billing check pending: `True`
- Push status: `pending`

## Scope

- Bounded deterministic UAT; no live Streamlit/browser launch was required.
- Uses the real Run & Review preview hook plus generated markdown previews.
- Does not change OCR routing, thresholds, providers, medical decision logic, or MKB writes.
