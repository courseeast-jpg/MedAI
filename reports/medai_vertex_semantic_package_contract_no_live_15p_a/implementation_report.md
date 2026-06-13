# MEDAI-VERTEX-SEMANTIC-PACKAGE-CONTRACT-NO-LIVE-15P-A

- Baseline commit: `b36453640854`
- Report generation commit: `b36453640854`
- Privacy result: `passed`
- Provider route: `vertex`
- Model: `gemini-2.5-flash-lite`
- Package families checked: `['cytology_pathology_narrative', 'urinalysis_table_like_lab', 'portal_result_cards', 'mixed_narrative_numeric_result']`
- Schema validation pass count: `4`
- Fake Vertex response valid count: `4`
- Source visible body preserved count: `4`
- Evidence anchor preserved count: `4`
- Candidate facts separated count: `4`
- Unknown values explicit count: `4`
- Uncertainty flags visible count: `4`
- Hallucinated field count: `0`
- Live call made: `False`
- External API used: `False`
- Active written count: `0`
- Auto-accept: `False`
- Review required: `True`
- Billing check pending: `True`

## Scope

- Defines a no-live Vertex semantic package request and response contract.
- Uses only deterministic synthetic 15O package fixtures and fake provider responses.
- Does not change OCR routing, thresholds, provider gates, medical decision logic, or MKB write logic.
- Does not require or read the Vertex live smoke gate.
