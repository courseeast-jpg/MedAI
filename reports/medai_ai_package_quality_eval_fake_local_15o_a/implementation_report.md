# MEDAI-AI-PACKAGE-QUALITY-EVAL-FAKE-LOCAL-15O-A

- Baseline commit: `b4a409a85a27`
- Report generation commit: `b4a409a85a27`
- Privacy result: `passed`
- Package families evaluated: `['cytology_pathology_narrative', 'urinalysis_table_like_lab', 'portal_result_cards', 'mixed_narrative_numeric_result']`
- Under 1 minute compare pass count: `4`
- Hallucinated field count: `0`
- Live call made: `False`
- External API used: `False`
- Active written count: `0`
- Auto-accept: `False`
- Review required: `True`
- Billing check pending: `True`
- Push status: `pending`

## Scope

- Deterministic fake/local fixtures only.
- Scores operator review-package usability, not clinical correctness.
- Does not call Gemini, Vertex, AI Studio, Claude, OpenAI, Ollama, or local LLM providers.
- Does not mutate runtime databases or write active MKB records.
