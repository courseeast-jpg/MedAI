# MEDAI-VERTEX-SEMANTIC-EVIDENCE-MATCH-CALIBRATION-15X-R1

- evidence_match_calibration_passed: `True`
- exact / whitespace / substring pass: `1` / `1` / `3`
- paraphrase / inferred / wrong-section / null-empty rejected: `2` / `1` / `1` / `2`
- recorded_15x_failure_preserved: `True`
- prompt_contract_hardened: `True` | evidence_text_verbatim_required: `True`
- embedding_or_llm_judge_used: `False`
- live_call_made: `False` | external_api_used: `False` | active_written_count: `0` | auto_accept_true_count: `0`
- privacy_result: `passed` | billing_check_pending: `True`

## Decision

- Evidence anchoring stays STRICT: only a whitespace/Unicode-normalized verbatim substring of the
  source body (and section, when constrained) is accepted. Paraphrase/synonym/inferred evidence is rejected.
- The Vertex semantic prompt contract now requires evidence_text copied verbatim, or null + uncertainty.
- No embeddings, no LLM judge, no fuzzy 'close enough' matching.
