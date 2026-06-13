# MEDAI-VERTEX-SEMANTIC-BILINGUAL-LABEL-CALIBRATION-15X-R3

- bilingual_label_calibration_passed: `True`
- exact / declared-alias / unicode-alias pass: `1` / `2` / `1`
- undeclared / unrelated / wrong-section rejected: `2` / `1` / `1`
- cal_cyrillic natural label supported / undeclared drift rejected: `True` / `True`
- evidence_anchor_strictness_preserved: `True` | cal_uncertainty_protection_preserved: `True`
- fuzzy_or_embedding_or_llm_label_judge_used: `False`
- live_call_made: `False` | external_api_used: `False` | active_written_count: `0` | auto_accept_true_count: `0`
- privacy_result: `passed` | billing_check_pending: `True`

## Decision

- Candidate-label matching now accepts the canonical label or an explicit, locally-declared,
  deterministic alias (whitespace/Unicode-normalized). Undeclared/inferred/fuzzy labels are rejected.
- cal_cyrillic declares aliases `pH` and `рН` for canonical `pH (bilingual)`.
- Evidence anchoring stays strictly verbatim (15X-R1); paraphrase and cal_uncertainty protection intact.
- No embeddings, LLM judge, or fuzzy similarity.
