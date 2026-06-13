# 15X-R1 Retry Recommendation

## Decision: **READY_TO_RERUN_15X_ONCE**

Rationale:

- Strict verbatim matcher behaves correctly on all calibration cases: `True`.
- Recorded 15X cal_uncertainty paraphrase stays a failure; a verbatim source span would pass: `True`.
- Prompt contract hardened (verbatim required, null+uncertainty fallback): `True` / `True`.
- No embeddings / LLM judge used; paraphrase still rejected: `True`.

The prompt now instructs Vertex to copy evidence_text verbatim (or null + uncertainty). A single bounded 15X live re-run is appropriate; the validator remains strict and will stop on the first non-verbatim/paraphrased evidence.

Note: re-running 15X live is a separate, explicitly-gated operator action; this block does not run it.
