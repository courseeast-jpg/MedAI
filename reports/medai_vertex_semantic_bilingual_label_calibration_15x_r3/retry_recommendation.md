# 15X-R3 Retry Recommendation

## Decision: **READY_TO_RERUN_15X_R4_ONCE**

Rationale:

- Declared bilingual aliases (`pH`, `рН`) for canonical `pH (bilingual)` accepted; undeclared/unrelated drift rejected: `True`.
- cal_cyrillic natural source-derived label (`pH`) now supported; undeclared drift still fails: `True` / `True`.
- Verbatim evidence anchoring preserved (paraphrase still fails, incl. cal_uncertainty): `True`.
- No fuzzy / embeddings / LLM label judge: `True`.

Label matching now accepts only explicit, locally-declared, deterministic source-derived aliases while evidence remains strictly verbatim. A single bounded 15X re-run (R4) is appropriate; the validator still stops on undeclared label drift or non-verbatim evidence.

Note: re-running 15X live is a separate, explicitly-gated operator action; this block does not run it.
