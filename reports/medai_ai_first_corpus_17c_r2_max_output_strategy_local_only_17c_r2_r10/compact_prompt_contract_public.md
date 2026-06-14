# 17C-R2 compact JSON output policy (R10)

Raised output ceiling: `2048` -> `8192` tokens.

Compact-output rules added to the prompt contract (schema NOT weakened):
- JSON object only — no explanatory text, preamble, or commentary.
- No duplicate evidence; no long copied passages.
- Evidence anchors must be short (minimal tokenized snippet).
- Omit optional verbose notes unless the schema requires them.
- Empty arrays for absent sections; every required top-level key still present.

Strict JSON (R6) and the required top-level skeleton (R7) remain in force.
