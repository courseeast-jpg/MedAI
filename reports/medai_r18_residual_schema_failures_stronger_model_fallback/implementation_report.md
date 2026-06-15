# MEDAI-R18-RESIDUAL-SCHEMA-FAILURES-STRONGER-MODEL-FALLBACK — implementation report

- Stronger model: `gemini-2.5-pro` via the existing Vertex adapter model override; reuses the full R17 salvage/skeleton/minimal contract + checkpoint/evidence/cost/redaction/no-MKB stack. response_schema not used; JSON mode on.
- Residual selection: Corpus 1 `337` (content before R18 `141`, not reprocessed); Corpus 2 `2` (21 not resent, 2 RTF/signal excluded).
- Estimated cost (worst-case @8192 pro): `$27.951877` within remaining `$46.966666`.
- Result: `PASS_PARTIAL`; Corpus 1 succeeded `18` (full `16`, minimal-review `2`), fast_fail `False`; Corpus 2 succeeded `0`.
- tokens `774692`, cost `1.93673`, same-day total `4.970064` (cap $50).
