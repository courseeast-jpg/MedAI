# MEDAI JSONL Loader Hardening And Credential Preflight 17C-R2-R2

## Status

- Local/preflight only. No Gemini/Vertex/Claude/OpenAI model call, no provider content
  request, no billing API call, no live gate, no live extraction.
- No MKB open/write, no auto-accept, no medical decision, no production queue mutation.

## Purpose

Fix the false JSONL integrity blocker that stopped 17C-R2: the loaders used
`str.splitlines()`, which splits on Unicode line separators (U+2028/U+2029/U+0085)
that can appear inside JSON string values, over-splitting one record into fragments.
17C-R2-R1 proved the canonical 478 batch was never corrupted and rebuilt it with
`ensure_ascii=True`, a SHA256 sidecar, a doc-id manifest, and read-only protection.

## What This Block Does

1. Adds a shared physical-newline JSONL reader (`execution/jsonl_framing.py`) that
   frames records on `"\n"` only, trims a trailing `"\r"`, and skips only a final
   empty line — never `splitlines()`.
2. Hardens the 17C-R2 live batch loader and the 17B-R2-R1 JSONL framing loads to use
   the shared reader. Validation rules are not weakened.
3. Verifies the sealed canonical 478 batch (SHA256 match + 478/478/478/0 + 0 residual
   PI) using physical-newline framing.
4. Runs a Vertex credential preflight (ADC refresh, project check) with no model call
   and no token printed.

## Privacy

Private outbound bodies, raw OCR, token maps, PI values, and credentials are never
committed or printed. Public reports carry counts/hashes/status only.
