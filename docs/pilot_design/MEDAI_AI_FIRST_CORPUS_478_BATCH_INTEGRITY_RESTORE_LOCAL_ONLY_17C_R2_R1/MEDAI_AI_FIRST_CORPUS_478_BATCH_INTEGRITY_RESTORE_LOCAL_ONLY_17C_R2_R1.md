# MEDAI AI-First Corpus 478 Batch Integrity Restore Local-Only 17C-R2-R1

## Status

- 17C-R2-R1 is local-only. No AI provider call, no live gate, no 17C live extraction.
- No MKB open/write, no auto-accept, no medical decision, no production queue mutation.

## Purpose

Restore the canonical 478-request private outbound batch after 17C-R2 blocked because the
private JSONL was corrupted/mutated (591 lines / 457 parseable / 457 unique / 134
malformed, expected 478). Add integrity sealing so future live runs verify line count,
parseability, doc-id count, SHA256, and manifest consistency before any provider call.

## Recovery Source Of Truth

The corrupted outbound JSONL is NOT trusted. The canonical 478 is rebuilt from the 17A
tokenized corpus plus the 17B-R2-R1 repair logic (imported, not duplicated), then the
JSONL is atomically replaced (truncate, never append).

## Validation After Rebuild

- exactly 478 lines, 478 parseable JSON objects, 478 unique doc IDs, 0 malformed
- request validation passed, 0 residual PI validator failures
- SHA256 + integrity sidecar + doc-id manifest written privately, outside the repo
- the rebuilt JSONL is set read-only after writing when supported

## Privacy

Private outbound bodies, raw OCR, token maps, and PI vault values are never committed.
Public reports carry counts, hashes, and status only.
