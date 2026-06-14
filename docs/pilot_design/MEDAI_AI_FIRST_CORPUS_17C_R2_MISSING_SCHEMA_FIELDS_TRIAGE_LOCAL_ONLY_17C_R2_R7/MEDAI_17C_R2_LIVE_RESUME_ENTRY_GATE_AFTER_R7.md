# MEDAI 17C-R2 Live Resume Entry Gate After R7

17C-R2 live extraction is NOT resumed by this block.

## Resume Policy

The 17C-R2 runner has no per-request checkpoint, so a resume re-sends from the first
request (request #1 is sent again). Resume policy = `restart_required`. Checkpointing is
deferred to a separate guarded block.

## A Resume May Proceed Only When ALL Of The Following Hold

1. The required-fields prompt skeleton and the stricter all-core-keys validation are in
   place (this block); schema unchanged; no field inferred/synthesized.
2. The canonical 478 batch is present in the operator context and sealed-valid.
3. The Vertex credential preflight passes (project = sot-knowledge-ocr).
4. Explicit user authorization remains in force; caps $0.05/chunk and $0.40 total with
   stop-on-first-failure and the live gate scoped per chunk.

## Caveat

The exact missing fields for the prior failure could not be confirmed (body
unavailable). The prompt skeleton instruction targets the general cause (model omitting
required keys). Watch request #2 on the next run; if it fails again, capture the private
response body and re-triage with the precise missing keys.
