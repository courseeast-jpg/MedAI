# MEDAI 17C-R2 Live Resume Entry Gate After R6

17C-R2 live extraction is NOT resumed by this block.

## Resume Policy

The 17C-R2 live runner has no per-request checkpoint: it sends sequentially from the
first request each run. Therefore a resume is a RESTART of the whole batch (request #1
will be sent again). Resume policy = `restart_required`.

## A Resume May Proceed Only When ALL Of The Following Hold

1. The strict-JSON prompt hardening and the safe normalizer are in place (this block).
2. The schema is unchanged; no partial JSON accepted; no fields inferred.
3. The canonical 478 batch is present in the operator context and sealed-valid.
4. The Vertex credential preflight passes (project = sot-knowledge-ocr).
5. Explicit user authorization remains in force; caps stay $0.05/chunk and $0.40 total
   with stop-on-first-failure and the live gate scoped per chunk.

## Caveat

The exact subclass of the prior not_strict_json failure could not be confirmed (the
failed response body was unavailable). If the failure recurs at request #2 after resume,
capture the private response body and re-triage (it may be prose/truncated/safety-refusal
rather than a Markdown fence, which the normalizer intentionally does not recover).
