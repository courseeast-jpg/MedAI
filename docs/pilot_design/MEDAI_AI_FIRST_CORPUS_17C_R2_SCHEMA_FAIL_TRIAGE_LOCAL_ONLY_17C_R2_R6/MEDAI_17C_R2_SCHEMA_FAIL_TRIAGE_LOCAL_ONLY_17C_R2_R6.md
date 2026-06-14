# MEDAI 17C-R2 Schema-Fail Triage Local-Only 17C-R2-R6

## Status

- Local-only triage. No Gemini/Vertex/Claude/OpenAI model call, no provider content
  request, no billing API call, no live gate, no live extraction, no new corpus request.
- No MKB open/write, no auto-accept, no medical decision, no production queue mutation.

## Context

The first real 17C-R2 live send sent 2 requests: request #1 succeeded; request #2
failed strict-JSON validation (failure_stage=schema, failure_category=not_strict_json);
the run stopped on first failure. No MKB write occurred.

## Evidence Availability

The private staged raw responses needed to classify the exact `not_strict_json`
subclass of response #2 are not present/retained in this operator context (the private
staging reflects an earlier blocked run; an external writer has been clearing
`MedAI_Private`). The exact subclass is therefore reported as `unknown_not_strict_json`.
This block does not fabricate or print any response body.

## Forward Fixes (do not weaken schema)

1. Prompt hardening: the prompt contract now explicitly requires ONE JSON object only,
   no Markdown fences, no prose, first char `{` and last `}`, and empty arrays/null when
   uncertain.
2. Safe response normalizer (`execution/strict_json.py`): strips ONE clean surrounding
   Markdown fence around a single JSON object and rejects prose-wrapped, multiple-object,
   and truncated responses. Wired into the 17C-R2 schema validator before the strict
   field + privacy checks. No partial JSON accepted; no missing fields inferred; schema
   unchanged.
3. JSON mode: the 17C-R2 request already sets `responseMimeType: application/json`.
4. Stale cap test updated to the authorized $0.40 total / $0.05 per-chunk caps.

## Verification

A synthetic local replay proves the normalizer recovers a fenced single JSON object and
rejects prose/multiple/truncated/empty responses. The real failed body could not be
replayed (unavailable in context).
