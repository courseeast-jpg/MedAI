# MEDAI 17C-R2 Authorized Cost Cap Update 17C-R2-R5

## Status

- Local/config/preflight only. No Gemini/Vertex/Claude/OpenAI model call, no provider
  content request, no billing API call, no live gate, no live extraction.
- No MKB open/write, no auto-accept, no medical decision, no production queue mutation.

## Change

The user authorized raising the 17C-R2 total cost cap from $0.25 to $0.40. The per-chunk
cap remains $0.05; chunk size remains 25; model remains gemini-2.5-flash-lite;
stop-on-first-failure, no-MKB-write, no-auto-accept, no-medical-decision, and the
per-chunk live gate scoping are all unchanged.

## Why

17C-R2 loaded all 478 validated requests and blocked before any provider call because
the local estimate `estimated_total_cost_before_run_usd = 0.325362` exceeded the old
$0.25 total cap. The newly authorized $0.40 total cap covers that estimate while keeping
the per-chunk cap unchanged. The estimate is approximate and local; no billing API is
called.

## Scope

Only the total cap constant in the 17C-R2 live runner is changed. Request batch
contents, privacy validation, response schema, and the prompt contract are unchanged.
