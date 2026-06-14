# 17C-R2 Adaptive Chunk + Cost Policy (R10)

## Cost model
Cost is estimated conservatively at the **full output ceiling** — every response is
assumed to use the maximum output tokens. This is the cap-governing worst case the live
runner enforces; real responses (with compact rules) will be smaller.

- input rate: $0.075 / 1M tokens
- output rate: $0.30 / 1M tokens
- output ceiling: 8192 tokens (raised from 2048)

## Hard caps (unchanged)
- total cap: **$0.40** (unchanged unless the user separately authorizes more)
- per-chunk cap: **$0.05**

## Adaptive chunk-size planning (`execution/cost_chunk_planner.py`)
`select_chunk_size` returns the largest chunk size in `[1, 25]` whose worst-case cost
stays within the $0.05 per-chunk cap. Worst-case for a chunk of size n is bounded by the
sum of the n most expensive requests, so **any** contiguous chunk of that size is within
the cap. If even a single request exceeds the per-chunk cap, it returns 0 and the runner
blocks. **Total request count never changes** — only the chunk size adapts.

## Result at the 8192 ceiling (478 requests)
- selected chunk size: **17** (from 25); per-chunk worst case within $0.05.
- worst-case total: **above $0.40** -> `estimated_total_within_authorized_cap=false`.

## Decision rules
- If worst-case total and per-chunk both within caps -> `ready_to_resume_17c_r2_live=true`.
- If per-chunk fits but worst-case total exceeds the cap (this case) ->
  `requires_new_cost_authorization=true`; recommend a revised total cap or a separately
  authorized smaller live batch. The runner's total-cap guard will block a live run until
  the worst-case total is within an authorized cap.
