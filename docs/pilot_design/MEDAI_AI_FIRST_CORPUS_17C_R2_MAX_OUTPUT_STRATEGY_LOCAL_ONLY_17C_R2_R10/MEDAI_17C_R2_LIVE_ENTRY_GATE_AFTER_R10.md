# 17C-R2 Live Entry Gate (after R10)

Before any future authorized live run, the gate (computed locally, no provider/billing
call) requires all four:

| Gate | Requirement | Current |
| --- | --- | --- |
| `canonical_batch_valid` | canonical 478 batch resolves and is readable | true |
| `credential_preflight_passed` | ADC refresh succeeds (no model call) | true |
| `estimated_total_within_authorized_cap` | worst-case total at 8192 ceiling ≤ $0.40 | **false** |
| `estimated_chunks_within_per_chunk_cap` | per-chunk worst case ≤ $0.05 at selected size | true |

`ready_to_resume_17c_r2_live` = logical AND of all four.

## Current snapshot
- new_max_output_tokens: 8192 (raised from 2048)
- selected_chunk_size: 17 (from 25), per-chunk worst case within $0.05
- worst-case total at the 8192 ceiling for all 478 requests: above $0.40
- **ready_to_resume_17c_r2_live: false**
- **requires_new_cost_authorization: true**

## What unblocks live
One of:
1. A revised total-cost authorization that covers the worst-case total at the 8192
   ceiling, or
2. A separately authorized smaller live batch (fewer requests in a single authorized run),
   keeping the per-chunk cap at $0.05 and the total within whatever cap is authorized.

The total request count is unchanged by design; the only levers are the cap or the
authorized batch size. R8 checkpoint/resume and failed-evidence preservation, and R9
public-report redaction, remain in force.
