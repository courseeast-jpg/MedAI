# 17C-R2 live entry gate (after R10)

- canonical_batch_valid: `True`
- credential_preflight_passed: `True` (`pass`)
- new_max_output_tokens: `8192`
- estimated_total_cost_with_new_output_ceiling_usd: `1.228029`
- authorized_hard_cost_cap_total_usd: `0.4`
- estimated_total_within_authorized_cap: `False`
- original_chunk_size: `25` -> selected_chunk_size: `17` (adaptive_applied=`True`)
- estimated_per_chunk_cost_with_selected_chunk_size_usd: `0.04902`
- hard_cost_cap_per_chunk_usd: `0.05`
- estimated_chunks_within_per_chunk_cap: `True`
- ready_to_resume_17c_r2_live: `False`
- requires_new_cost_authorization: `True`

## Finding
At the 8192-token ceiling the adaptive planner keeps each chunk within the $0.05 per-chunk cap (selected chunk size `17`), but the worst-case total for all 478 requests is `$1.228029`, which is above the $0.40 total cap. The total request count is unchanged by design.

## Consequence
A revised total-cost authorization (or a separately authorized smaller live batch) is needed before a live run at this ceiling. The entry gate is not green until the worst-case total is within an authorized cap.
