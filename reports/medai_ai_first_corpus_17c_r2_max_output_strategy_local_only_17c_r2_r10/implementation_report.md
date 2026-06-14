# MEDAI-AI-FIRST-CORPUS-17C-R2-MAX-OUTPUT-STRATEGY-LOCAL-ONLY-17C-R2-R10 — implementation report

## Strategy change
- Output ceiling raised `2048` -> `8192` tokens in the live runner (root cause of the R9 truncation).
- Compact-output rules added to `config/medai_ai_extraction_prompt_contract_17b.md` (schema not weakened; strict JSON + required skeleton preserved).
- Adaptive chunk planning added via `execution/cost_chunk_planner.py` and wired into the runner: the largest chunk size within the $0.05 per-chunk cap is selected automatically; total request count is unchanged.

## Cost recomputation (worst case at the new ceiling)
- estimated_total = `$1.228029` (cap `$0.4` -> within_cap=`False`).
- selected_chunk_size = `17` (from `25`); per-chunk worst case = `$0.04902` (cap `$0.05` -> within_cap=`True`).

## Entry gate
- ready_to_resume_17c_r2_live = `False`; requires_new_cost_authorization = `True`.
- Per-chunk cost fits the cap at the adaptive size, but the worst-case total for all requests at the raised ceiling is above the $0.40 total cap, so a revised cap (or a separately authorized smaller live batch) is required before a live run.

## Preserved guarantees
- R8 checkpoint/resume and failed-evidence preservation intact.
- R9 public-report redaction intact (no private paths / secrets in public reports).
