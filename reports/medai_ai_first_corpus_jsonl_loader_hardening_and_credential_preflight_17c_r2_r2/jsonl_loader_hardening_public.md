# JSONL loader hardening (public)

- jsonl_splitlines_usage_removed_from_live_loader: `True`
- jsonl_physical_newline_reader_used: `True`

A shared reader (`execution/jsonl_framing.py`) frames JSONL on the physical
newline only and never uses `splitlines()`. The 17C-R2 live batch loader and the
17B-R2-R1 JSONL framing loads now use it. Validation rules are unchanged.
