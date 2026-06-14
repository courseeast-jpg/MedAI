# Corpus 2 / P2 live entry gate

- outbound package present: `True` (jsonl + sha256 + integrity + manifest)
- token maps present (private): `True`
- tokenized_request_count: `25` (expected `25`)
- high_confidence_uncovered_pi_count: `0`
- vault filled values: `8`; provider/facility coverage: `False`
- vault_manual_review_required: `True`
- estimated_total_cost_usd: `3.289151` (cap `$2.0`) -> within_cap=`False`
- selected_chunk_size: `0`; per-chunk worst case: `0.0` (cap `$0.05`) -> within_cap=`False`
- credential_preflight_passed: `True` (`pass`)
- live_entry_gate_passed: `False`
- block_reason: `vault_manual_review_required`

## Why live is withheld
High-confidence structured PII in the tokenized payloads is 0. However the P2 PI vault has only `8` filled values and NO provider/facility class entries, while the corpus shows provider/facility cues and person-like candidates. Those would be transmitted un-tokenized to the external provider. The live authorization is conditional on vault coverage passing; manual vault review/expansion (add provider/facility/secondary-person identifiers) is required before live extraction is authorized.

## To proceed later
Add the missing identifiers to the private vault CSV, re-run Corpus 2 P2 prep to re-tokenize and rebuild the outbound package, then re-run this local gate. When the gate passes, run `--live` once.
