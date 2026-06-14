# Corpus 2 / P2 — live AI extraction readiness

- pi_vault_loaded: `True` (private values: `8`)
- files_discovered_total: `25` (pdf: `10`)
- extraction succeeded / failed: `25` / `0` (ocr used: `7`)
- tokenized_request_count: `25` (blocked for review: `0`)
- residual PI patterns / vault-value leaks in payloads: `0` / `0`
- outbound package integrity ok: `True`
- ready_for_corpus2_p2_live_ai_extraction: `True`
- requires_human_fix_before_live: `False`

## Vault coverage caveat
The Person 2 PI vault has `8` filled values. Readiness uses the same gate as Corpus 1: zero high-confidence PII patterns (email/phone/MRN/account/accession/specimen/local-path/date) AND zero vault-value leaks in every tokenized payload. Free-text names or providers NOT present in the vault and NOT matching a high-confidence pattern are not tokenized by this gate. The operator should confirm the vault covers all person/provider identifiers before authorizing live extraction.

Live AI extraction is a separate, explicitly authorized step. This block performs no provider call and sets no live gate. The outbound package, token maps, raw extraction, and PI vault remain private and outside the repository.
