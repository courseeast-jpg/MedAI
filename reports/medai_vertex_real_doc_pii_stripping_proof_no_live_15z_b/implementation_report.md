# MEDAI-VERTEX-REAL-DOC-PII-STRIPPING-PROOF-NO-LIVE-15Z-B

- pii_stripping_proof_created: `True`
- pii_cases_total / passed: `10` / `10`
- pii_findings_total / pii_tokens_total: `31` / `30`
- token_classes_detected_count: `9` (ACCESSION, ADDRESS, DATE, EMAIL, FACILITY, MRN, PATIENT_NAME, PHONE, PROVIDER)
- deterministic_repeated_token_count: `1`
- outbound_payloads_created_count: `10`
- raw_pii_in_outbound_payload_count: `0` | raw_pii_in_report_count: `0`
- token_map_in_outbound_payload_count: `0` | token_map_in_report_count: `0`
- vault_records_created / isolated: `10` / `10`
- readiness_cases_fed_count: `10` | no_live_replay_allowed_count: `5` | blocked_case_count: `5`
- real_doc_live_allowed_count: `0`
- live_call_made: `False` | external_api_used: `False` | active_written_count: `0` | auto_accept_true_count: `0`
- privacy_result: `passed` | billing_check_pending: `True`

## Behavior proven

- All nine PII-like token classes are deterministically detected and tokenized; repeated identifiers map to the same token.
- Outbound-safe payloads contain `[CLASS_n]` tokens only; the token map stays in an isolated vault record (counts + fingerprint in reports, never the mapping).
- Unredacted residue, token-map-in-payload, and token-map-in-report each force BLOCKED.
- Unknown and real/private provenance stay BLOCKED even when redaction succeeds.
- Sanitized payloads feed the 15Z-A framework as no-live replay candidates only; `live_call_allowed=False` for every case.

## Safety

- No provider call, no live gate, no network, no real document.
- No active MKB write, no auto-accept; review_required=true for all cases.
- Reports carry tokens, counts, token classes, and fingerprints only — no raw PII-like values, no token map, no credentials/paths.
- Token classes covered: PATIENT_NAME, PROVIDER, FACILITY, MRN, ACCESSION, DATE, PHONE, EMAIL, ADDRESS.
