# Phase 49 Privacy Gate Validation

- Generated at: `2026-05-02T00:15:42.801235+00:00`
- Local-only default: `True`
- External API default allowed: `False`
- PII scrub required: `True`
- Cloud payload contains raw PII: `False`
- Raw PHI logged in reports: `False`
- PHI/report artifacts tracked: `False`
- Conclusion: `privacy_gate_ready`

## Detection

- english_synthetic_pii_detected: `True`
- russian_cyrillic_synthetic_pii_detected: `True`
- mrn_insurance_like_ids_detected: `True`
- english_counts_by_type: `{'PERSON': 1, 'DOB': 1, 'MRN': 1, 'PHONE': 1, 'EMAIL': 1}`
- russian_counts_by_type: `{'RU_PERSON': 1, 'RU_DOB': 1, 'RU_INSURANCE_ID': 1, 'RU_PHONE': 1, 'RU_ADDRESS': 1}`
- id_counts_by_type: `{'INSURANCE_ID': 1, 'SSN': 1}`

## Redaction

- redaction_passed: `True`
- redaction_counts: `{'ID': 1, 'INSURANCE_ID': 2, 'ADDRESS': 1, 'PHONE': 2, 'DOB': 2, 'RU_PERSON': 1, 'EMAIL': 1, 'MRN': 1, 'PERSON': 1}`
- redacted_payload_has_raw_samples: `False`

## Outbound Gate

- raw_pii_blocked: `True`
- redacted_payload_allowed_when_external_enabled: `True`
- local_only_blocks_external: `True`
- redaction_failure_blocks_external: `True`

## UI Safety Panel

- local_only_label: `ON`
- external_apis_label: `DISABLED`
- pii_scrub_required_label: `YES`
- privacy_warning: `Local-only mode active. External APIs blocked.`
