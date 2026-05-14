# Terminology Sources Preflight

This report validates canonical local terminology source locations using metadata only. It does not import terminology data, create runtime indexes, read private license acknowledgment contents, or print licensed terminology rows.

- Manifest source: `config/terminology_sources.example.json`
- Conclusion: `terminology_sources_preflight_ready`
- Import performed: `False`

## Canonical Source Table

| Source | Role | Canonical path | Status | Ready | Missing |
| --- | --- | --- | --- | --- | --- |
| loinc | primary | terminology_data/Loinc_2.82 | local_private_required | True | none |
| rxnorm | primary | terminology_data/RxNorm_full_05042026 | local_private_required | True | none |
| rxnorm_prescribable | auxiliary | terminology_data/RxNorm_full_prescribe_05042026 | local_private_optional | True | none |
| snomed_ct_us | primary | terminology_data/SnomedCT_ManagedServiceUS_PRODUCTION_US1000124_20260301T120000Z | local_private_required | True | none |
| snomed_ct_international | secondary | terminology_data/SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z | local_private_optional | True | none |
| umls | separate_future_import | terminology_data/umls 2026AA-full | local_private_required_future_block | True | none |
| license_ack_private | private_license_confirmation | terminology_data/LICENSE_ACK_PRIVATE.json | local_private_required | True | none |

## Readiness Table

| Readiness flag | Value |
| --- | --- |
| ready_for_loinc_preflight | True |
| ready_for_rxnorm_preflight | True |
| ready_for_snomed_us_preflight | True |
| umls_present_but_future_gated | True |
| license_ack_present_presence_only | True |

## Duplicate Candidate Warnings

| Source | Candidate count | Candidate paths | Action |
| --- | --- | --- | --- |
| loinc | 2 | terminology_data/loinc<br>terminology_data/Loinc_2.82 | warning_only_select_canonical_before_import_changes |

## Privacy / Safety Assertions

| Assertion | Value |
| --- | --- |
| import_performed | False |
| runtime_db_or_index_created | False |
| license_ack_contents_read | False |
| licensed_rows_printed | False |
| absolute_paths_in_report | False |
| external_api_used | False |
| b07_behavior_changed | False |
| clinical_logic_changed | False |
| ocr_extractor_safety_gates_changed | False |

## Next Recommended Gated Block

Operator approval for a separate, parser-specific import or adapter validation block; do not infer import readiness beyond this preflight.
