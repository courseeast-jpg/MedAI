# CKA-B01 MKB Foundation Validation Report

**Block:** CKA-B01
**Conclusion:** `cka_b01_mkb_foundation_ready`

## Record Counts
- Inserted: 11
- Active: 4
- Hypothesis: 4
- Quarantined: 2
- Superseded: 1
- Ledger events: 12

## Safety Flags
- external_api_used: False
- raw_phi_logged_in_public_reports: False
- private_filename_path_leaks: 0
- production_ocr_changed: False
- production_extractor_changed: False
- safety_gate_changed: False
- frozen_hitl_release_reopened: False

## Encryption
- sqlcipher_encryption_active: False
- encryption_boundary_ready: True

## Feature Flags
```json
{
  "ENABLE_GRAPH": false,
  "ENABLE_LOCAL_LLM": false,
  "ENRICH_PROMOTE": false,
  "ACTIVE_CONNECTORS": [
    "dxgpt_stub"
  ],
  "ENABLE_WEB_INGESTION": false,
  "ENABLE_EPUB": false,
  "ENABLE_YOUTUBE": false,
  "SAFE_MODE_THRESHOLD": 0.4,
  "MEDAI_LOCAL_ONLY": true,
  "EXTERNAL_APIS_ENABLED": false
}
```