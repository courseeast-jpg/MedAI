# Phase 22 Accuracy Calibration Report

- Generated at: `2026-04-25T03:41:34.252300+00:00`
- Dataset: `test_data\final_batch_50`
- Attempted documents: `50`
- Processed documents: `46`
- Written documents: `46`
- Queued for review documents: `0`
- External quota blocked: `4`
- Hard failures: `0`
- Average raw confidence: `0.7`
- Average calibrated confidence: `0.7`
- Route mismatch count: `1`

## Calibration Summary

- Confidence band counts: `{'acceptable': 46}`
- Calibration reason counts: `{'raw_confidence_retained': 45, 'raw_confidence_retained,requested_route_mismatch_observed,fallback_connector_used': 1}`
- Review recommendation counts: `{'accept': 45, 'accept_with_route_audit': 1}`
- Extractor route counts: `{'spacy': 46}`
- Extractor actual counts: `{'spacy': 46}`

## Document Audit

- `long_noisy_01.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `long_noisy_02.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept_with_route_audit route_mismatch=True
- `long_noisy_03.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `long_noisy_05.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `long_noisy_07.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `long_noisy_09.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_01.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_02.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_03.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_04.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_05.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_06.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_07.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_08.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_09.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_10.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_11.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_12.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_13.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_14.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_15.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_16.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_17.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_18.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_19.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_20.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_21.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_22.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_23.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_24.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_25.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_26.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_27.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_28.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_29.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_30.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_31.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_32.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_33.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_34.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_35.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_36.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_37.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_38.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_39.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False
- `short_40.pdf` -> band=acceptable raw=0.7 calibrated=0.7 recommendation=accept route_mismatch=False

## Stability Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- Review-band confidence remains observable through the review queue and audit fields.
- Reject-band confidence is not written as accepted output.
