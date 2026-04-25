# Phase 24 Semantic Enrichment Report

- Generated at: `2026-04-25T13:18:33.660436+00:00`
- Dataset: `test_data\final_batch_50`
- Attempted documents: `50`
- Processed documents: `50`
- Written documents: `45`
- Queued for review documents: `5`
- External quota blocked: `0`
- Hard failures: `0`
- Enrichment applied count: `50`
- Negation detected count: `0`
- Temporal detected count: `0`
- Relationships detected count: `0`

## Document Audit

- `long_noisy_01.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `long_noisy_02.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `long_noisy_03.pdf` -> applied=True band=review negation=0 temporal=0 relationships=0
- `long_noisy_04.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `long_noisy_05.pdf` -> applied=True band=review negation=0 temporal=0 relationships=0
- `long_noisy_06.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `long_noisy_07.pdf` -> applied=True band=review negation=0 temporal=0 relationships=0
- `long_noisy_08.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `long_noisy_09.pdf` -> applied=True band=review negation=0 temporal=0 relationships=0
- `long_noisy_10.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_01.pdf` -> applied=True band=review negation=0 temporal=0 relationships=0
- `short_02.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_03.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_04.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_05.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_06.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_07.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_08.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_09.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_10.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_11.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_12.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_13.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_14.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_15.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_16.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_17.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_18.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_19.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_20.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_21.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_22.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_23.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_24.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_25.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_26.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_27.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_28.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_29.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_30.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_31.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_32.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_33.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_34.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_35.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_36.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_37.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_38.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_39.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_40.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0

## Non-Destructive Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- Semantic enrichment is additive metadata only and does not alter confidence, routing, or review decisions.
- Reject-band outputs are not semantically enriched.

## Safety Note

- `long_noisy_03.pdf` remains on the protected review path in the current Phase 24 rerun: `raw_confidence=0.68`, `calibrated_confidence=0.68`, `confidence_band=review`, `intended_route=gemini`, `actual_route=phi3`, `outcome=queued_for_review`.
- Semantic enrichment did run on `long_noisy_03.pdf`, but only after calibration and without feeding back into confidence, routing, review, or write decisions.
- The review-band threshold is unchanged. `execution/confidence_calibration.py` still classifies `>= 0.70` as `acceptable`, `>= 0.50 and < 0.70` as `review`, and `< 0.50` as `reject`.
- Any remaining drift in other `long_noisy_*` documents during live reruns is attributable to routing/connector behavior, not to the Phase 24 semantic enrichment layer.
