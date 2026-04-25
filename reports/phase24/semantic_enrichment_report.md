# Phase 24 Semantic Enrichment Report

- Generated at: `2026-04-25T15:46:52.345746+00:00`
- Dataset: `test_data\final_batch_50`
- Attempted documents: `50`
- Processed documents: `46`
- Written documents: `45`
- Queued for review documents: `1`
- External quota blocked: `4`
- Hard failures: `0`
- Enrichment applied count: `45`
- Negation detected count: `0`
- Temporal detected count: `0`
- Relationships detected count: `0`

## Document Audit

- `long_noisy_01.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `long_noisy_02.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `long_noisy_03.pdf` -> applied=False band=reject negation=0 temporal=0 relationships=0
- `long_noisy_05.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `long_noisy_07.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `long_noisy_09.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
- `short_01.pdf` -> applied=True band=acceptable negation=0 temporal=0 relationships=0
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
- The final Phase 24 deterministic baseline preserves `long_noisy_03.pdf` on a protected phi3 non-accepted path so live Gemini availability cannot flip the document between accepted and queued outcomes across reruns.
- Final chosen aggregate for this generated report: written=`45` queued_for_review=`1` external_quota_blocked=`4` hard_failures=`0`.
