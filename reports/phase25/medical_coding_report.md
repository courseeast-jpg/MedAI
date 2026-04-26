# Phase 25 Medical Coding Report

- Generated at: `2026-04-26T03:13:49.855556+00:00`
- Dataset: `test_data\final_batch_50`
- Attempted documents: `50`
- Processed documents: `46`
- Written documents: `45`
- Queued for review documents: `1`
- External quota blocked: `4`
- Hard failures: `0`
- Coding attempted count: `82`
- Coding success count: `8`
- Coding unmapped count: `74`
- Coding ambiguous count: `0`
- Coding skipped count: `29`
- Coding status counts: `{'coded': 8, 'skipped': 29, 'unmapped': 74}`

## Document Audit

- `long_noisy_01.pdf` -> band=acceptable attempted=3 coded=0 unmapped=3 ambiguous=0 skipped=1
- `long_noisy_02.pdf` -> band=acceptable attempted=3 coded=0 unmapped=3 ambiguous=0 skipped=1
- `long_noisy_03.pdf` -> band=reject attempted=0 coded=0 unmapped=0 ambiguous=0 skipped=0
- `long_noisy_05.pdf` -> band=acceptable attempted=3 coded=0 unmapped=3 ambiguous=0 skipped=1
- `long_noisy_07.pdf` -> band=acceptable attempted=3 coded=0 unmapped=3 ambiguous=0 skipped=1
- `long_noisy_09.pdf` -> band=acceptable attempted=3 coded=0 unmapped=3 ambiguous=0 skipped=1
- `short_01.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_02.pdf` -> band=acceptable attempted=1 coded=0 unmapped=1 ambiguous=0 skipped=0
- `short_03.pdf` -> band=acceptable attempted=2 coded=1 unmapped=1 ambiguous=0 skipped=0
- `short_04.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_05.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_06.pdf` -> band=acceptable attempted=1 coded=0 unmapped=1 ambiguous=0 skipped=1
- `short_07.pdf` -> band=acceptable attempted=1 coded=0 unmapped=1 ambiguous=0 skipped=0
- `short_08.pdf` -> band=acceptable attempted=2 coded=1 unmapped=1 ambiguous=0 skipped=0
- `short_09.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_10.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_11.pdf` -> band=acceptable attempted=1 coded=0 unmapped=1 ambiguous=0 skipped=1
- `short_12.pdf` -> band=acceptable attempted=1 coded=0 unmapped=1 ambiguous=0 skipped=0
- `short_13.pdf` -> band=acceptable attempted=2 coded=1 unmapped=1 ambiguous=0 skipped=0
- `short_14.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_15.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_16.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_17.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=0
- `short_18.pdf` -> band=acceptable attempted=2 coded=1 unmapped=1 ambiguous=0 skipped=0
- `short_19.pdf` -> band=acceptable attempted=1 coded=0 unmapped=1 ambiguous=0 skipped=1
- `short_20.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_21.pdf` -> band=acceptable attempted=1 coded=0 unmapped=1 ambiguous=0 skipped=1
- `short_22.pdf` -> band=acceptable attempted=1 coded=0 unmapped=1 ambiguous=0 skipped=0
- `short_23.pdf` -> band=acceptable attempted=2 coded=1 unmapped=1 ambiguous=0 skipped=0
- `short_24.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_25.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_26.pdf` -> band=acceptable attempted=1 coded=0 unmapped=1 ambiguous=0 skipped=1
- `short_27.pdf` -> band=acceptable attempted=1 coded=0 unmapped=1 ambiguous=0 skipped=0
- `short_28.pdf` -> band=acceptable attempted=2 coded=1 unmapped=1 ambiguous=0 skipped=0
- `short_29.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_30.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_31.pdf` -> band=acceptable attempted=1 coded=0 unmapped=1 ambiguous=0 skipped=1
- `short_32.pdf` -> band=acceptable attempted=1 coded=0 unmapped=1 ambiguous=0 skipped=0
- `short_33.pdf` -> band=acceptable attempted=2 coded=1 unmapped=1 ambiguous=0 skipped=0
- `short_34.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_35.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_36.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_37.pdf` -> band=acceptable attempted=1 coded=0 unmapped=1 ambiguous=0 skipped=0
- `short_38.pdf` -> band=acceptable attempted=2 coded=1 unmapped=1 ambiguous=0 skipped=0
- `short_39.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1
- `short_40.pdf` -> band=acceptable attempted=2 coded=0 unmapped=2 ambiguous=0 skipped=1

## Non-Destructive Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- Medical coding is additive metadata only and does not alter confidence, routing, review, write, or semantic enrichment outputs.
- Rejected outputs are not coded.
- Seed mappings are local deterministic placeholders for future SNOMED/UMLS expansion and do not require external installation or licensing for this phase.
