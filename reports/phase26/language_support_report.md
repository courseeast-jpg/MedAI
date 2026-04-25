# Phase 26 Language Support Report

- Generated at: `2026-04-25T15:46:52.345746+00:00`
- Dataset: `test_data\final_batch_50`
- Attempted documents: `50`
- Processed documents: `46`
- Written documents: `45`
- Queued for review documents: `1`
- External quota blocked: `4`
- Hard failures: `0`
- Language detected counts: `{'english': 46}`
- Cyrillic detected count: `0`
- Mixed language count: `0`
- Pending translation count: `0`
- Requires OCR count: `0`
- Language unknown count: `0`

## Document Audit

- `long_noisy_01.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `long_noisy_02.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `long_noisy_03.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `long_noisy_05.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `long_noisy_07.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `long_noisy_09.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_01.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_02.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_03.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_04.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_05.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_06.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_07.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_08.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_09.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_10.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_11.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_12.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_13.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_14.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_15.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_16.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_17.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_18.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_19.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_20.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_21.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_22.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_23.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_24.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_25.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_26.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_27.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_28.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_29.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_30.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_31.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_32.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_33.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_34.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_35.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_36.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_37.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_38.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_39.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required
- `short_40.pdf` -> language=english script=latin cyrillic=False requires_ocr=False translation=not_required

## Non-Destructive Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- Language support is metadata only in this phase and does not alter confidence, routing, review, write, semantic enrichment, or medical coding outputs.
- OCR and translation are not executed in this phase; `requires_ocr` and `translation_status` are advisory metadata only.
