# MedAI v2 OCR/Layout HITL Release

- Release name: MedAI v2 OCR/Layout HITL Release
- Validated source commit: `31024c7f18b65144addf0141876b040fbf92eaaf`
- Snapshot ID: `MedAI_Snapshot_Phase48_2026-05-01`
- Test result: `411 passed, 5 warnings`
- Validation result: Phase47 `release_candidate_ready`
- Final holdout metrics: total `8`, accepted `2`, review `6`, review_ocr_quality `2` as a review subset, empty `2` as a review subset

## What This Release Does

This release adds a practical v2 OCR/Layout layer for human-in-the-loop document intake. It profiles OCR/input quality, classifies document layout risk, routes poor inputs away from auto-acceptance, normalizes recoverable lab-table rows for review, handles Cyrillic/Russian OCR candidates, and locks validation against a frozen benchmark so runtime drift is visible instead of hidden.

## What This Release Does Not Do

This release does not provide autonomous clinical diagnosis, does not replace clinician review, does not auto-accept low-confidence extraction, does not weaken confidence gates, does not introduce broad terminology systems such as SNOMED/UMLS, and does not add new premium model dependencies or production orchestration infrastructure.

## Supported Document Behavior

- Clean digital PDFs: may be accepted only through existing confidence and safety gates.
- Lab reports: lab-table structure can be recovered for review when deterministic row parsing is strong enough; recovery cannot produce accepted status.
- Cyrillic/Russian medical documents: OCR/Layout can generate language-aware OCR candidates and route non-lab Cyrillic documents to review instead of mislabeling them as OCR-quality failures.
- Prescription/non-lab Cyrillic documents: can move from `review_ocr_quality` to `review` when OCR quality is recovered and document type is safely identified; never accepted by reconciliation alone.
- Bad OCR / weak scans: remain review-bound as `review_ocr_quality` or review, depending on diagnostics; poor OCR cannot auto-accept.

## Safety Guarantees

- Poor OCR cannot become accepted.
- Empty extraction cannot become accepted.
- Lab normalization cannot produce accepted status.
- Cyrillic non-lab reconciliation cannot produce accepted status.
- Phase37 confidence and safety gates remain enforced.
- Runtime drift is reported separately and cannot be counted as a phase improvement.
- No copied PDFs or PHI artifacts are tracked under report archive/review folders.

## Known Limitations

- The release remains HITL and validation-ready, not production-autonomous.
- OCR quality depends on local OCR capabilities and source scan quality.
- Some live extractor outputs can drift near thresholds; Phase46 locks and labels this drift.
- `review_ocr_quality` and `empty` are review subsets, not separate top-level totals.
- Operators must inspect all review queues before using extracted facts downstream.
