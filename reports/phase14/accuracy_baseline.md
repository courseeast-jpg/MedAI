# Phase 14 Accuracy Baseline

Baseline commit: `403f136b743bf95760ac9ef9104fb5ac8a1f0bb6`  
Baseline snapshot: `phase13_metrics_stabilized_20260424`

## Scope

This review is limited to the completed Phase 12 quota-safe validation documents only. It does not evaluate quota-blocked documents as accuracy failures, and it does not revisit routing, fallback consensus, thresholds, or quota-safe classification behavior.

## Completed Document Baseline

Current Phase 12 quota-safe summary:

- `documents_processed=6`
- `written=6`
- `queued_for_review=0`
- `external_quota_blocked=4`
- `hard_failures=0`
- `avg_confidence=0.7`

Completed documents:

| Document | Outcome | Validation | Requested Route | Terminal Extractor | Confidence |
| --- | --- | --- | --- | --- | --- |
| `long_noisy_01.pdf` | `written` | `accepted` | `spacy` | `spacy` | `0.7` |
| `long_noisy_02.pdf` | `written` | `accepted` | `gemini` | `spacy` | `0.7` |
| `long_noisy_03.pdf` | `written` | `accepted` | `spacy` | `spacy` | `0.7` |
| `long_noisy_05.pdf` | `written` | `accepted` | `spacy` | `spacy` | `0.7` |
| `long_noisy_07.pdf` | `written` | `accepted` | `spacy` | `spacy` | `0.7` |
| `long_noisy_09.pdf` | `written` | `accepted` | `spacy` | `spacy` | `0.7` |

Route behavior on completed documents:

- Five completed documents were both requested and completed on `spacy`.
- One completed document (`long_noisy_02.pdf`) was requested as `gemini` but completed on terminal `spacy`.
- All six completed documents were accepted without review reasons.
- There is no current evidence that accepted-document accuracy issues are caused by validation, thresholds, fallback consensus, or quota-safe handling.

## Evidence From Accepted Output

The strongest concrete defect in the completed-document baseline is a false positive accepted as a `test_result` on `long_noisy_01.pdf`.

Accepted active records for `long_noisy_01.pdf` include:

- `Diagnosis: seizure disorder`
- `Medication: Levetiracetam 1000mg`
- `Medication: Valproate 500mg`
- `Test: Sodium: 139 mmol/L`
- `Test: Page: 1`

`Test: Page: 1` is not a clinically meaningful entity. It is consistent with page-header/page-number noise being matched by the spaCy regex fallback as a test result. That is an extraction normalization problem, not a routing or validation problem.

The surrounding extracted content still shows valid signal, which makes the false positive suitable for a narrow fix rather than a broader extractor redesign.

## Candidate Improvements

### A. Prompt/schema issue

Assessment: not justified from the completed-document baseline.

Reasoning:

- All completed documents are terminal `spacy` writes.
- The concrete observed defect is produced by regex-style local extraction, not by a prompt or response schema.
- No prompt-side error pattern is needed to explain the accepted false positive.

### B. spaCy extraction normalization issue

Assessment: justified.

Evidence:

- `Test: Page: 1` is present in accepted output for `long_noisy_01.pdf`.
- The entity format matches the spaCy extractor's regex-based `test_result` path.
- This is a narrow normalization failure: non-clinical pagination text is being promoted as a medical test result.

Minimal targeted fix:

- Filter page-number-style noise such as `Page 1` from regex-derived `test_result` entities inside the spaCy extractor.
- Keep real numeric test results such as `Sodium 139 mmol/L`.
- Do not change thresholds, routing, fallback consensus, or validation behavior.

### C. validation/reporting issue

Assessment: not justified from the completed-document baseline.

Reasoning:

- The accepted documents are reported consistently as `written` and `accepted`.
- There is no evidence that the reporting layer created the false positive.
- The defect exists in the accepted extracted content itself.

### D. document quality issue

Assessment: weak evidence only.

Reasoning:

- The dataset is noisy by design, so pagination/header artifacts are plausible input noise.
- Even if the source contains page markers, treating `Page 1` as a clinical test result is still an extractor normalization problem.
- Document quality alone does not explain why the content was normalized into a medical fact.

### E. not enough evidence

Items intentionally not addressed in this phase:

- Missing medication frequency details such as `twice daily` or `nightly`
- Missing procedural/clinical findings such as EEG summary text
- Broader recall improvements for noisy long-form notes

These may be real quality gaps, but the current completed-document evidence is not yet strong enough to justify a broader accuracy change under the Phase 14 constraints.

## Decision

One minimal targeted fix is justified under category **B. spaCy extraction normalization issue**:

- suppress page-number-style false positives from regex-derived spaCy `test_result` entities

This is the smallest evidence-based accuracy change available from the current completed-document baseline.
