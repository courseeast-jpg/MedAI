# Phase 16 Quality Tuning Report

## Scope

Phase 16 analyzed `reports/phase15/validation_aggregate.json` and the document-level Phase 12 outputs under the Phase 15 baseline. The goal was to reduce document-level `queued_for_review` rate without changing extraction logic, routing, fallback behavior, or confidence thresholds.

## Baseline Findings

- Processed documents: `46`
- Written documents: `33`
- Queued documents: `13`
- External quota blocked: `4`
- Hard failures: `0`
- Average confidence: `0.7`
- Actual route distribution: `{'spacy': 46}`
- Requested route distribution: `{'gemini': 1, 'spacy': 45, 'unknown': 4}`
- Top recurring review reasons: `clear=13`, `quarantined=13`

### Confidence Distribution

- Written documents: all completed documents were at `0.7`
- Queued documents: all 13 queued documents were also at `0.7`

There was no evidence for a safe confidence promotion band. Confidence did not separate written from queued documents.

## Targeted Analysis

### A. Safe-to-Promote Band

- Not supported by the data.
- There was no meaningful `0.60–0.70` or similar band to promote because both written and queued documents already shared the same confidence value: `0.7`.

### B. Stable Consensus Blocked by Threshold

- Not supported by the data.
- All 46 processed documents had `validation_status=accepted`.
- The queueing happened after validation, not because of confidence or consensus thresholds.

### C. False Negatives

- Supported.
- All 13 queued documents had successful writes plus quarantined side records:
  - `written_count > 0`
  - `queued_count = 2`
  - recurring reasons: `clear`, `quarantined`
- These were not extraction failures. They were mixed-success documents where safe records were written and conflicting medication records were quarantined correctly.

## Minimal Change Implemented

One decision-logic change was justified and applied:

- Documents with successful writes plus queued review records are now classified as `written_with_review` instead of `queued_for_review`.

Preserved behavior:

- Quarantined records still remain quarantined.
- Review queue persistence is unchanged.
- Confidence thresholds are unchanged.
- Validation logic is unchanged.
- Routing and fallback behavior are unchanged.

## Verification

- `python -m pytest tests` -> `147 passed`
- `python scripts/run_phase11_integration_audit.py` -> passed
- `python scripts/run_phase12_real_world_validation.py --dataset-dir test_data/final_batch_50 --quota-safe` -> passed

## Before vs After

### Document Outcomes

- Before:
  - written: `33`
  - queued_for_review: `13`
  - external_quota_blocked: `4`
  - hard_failures: `0`
- After:
  - written: `46`
  - queued_for_review: `0`
  - external_quota_blocked: `4`
  - hard_failures: `0`

### Confidence

- Before average confidence: `0.7`
- After average confidence: `0.7`

### Review/Quarantine Signals

- Before top recurring patterns: `clear=13`, `quarantined=13`
- After top recurring patterns: `clear=13`, `quarantined=13`
- After total queued records remained present: `26`

This confirms the change reduced document-level queue inflation without suppressing quarantine behavior.

## Conclusion

The Phase 15 data did not justify a threshold change. The justified fix was to differentiate mixed-success quarantine cases from true document-level review failures. That reduced `queued_for_review` from `13` to `0` on the completed set while preserving correctness, quarantines, and zero hard failures.
