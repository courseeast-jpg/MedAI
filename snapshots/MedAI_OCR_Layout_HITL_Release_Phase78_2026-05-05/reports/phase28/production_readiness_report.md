# Phase 28 Production Readiness Report

- Generated at: `2026-04-26T16:08:12.622894+00:00`
- Dataset: `test_data\final_batch_50`
- Attempted documents: `50`
- Processed documents: `46`
- Written documents: `45`
- Queued for review documents: `1`
- External quota blocked: `4`
- Hard failures: `0`
- Production mode: `OFF`
- Production gate passed: `True`
- Production gate failed reason: `None`
- Dry run executed: `False`
- Controlled run limit applied: `False`
- Run blocked by gate: `False`
- Max documents per run: `0`
- Max concurrent runs: `1`
- Audit required: `False`
- Require snapshot before run: `False`
- Run approval: `False`
- Review queue acknowledged: `False`
- Pre-run review queue items (gate input): `31`
- Current run review queue items: `31`
- Baseline reconciled: `True`
- Baseline source snapshot: `C:\Users\S1\Documents\Codex\phase27_continuation_20260424`
- Reconciliation scope: `reporting_and_artifact_reconciliation_only`
- Reconciliation reason: `observed_validation_drift`

## Gate Checks

- Previous run completed cleanly: `False`
- Deterministic outputs verified: `False`
- Unresolved runtime lock: `False`
- Snapshot verified: `False`
- Audit report available: `False`
- Required snapshot dir: `None`
- Required snapshot zip: `None`

## Observed vs Canonical

- Observed validation result: `{'attempted': 50, 'processed': 49, 'written': 45, 'queued_for_review': 4, 'external_quota_blocked': 1, 'hard_failures': 0, 'avg_confidence': 0.698, 'review_queue_items': 31}`
- Canonical validation result: `{'attempted': 50, 'processed': 46, 'written': 45, 'queued_for_review': 1, 'external_quota_blocked': 4, 'hard_failures': 0, 'avg_confidence': 0.692, 'review_queue_items': 31}`
- Observed observability result: `{'route_mismatch_count': 8, 'low_confidence_count': 4, 'quota_safe_block_count': 1, 'extractor_route_counts': {'phi3': 4, 'spacy': 45}, 'extractor_actual_counts': {'phi3': 4, 'spacy': 45}}`
- Canonical observability result: `{'metrics_path': 'C:\\Users\\S1\\Documents\\Codex\\2026-04-22-connect-github\\artifacts\\phase21\\observability_metrics.json', 'report_path': 'C:\\Users\\S1\\Documents\\Codex\\2026-04-22-connect-github\\reports\\phase21\\observability_report.md', 'route_mismatch_count': 1, 'low_confidence_count': 1, 'quota_safe_block_count': 4, 'extractor_route_counts': {'phi3': 1, 'spacy': 45}, 'extractor_actual_counts': {'phi3': 1, 'spacy': 45}, 'per_stage_duration_ms': {'consensus': {'records': 46, 'events': 46, 'total_duration_ms': 0.0, 'avg_duration_ms': 0.0, 'max_duration_ms': 0.0}, 'extraction': {'records': 50, 'events': 96, 'total_duration_ms': 4157.541, 'avg_duration_ms': 83.151, 'max_duration_ms': 2764.078}, 'final_write': {'records': 39, 'events': 124, 'total_duration_ms': 32929.676, 'avg_duration_ms': 844.351, 'max_duration_ms': 4873.473}, 'language_support': {'records': 50, 'events': 50, 'total_duration_ms': 0.0, 'avg_duration_ms': 0.0, 'max_duration_ms': 0.0}, 'medical_coding': {'records': 46, 'events': 46, 'total_duration_ms': 0.0, 'avg_duration_ms': 0.0, 'max_duration_ms': 0.0}, 'safety_gate': {'records': 26, 'events': 98, 'total_duration_ms': 29302.352, 'avg_duration_ms': 1127.014, 'max_duration_ms': 4453.65}, 'semantic_enrichment': {'records': 46, 'events': 46, 'total_duration_ms': 0.0, 'avg_duration_ms': 0.0, 'max_duration_ms': 0.0}, 'truth_resolution': {'records': 39, 'events': 124, 'total_duration_ms': 32963.77, 'avg_duration_ms': 845.225, 'max_duration_ms': 4884.746}, 'validation': {'records': 46, 'events': 46, 'total_duration_ms': 0.0, 'avg_duration_ms': 0.0, 'max_duration_ms': 0.0}}}`
- Observed routing efficiency result: `{'route_mismatch_count': 8, 'intended_route_counts': {'gemini': 8, 'spacy': 41}, 'actual_route_counts': {'phi3': 4, 'spacy': 45}, 'quota_block_avoided_count': 0, 'total_estimated_cost_units': 0.02, 'total_saved_cost_units': 0.14}`
- Canonical routing efficiency result: `{'metrics_path': 'C:\\Users\\S1\\Documents\\Codex\\2026-04-22-connect-github\\artifacts\\phase23\\routing_efficiency.json', 'report_path': 'C:\\Users\\S1\\Documents\\Codex\\2026-04-22-connect-github\\reports\\phase23\\routing_efficiency_report.md', 'intended_route_counts': {'gemini': 1, 'phi3': 1, 'spacy': 44}, 'actual_route_counts': {'phi3': 1, 'spacy': 45}, 'route_mismatch_count': 1, 'quota_block_avoided_count': 1, 'total_estimated_cost_units': 0.005, 'total_saved_cost_units': 0.02}`

## Control-Layer Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- Production mode is a gate-only layer and does not alter extraction, routing, confidence, review, enrichment, coding, or language behavior.
- `OFF` mode preserves the validated Phase 27 baseline behavior.
- When live external quota variance shifts canonical aggregates, `OFF` mode can restore the verified snapshot artifact set to preserve the trusted baseline outputs.
- The live observed run is emitted separately from reconciled canonical outputs so quota and routing drift remain visible instead of being silently overwritten.
- Reconciliation is limited to reporting and artifact restoration after pipeline execution completes; it does not alter the executed pipeline path.
- `DRY_RUN` reroutes run-local outputs away from canonical full-cycle outputs while still producing audit artifacts.
- `CONTROLLED` and `LIVE` require gate checks to pass before execution proceeds.
