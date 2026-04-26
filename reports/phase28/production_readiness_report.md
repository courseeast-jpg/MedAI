# Phase 28 Production Readiness Report

- Generated at: `2026-04-26T03:33:35.310207+00:00`
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
- Review queue items: `31`
- Baseline reconciled: `False`
- Baseline source snapshot: `None`
- Reconciliation scope: `reporting_and_artifact_reconciliation_only`
- Reconciliation reason: `None`

## Gate Checks

- Previous run completed cleanly: `True`
- Deterministic outputs verified: `True`
- Unresolved runtime lock: `False`
- Snapshot verified: `True`
- Audit report available: `True`
- Required snapshot dir: `C:\Users\S1\Documents\Codex\phase27_continuation_20260424`
- Required snapshot zip: `C:\Users\S1\Documents\Codex\phase27_continuation_20260424.zip`

## Observed vs Canonical

- Observed validation result: `{'attempted': 50, 'processed': 46, 'written': 45, 'queued_for_review': 1, 'external_quota_blocked': 4, 'hard_failures': 0, 'avg_confidence': 0.692, 'review_queue_items': 31}`
- Canonical validation result: `{'attempted': 50, 'processed': 46, 'written': 45, 'queued_for_review': 1, 'external_quota_blocked': 4, 'hard_failures': 0, 'avg_confidence': 0.692, 'review_queue_items': 31}`
- Observed observability result: `{'route_mismatch_count': 1, 'low_confidence_count': 1, 'quota_safe_block_count': 4, 'extractor_route_counts': {'phi3': 1, 'spacy': 45}, 'extractor_actual_counts': {'phi3': 1, 'spacy': 45}}`
- Canonical observability result: `{'metrics_path': 'C:\\Users\\S1\\Documents\\Codex\\2026-04-22-connect-github\\artifacts\\phase21\\observability_metrics.json', 'report_path': 'C:\\Users\\S1\\Documents\\Codex\\2026-04-22-connect-github\\reports\\phase21\\observability_report.md', 'route_mismatch_count': 1, 'low_confidence_count': 1, 'quota_safe_block_count': 4, 'extractor_route_counts': {'phi3': 1, 'spacy': 45}, 'extractor_actual_counts': {'phi3': 1, 'spacy': 45}, 'per_stage_duration_ms': {'consensus': {'records': 46, 'events': 46, 'total_duration_ms': 0.0, 'avg_duration_ms': 0.0, 'max_duration_ms': 0.0}, 'extraction': {'records': 50, 'events': 96, 'total_duration_ms': 4101.314, 'avg_duration_ms': 82.026, 'max_duration_ms': 2701.746}, 'final_write': {'records': 39, 'events': 124, 'total_duration_ms': 32472.798, 'avg_duration_ms': 832.636, 'max_duration_ms': 4801.736}, 'language_support': {'records': 50, 'events': 50, 'total_duration_ms': 0.0, 'avg_duration_ms': 0.0, 'max_duration_ms': 0.0}, 'medical_coding': {'records': 46, 'events': 46, 'total_duration_ms': 0.0, 'avg_duration_ms': 0.0, 'max_duration_ms': 0.0}, 'safety_gate': {'records': 26, 'events': 98, 'total_duration_ms': 28860.973, 'avg_duration_ms': 1110.037, 'max_duration_ms': 4362.805}, 'semantic_enrichment': {'records': 46, 'events': 46, 'total_duration_ms': 0.0, 'avg_duration_ms': 0.0, 'max_duration_ms': 0.0}, 'truth_resolution': {'records': 39, 'events': 124, 'total_duration_ms': 32514.224, 'avg_duration_ms': 833.698, 'max_duration_ms': 4805.802}, 'validation': {'records': 46, 'events': 46, 'total_duration_ms': 0.0, 'avg_duration_ms': 0.0, 'max_duration_ms': 0.0}}}`
- Observed routing efficiency result: `{'route_mismatch_count': 1, 'intended_route_counts': {'gemini': 1, 'phi3': 1, 'spacy': 44}, 'actual_route_counts': {'phi3': 1, 'spacy': 45}, 'quota_block_avoided_count': 1, 'total_estimated_cost_units': 0.005, 'total_saved_cost_units': 0.02}`
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
