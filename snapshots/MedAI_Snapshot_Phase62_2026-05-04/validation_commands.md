# Validation Commands

Run from repository root:

```powershell
python -m pytest tests
python scripts/run_phase47_final_regression_hardening.py
python scripts/run_phase48_release_snapshot_validation.py
python scripts/run_phase49_privacy_gate_validation.py
python scripts/run_phase57_full_corpus_inventory_audit.py
python scripts/run_phase58_stratified_problem_fix_plan.py
python scripts/run_phase59_empty_extraction_forensics.py
python scripts/run_phase60_text_extraction_gap_diagnostic.py
python scripts/run_phase61_header_label_inference_diagnostic.py
python scripts/run_phase62_table_geometry_header_inference_prototype.py
```

Expected safety posture:

- Local-only mode remains enabled.
- External APIs remain disabled by default.
- No production extractor or threshold change is introduced by Phase62.
- Public reports remain PHI-safe.

