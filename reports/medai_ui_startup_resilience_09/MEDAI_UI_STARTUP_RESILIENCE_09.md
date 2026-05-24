# MEDAI-UI-STARTUP-RESILIENCE-09 — Short Summary

Conclusion: **ui_startup_resilience_09_ready**

- all_ok: `app_startup_ok` ok=True
- vector_failed: `app_startup_ok_with_degraded_vector` ok=True
- pipeline_failed: `app_startup_ok_with_degraded_pipeline` ok=True
- both_optional_failed: `app_startup_ok_sqlite_only` ok=True
- sqlite_failed: `app_startup_failed` ok=False

Operator repair command:

```
python scripts/run_medai_local_self_healing_validation_07.py
```
