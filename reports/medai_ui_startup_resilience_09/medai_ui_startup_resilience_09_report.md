# MEDAI-UI-STARTUP-RESILIENCE-09 — Report

Conclusion: **ui_startup_resilience_09_ready**

## Cases

- `all_ok`: status `app_startup_ok` ok=True sql_ok=True execution_ok=True vector_ok=True
- `vector_failed`: status `app_startup_ok_with_degraded_vector` ok=True sql_ok=True execution_ok=True vector_ok=False
- `pipeline_failed`: status `app_startup_ok_with_degraded_pipeline` ok=True sql_ok=True execution_ok=False vector_ok=True
- `both_optional_failed`: status `app_startup_ok_sqlite_only` ok=True sql_ok=True execution_ok=False vector_ok=False
- `sqlite_failed`: status `app_startup_failed` ok=False sql_ok=False execution_ok=False vector_ok=False

## Operator repair command

```
python scripts/run_medai_local_self_healing_validation_07.py
```

## Safety

- external API used: False
- auto-accept enabled: False
- privacy check passed: True
