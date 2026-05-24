# MEDAI-LOCAL-SELF-HEALING-VALIDATION-07 — Report

Branch: `clinical-knowledge-architecture` | HEAD: `dev`

Final conclusion: **local_operator_validation_ready_adapter_path**

## Inputs

- dependency prepare result: `ok`
- input mode: `synthetic_fallback`
- input suffix: `.txt`
- input safe hash: `selfheal_0ca51e2abc`
- 05 one-doc conclusion: `not_ready`

## Path readiness

- pipeline path ready: False
- adapter path ready: True

## Counts

- structured facts extracted: 6
- review-bound MKB records persisted: 6
- SQLite retrieval proof: 6
- UI preview rows: 6
- operator action proof count: 3
- UI smoke passed: True
- streamlit launch attempted: False

## Safety

- external API used: False
- auto-accept enabled: False
- privacy check passed: True

## Next operator command

```
streamlit run app/main.py
```
