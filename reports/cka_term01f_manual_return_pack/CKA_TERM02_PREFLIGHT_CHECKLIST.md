# CKA-TERM-02 Preflight Checklist

- `terminology_data/` exists locally.
- Supported subfolders exist: `umls/`, `snomed_ct/`, `rxnorm/`, `loinc/`.
- Expected files are present for at least one licensed system.
- `terminology_data/LICENSE_ACK_PRIVATE.json` exists locally.
- `operator_acknowledged` is true.
- `acknowledged_systems` covers every system with files present.
- Readiness checker reports at least one import-ready system.
- No `terminology_data/` files are staged.
- No `data/terminology/` files are staged.
- TERM-02 has not started before the gate passes.
