# CKA-TERM-01F Manual Return Guide

TERM-02 has not started. Real terminology import remains blocked until the operator manually supplies licensed files and a private acknowledgment.

## Manual Files To Obtain

- UMLS: licensed local files containing MRCONSO.RRF and MRSTY.RRF.
- SNOMED CT: licensed local release files containing sct2_Concept and sct2_Description files.
- RxNorm: licensed local files containing RXNCONSO.RRF.
- LOINC: licensed local files containing Loinc.csv or LoincTable.csv.

## Where To Place Files

Place files under the local gitignored `terminology_data/` folder using these subfolders: `umls/`, `snomed_ct/`, `rxnorm/`, and `loinc/`.

## Private License Acknowledgment

Use `terminology_data/LICENSE_ACK_PRIVATE.json` locally. This file must stay private and must not be committed. Use only the safe fields `operator_acknowledged` and `acknowledged_systems`. Keep vendor terms out of reports and public files.

Example structure:

```json
{
  "operator_acknowledged": true,
  "acknowledged_systems": ["umls", "snomed_ct", "rxnorm", "loinc"]
}
```

## Before TERM-02

Run `python scripts/cka_term02_preflight_gate.py`. TERM-02 cannot start until the preflight gate passes.

No external APIs or downloads are used by the readiness checks. No clinical advice is generated.
