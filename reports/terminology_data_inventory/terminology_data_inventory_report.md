# MedAI Terminology Data Inventory

This is a report-only inventory. It does not import terminology data, modify terminology files, read private license acknowledgment contents, or print licensed terminology rows.

## Summary

- Terminology root: `terminology_data`
- Root exists: `True`
- Folders: `839`
- Files: `4211`
- Total size: `17.8 GB`
- External API used: `False`
- Import performed: `False`
- License acknowledgment contents read: `False`

## Product Readiness Table

| System | Package | Status | Candidate count | Recommendation |
| --- | --- | --- | --- | --- |
| loinc | LOINC main package | present | 2 | Use the folder containing Loinc.csv as the canonical LOINC source. |
| rxnorm | RxNorm full monthly release | present | 1 | Use the full monthly release as canonical; keep prescribable subset auxiliary. |
| rxnorm_prescribable_subset | RxNorm prescribable subset | auxiliary_present | 1 | Treat as auxiliary only; avoid replacing the full RxNorm release with this subset. |
| umls | UMLS full distribution or generated RRF subset | present | 1 | Keep separate from RxNorm/LOINC imports unless a future licensed UMLS import is explicitly approved. |
| snomed_ct_us | SNOMED CT US Edition | present | 1 | Prefer US Edition as primary SNOMED source for US clinical workflows. |
| snomed_ct_international | SNOMED CT International | secondary_present | 1 | Treat International Edition as secondary unless a future import plan selects it explicitly. |
| license_ack_private | Private license acknowledgment | present | 1 | Presence only was checked; contents were not read. |

## Key File Detection

| System | Detected key files | Mode |
| --- | --- | --- |
| loinc | 5 | metadata only |
| rxnorm | 14 | metadata only |
| umls | 4 | metadata only |
| snomed_ct | 18 | metadata only |
| license_ack_private | 1 | presence only |

## Canonical Folder Recommendations

- LOINC: use the folder containing Loinc.csv as canonical.
- RxNorm: use the full monthly release as canonical; keep prescribable subset auxiliary.
- UMLS: keep separate until a licensed UMLS-specific import block is approved.
- SNOMED CT: prefer US Edition as primary; keep International Edition secondary unless explicitly selected.

## Duplicate / Parallel Folder Warnings

- loinc has 2 candidate folders; select one canonical folder before import changes.

## .gitignore Protection

| Pattern | Present directly | Protected | Note |
| --- | --- | --- | --- |
| terminology_data/ | True | True | direct |
| LICENSE_ACK_PRIVATE.json | True | True | direct |
| *.RRF | True | True | direct |
| *.rrf | True | True | direct |
| *.nlm | True | True | direct |
| *.zip | True | True | direct |
| data/terminology/ | True | True | direct |
| *.sqlite | True | True | direct |
| *.sqlite3 | True | True | direct |
| *.db | True | True | direct |

## Missing .gitignore Patterns

- No unprotected required patterns detected.

## Direct .gitignore Pattern Recommendations

- All requested patterns are present directly.

## Next Recommended MedAI Codebase Update Plan

- No import behavior changes should be made from this inventory alone.
- Confirm canonical folders with the operator before any import or adapter expansion.
- If SNOMED or UMLS support is approved later, create a separate gated parser/import block with synthetic tests first.
- Keep reports public-safe and keep licensed files and runtime indexes untracked.
