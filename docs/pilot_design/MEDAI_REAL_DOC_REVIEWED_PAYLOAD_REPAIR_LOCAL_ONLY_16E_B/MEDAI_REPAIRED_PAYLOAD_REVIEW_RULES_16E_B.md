# MEDAI Repaired Payload Review Rules 16E-B

## Status

Local-only. No provider call. No live gate activation. 16D retry is not started.

## Repair Rules

1. Facility/lab identifiers must be tokenized. Raw `Labcorp` (and case variants) must
   be replaced with a facility token such as `[FACILITY_1]` and must not remain in the
   repaired payload.
2. Clinical/test terms and lab table labels are restored from a curated allowlist when
   confidently safe, to preserve payload utility. Examples include Urinalysis, Specific
   Gravity, pH, Urine Color, Appearance, WBC Esterase, Protein, Glucose, Ketones,
   Occult Blood, Bilirubin, Urobilinogen, Nitrite, Microscopic Examination, WBC, RBC,
   Epithelial Cells, Casts, Bacteria, Urine Culture, Result, No growth, Reference
   Interval, Current Result, Previous Result, Units, Abnormal.
3. Dates remain tokenized unless explicitly approved later.
4. Patient identifiers are never restored.
5. Token maps are never included in public reports and are never sent.

## Verification

- The repaired payload is scanned to confirm the raw facility/lab identifier no longer
  remains.
- Public reports carry counts, booleans, hashes, and safe class labels only.
- A residual review by a human/operator is still required: an allowlist restore and a
  single facility token do not constitute proof of full de-identification.

## Required Human Review

Human/operator review of the private repaired payload remains mandatory before any
future live send. This block produces private review material only; it sends nothing.
