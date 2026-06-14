# MEDAI 17C-R2 Required Schema Fields Policy R7

## Required Top-Level Keys

Every extraction response must include all required top-level keys. The validator
requires ALL core keys: `extracted_labs`, `extracted_diagnoses`,
`extracted_medications`, `needs_review`, `source_evidence`, `extraction_warnings`. The
prompt also requires the full schema skeleton of top-level keys.

## Allowed

- Instruct the model to emit every required key, using `[]` for list fields and `null`
  only where the schema permits null.
- Report exactly which required keys are missing (names/paths only).

## Not Allowed

- Do not infer required clinical fields.
- Do not synthesize missing values.
- Do not weaken the schema.
- Do not make required fields optional just to pass.
- Do not accept incomplete output as success.
- Do not write to MKB.

## Defaulting

Defaulting a missing OPTIONAL list field to `[]` is permitted only where the schema
explicitly allows it; required fields are never defaulted or synthesized. Because the
failed response body is unavailable in this context, no defaulting/recovery was applied
to the real response.
