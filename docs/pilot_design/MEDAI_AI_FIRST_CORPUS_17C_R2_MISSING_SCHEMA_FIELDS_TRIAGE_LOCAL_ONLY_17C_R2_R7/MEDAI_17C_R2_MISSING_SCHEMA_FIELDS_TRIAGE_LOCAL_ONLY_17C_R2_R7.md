# MEDAI 17C-R2 Missing Schema Fields Triage Local-Only 17C-R2-R7

## Status

- Local-only triage. No Gemini/Vertex/Claude/OpenAI model call, no provider content
  request, no billing API call, no live gate, no live extraction, no new corpus request.
- No MKB open/write, no auto-accept, no medical decision, no production queue mutation.

## Context

After the R6 hardening the live route reaches Vertex/Gemini and returns JSON. Request #1
succeeded; request #2 failed because the returned JSON object did not contain the
required top-level schema keys (failure_stage=schema,
failure_category=missing_expected_schema_fields). The run stopped on first failure; no
MKB write.

## Evidence Availability

The private staged raw/parsed responses for #2 are not retained in this operator context
(the staging reflects an earlier blocked run; the Downloads preserved copy is absent; an
external writer keeps clearing MedAI_Private). The exact missing field names therefore
could not be enumerated from evidence; the failure subclass is reported as
`unknown_missing_expected_schema_fields`. No response body is printed or committed. The
failure category itself confirms the provider returned a JSON OBJECT (the field check
only runs after a successful JSON parse).

## Fixes (do not weaken schema; do not synthesize)

1. Prompt hardening: the contract now lists every required top-level key and instructs
   the model to include ALL of them even when empty (empty array for lists, null only
   where the schema allows), and to never omit a required key or invent values.
2. Required-fields precheck (`execution/strict_json.missing_required_keys`): reports the
   exact missing required keys (names only) for a parsed object.
3. Stricter validation: the 17C-R2 validator now requires ALL core top-level keys (was
   "any one key"); incomplete output is never accepted as success. Schema is not
   weakened; no field is made optional; nothing is inferred or synthesized.

## Replay

Bodies are unavailable, so the real #2 could not be replayed/recovered. A synthetic
precheck demonstration proves the precheck reports missing keys and the validator passes
only a full skeleton.
