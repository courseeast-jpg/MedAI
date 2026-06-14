# MEDAI 17C-R2 Strict JSON Response Policy R6

## Accepted

- Exactly one JSON object, optionally wrapped in a single clean Markdown fence
  (```json ... ``` or ``` ... ```), which the normalizer strips.

## Rejected (hard fail, no recovery)

- Empty response.
- Prose before or after the JSON.
- Multiple JSON objects or any trailing text after the closing brace.
- Truncated / partial / invalid JSON.
- Unterminated Markdown fence.
- A non-object JSON value.

## Non-negotiable

- The schema is never weakened. After normalization, the object must still contain the
  expected extraction fields and contain no residual raw PI; otherwise it fails.
- Partial JSON is never accepted. Missing fields are never inferred or repaired.
- The normalizer never extracts a JSON substring out of prose.
