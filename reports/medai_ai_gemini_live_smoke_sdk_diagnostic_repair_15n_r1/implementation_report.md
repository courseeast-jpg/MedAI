# MEDAI-AI-GEMINI-LIVE-SMOKE-SDK-DIAGNOSTIC-REPAIR-15N-R1

- Privacy result: `passed`
- No live call made during 15N-R1: `True`
- External API used (repair run): `False`
- Real network call used (repair run): `False`
- Gemini real call attempted (repair run): `False`
- Provider error capture repaired: `True`
- Fake failure matrix all safe: `True`
- SDK call style detected: `legacy_generativeai`
- Default model: `gemini-2.5-flash-lite`
- Doctrine phrases present: `True`
- Flat suite: `passed` counts `{'passed': 261, 'deselected': 8}` (244.187s)

## What was repaired

- Replaced the broad exception handler that recorded only `gemini_live_call_failed`.
- Provider exceptions are now classified (SDK signature / invalid-argument / not-found /
  permission / quota / timeout / transport / unknown) and the message is sanitized
  (credential, API-key patterns, bearer tokens, paths, prompt/payload literals) and truncated.
- SDK-aware client boundary: prefers google-genai, falls back to legacy google-generativeai,
  detected without network; model name configurable via GEMINI_MODEL (default text model).

## Important distinction

- Local subprocess is used ONLY to run pytest/python test commands via the flat harness.
- NO provider-execution subprocess path; NO real Gemini call was made in 15N-R1.

## Failed 15M live-run preservation

- Sanitized count/flag-only summary is preserved in failed_live_run_diagnostic_summary.json.
- The pre-repair harness discarded the provider error; it is not reconstructed.

## Next recommended block

- MEDAI-AI-GEMINI-LIVE-SMOKE-OPERATOR-RERUN-15N-R2: operator performs exactly one new live
  call after this repair; the next failure (if any) will be captured + classified safely.
