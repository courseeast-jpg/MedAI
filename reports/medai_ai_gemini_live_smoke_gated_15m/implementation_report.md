# MEDAI-AI-GEMINI-LIVE-SMOKE-GATED-15M

- Overall status: `FAIL_LIVE_SMOKE`
- Privacy result: `passed`
- Selected provider: `gemini` | Effective: `fake_local`
- Live call status: `failed`
- Live call attempted: `True`
- Missing live gates: `[]`
- External API used: `True`
- Real network call used: `True`
- Gemini real call attempted: `True`
- Claude/OpenAI/Ollama real call attempted: `False`
- Local model call used: `False`
- Provider-execution subprocess path: `False`
- Active written count: `0`
- Auto-accept: `False`
- Review-bound package count: `0`
- Doctrine phrases present: `True`
- Flat suite status: `passed` counts `{'passed': 222, 'deselected': 8}` (79.657s)

## Live gate model

- A single live Gemini call occurs only if ALL gates pass: synthetic payload class,
  redacted text/layout summary, privacy/payload/budget/dry-run passed, operator staged,
  both env approval flags, GEMINI_API_KEY present, per-call budget cap, call_limit==1.
- Default (any gate missing): no call, no SDK import, no network, BLOCKED_READY_FOR_LIVE_SMOKE.
- Credential value is never read into reports/logs/UI (presence only).
- Output is always review-bound; no active MKB write; no auto-accept.

## Validation strategy

- Uses the 15L flat bounded harness over 15A-15M; recursive tests deselected.
- 12A and 13C run as direct commands; no recursive prior-block scripts invoked.

## Next recommended block

- MEDAI-AI-GEMINI-LIVE-SMOKE-OPERATOR-RUN-15N (operator executes the single approved live call).
