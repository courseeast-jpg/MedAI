# MEDAI-AI-VALIDATION-HARNESS-DERECURSION-15L

- Privacy result: `passed`
- Flat suite status: `passed`
- Flat suite counts: `{'passed': 185, 'deselected': 8}`
- Flat suite duration (s): `49.781`
- Total harness duration (s): `57.828`
- Deselected recursive tests: `8`
- 12A: `passed` | 13C: `passed`
- Nested prior-block scripts invoked: `False`
- Nested scripts refused by default: `True`
- Depth guard passed: `True`
- Timeout guard passed: `True`
- Subprocess use: `local_test_runner_only` (no provider-execution subprocess path)
- External API used: `False`
- Doctrine phrases present: `True`

## 15K recursion root cause

- Script-to-script: PRIOR_BASELINE_COMMANDS ran prior-block scripts, which ran theirs.
- Test-to-test: test_*regressions_still_pass subprocess-ran prior focused files, which ran theirs.
- Combined -> exponential nested execution and the ~4h hang.

## Validation strategy for future blocks

- Run focused test files flat in one pytest process with recursive tests deselected.
- Run 12A and 13C as direct commands once; never via nested prior-block scripts.
- Refuse nested prior-block scripts by default; MEDAI_VALIDATION_DEPTH guards depth>0.
- Enforce explicit per-command and total timeouts.

## Important distinction

- The harness uses subprocess ONLY to run local pytest/python test commands.
- It introduces NO provider-execution subprocess path: no SDK, no network, no localhost/Ollama.

## Next recommended block

- MEDAI-AI-LIVE-PROVIDER-SMOKE-TEST-GATED-15M (single operator-approved, audited live smoke test).
