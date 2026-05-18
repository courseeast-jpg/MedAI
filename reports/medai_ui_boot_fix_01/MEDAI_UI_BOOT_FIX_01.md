# MEDAI-UI-BOOT-FIX-01 Startup Resilience

If MKB initialization fails, Streamlit should render a diagnostics-only startup panel instead of crashing.

Operator guidance:

- No clinical processing started.
- Avoid manual DB deletion.
- Run Git Safety Check or startup diagnostics.
- Use a Codex repair block if database quarantine or restore is needed.
