"""CKA-B08 Multi-Connector Execution package.

Provides:
- ConnectorSpec, ConnectorExecutionRequest, ConnectorExecutionResult models
- ConnectorRegistry
- Privacy-gated request builder
- Local connector stubs (dxgpt_stub, sage_epilepsy_stub, patientnotes_ddi_stub)
- Connector executor (deterministic, local-only)
- Response normalizer

Rules:
- No real external API calls.
- No real DxGPT/SAGE/PatientNotes/LLM integration.
- synthetic_only=True by default.
- external_api_used is always False.
- Privacy gate (CKA-B02) applied before any stub execution.
"""
