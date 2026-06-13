# MEDAI-VERTEX-SEMANTIC-PACKAGE-COMPARISON-LIVE-15P-C

- Status: `PASS`
- Live call made: `True` (exactly one; no retries)
- Provider response received: `True`
- Posted body allowed-keys-only: `True` (keys: `['contents', 'generationConfig']`)
- Provider route: `vertex` | Model: `gemini-2.5-flash-lite`
- Schema validation pass: `True`
- Hallucinated field count: `0`
- Token counts: prompt=`374` output=`371` total=`745`
- Review required: `True` | Auto-accept: `False` | Active written count: `0`
- Privacy result: `passed` | Billing check pending: `True`

## 15P-B failure (sanitized, prior block)

- 15P-B failed HTTP 400 request_config_rejected (MedAI metadata keys posted); fixed here by posting only contents+generationConfig.

## Safety

- Exactly one live Vertex call against one synthetic redacted portal result-card fixture.
- Posted body contains only `contents` and `generationConfig` (no MedAI metadata).
- JSON-only, temperature=0, maxOutputTokens<=512; no diagnosis/treatment requested.
- No credentials/tokens/auth headers/ADC paths recorded; sanitized response only.
- No active MKB writes; output remains review-bound; no auto-accept.
