# Phase 13 Performance Summary

- Generated at: 2026-04-25T14:07:19.051260+00:00
- Documents processed: 46/50
- Written: 45
- Queued for review: 1
- External quota blocked: 4
- Hard failures: 0
- Average confidence: 0.692
- Total pipeline time (ms): 9355.043
- Average document time (ms): 187.101

## Route Distribution

- Actual routes: {'phi3': 1, 'spacy': 45}
- Requested routes: {'gemini': 1, 'phi3': 1, 'spacy': 44, 'unknown': 4}

## Extractor Timing

- `phi3` -> documents=1 total_ms=89.73 avg_ms=89.73
- `spacy` -> documents=45 total_ms=7763.237 avg_ms=172.516

## Retry Visibility

- Retry events observed: 4
- Total suggested backoff seconds: 184.756
- `long_noisy_04.pdf` -> status=external_quota_blocked retry_delay_seconds=46.189066049 reason=external_quota
- `long_noisy_06.pdf` -> status=external_quota_blocked retry_delay_seconds=46.189066049 reason=external_quota
- `long_noisy_08.pdf` -> status=external_quota_blocked retry_delay_seconds=46.189066049 reason=external_quota
- `long_noisy_10.pdf` -> status=external_quota_blocked retry_delay_seconds=46.189066049 reason=external_quota
