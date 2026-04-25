# Phase 13 Performance Summary

- Generated at: 2026-04-25T14:40:02.288659+00:00
- Documents processed: 46/50
- Written: 45
- Queued for review: 1
- External quota blocked: 4
- Hard failures: 0
- Average confidence: 0.692
- Total pipeline time (ms): 11948.069
- Average document time (ms): 238.961

## Route Distribution

- Actual routes: {'phi3': 1, 'spacy': 45}
- Requested routes: {'gemini': 1, 'phi3': 1, 'spacy': 44, 'unknown': 4}

## Extractor Timing

- `phi3` -> documents=1 total_ms=105.203 avg_ms=105.203
- `spacy` -> documents=45 total_ms=10296.381 avg_ms=228.808

## Retry Visibility

- Retry events observed: 4
- Total suggested backoff seconds: 17.83
- `long_noisy_04.pdf` -> status=external_quota_blocked retry_delay_seconds=4.457540095 reason=external_quota
- `long_noisy_06.pdf` -> status=external_quota_blocked retry_delay_seconds=4.457540095 reason=external_quota
- `long_noisy_08.pdf` -> status=external_quota_blocked retry_delay_seconds=4.457540095 reason=external_quota
- `long_noisy_10.pdf` -> status=external_quota_blocked retry_delay_seconds=4.457540095 reason=external_quota
