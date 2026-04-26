# Phase 13 Performance Summary

- Generated at: 2026-04-26T03:33:33.101317+00:00
- Documents processed: 46/50
- Written: 45
- Queued for review: 1
- External quota blocked: 4
- Hard failures: 0
- Average confidence: 0.692
- Total pipeline time (ms): 9878.29
- Average document time (ms): 197.566

## Route Distribution

- Actual routes: {'phi3': 1, 'spacy': 45}
- Requested routes: {'gemini': 1, 'phi3': 1, 'spacy': 44, 'unknown': 4}

## Extractor Timing

- `phi3` -> documents=1 total_ms=96.741 avg_ms=96.741
- `spacy` -> documents=45 total_ms=8241.183 avg_ms=183.137

## Retry Visibility

- Retry events observed: 4
- Total suggested backoff seconds: 125.396
- `long_noisy_04.pdf` -> status=external_quota_blocked retry_delay_seconds=31.348890647 reason=external_quota
- `long_noisy_06.pdf` -> status=external_quota_blocked retry_delay_seconds=31.348890647 reason=external_quota
- `long_noisy_08.pdf` -> status=external_quota_blocked retry_delay_seconds=31.348890647 reason=external_quota
- `long_noisy_10.pdf` -> status=external_quota_blocked retry_delay_seconds=31.348890647 reason=external_quota
