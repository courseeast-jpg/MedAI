# Phase 13 Performance Summary

- Generated at: 2026-04-25T02:31:12.027145+00:00
- Documents processed: 46/50
- Written: 45
- Queued for review: 1
- External quota blocked: 4
- Hard failures: 0
- Average confidence: 0.7
- Total pipeline time (ms): 14502.349
- Average document time (ms): 290.047

## Route Distribution

- Actual routes: {'phi3': 1, 'spacy': 45}
- Requested routes: {'gemini': 2, 'spacy': 44, 'unknown': 4}

## Extractor Timing

- `phi3` -> documents=1 total_ms=2636.151 avg_ms=2636.151
- `spacy` -> documents=45 total_ms=10321.781 avg_ms=229.373

## Retry Visibility

- Retry events observed: 4
- Total suggested backoff seconds: 209.274
- `long_noisy_04.pdf` -> status=external_quota_blocked retry_delay_seconds=52.318439258 reason=external_quota
- `long_noisy_06.pdf` -> status=external_quota_blocked retry_delay_seconds=52.318439258 reason=external_quota
- `long_noisy_08.pdf` -> status=external_quota_blocked retry_delay_seconds=52.318439258 reason=external_quota
- `long_noisy_10.pdf` -> status=external_quota_blocked retry_delay_seconds=52.318439258 reason=external_quota
