# Phase 13 Performance Summary

- Generated at: 2026-04-24T21:49:57.994762+00:00
- Documents processed: 47/50
- Written: 46
- Queued for review: 1
- External quota blocked: 3
- Hard failures: 0
- Average confidence: 0.7
- Total pipeline time (ms): 16666.994
- Average document time (ms): 333.34

## Route Distribution

- Actual routes: {'phi3': 1, 'spacy': 46}
- Requested routes: {'gemini': 3, 'spacy': 44, 'unknown': 3}

## Extractor Timing

- `phi3` -> documents=1 total_ms=2647.892 avg_ms=2647.892
- `spacy` -> documents=46 total_ms=12890.287 avg_ms=280.224

## Retry Visibility

- Retry events observed: 3
- Total suggested backoff seconds: 18.488
- `long_noisy_06.pdf` -> status=external_quota_blocked retry_delay_seconds=6.162811648 reason=external_quota
- `long_noisy_08.pdf` -> status=external_quota_blocked retry_delay_seconds=6.162811648 reason=external_quota
- `long_noisy_10.pdf` -> status=external_quota_blocked retry_delay_seconds=6.162811648 reason=external_quota
