# Phase 13 Performance Summary

- Generated at: 2026-04-24T20:53:15.968515+00:00
- Documents processed: 46/50
- Written: 33
- Queued for review: 13
- External quota blocked: 4
- Hard failures: 0
- Average confidence: 0.7
- Total pipeline time (ms): 17481.131
- Average document time (ms): 349.623

## Route Distribution

- Actual routes: {'spacy': 46}
- Requested routes: {'gemini': 1, 'spacy': 45, 'unknown': 4}

## Extractor Timing

- `spacy` -> documents=46 total_ms=15567.388 avg_ms=338.421

## Retry Visibility

- Retry events observed: 4
- Total suggested backoff seconds: 224.619
- `long_noisy_04.pdf` -> status=external_quota_blocked retry_delay_seconds=56.154655569 reason=external_quota
- `long_noisy_06.pdf` -> status=external_quota_blocked retry_delay_seconds=56.154655569 reason=external_quota
- `long_noisy_08.pdf` -> status=external_quota_blocked retry_delay_seconds=56.154655569 reason=external_quota
- `long_noisy_10.pdf` -> status=external_quota_blocked retry_delay_seconds=56.154655569 reason=external_quota
