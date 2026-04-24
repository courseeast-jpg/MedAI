# Phase 13 Performance Summary

- Generated at: 2026-04-24T20:59:40.405025+00:00
- Documents processed: 46/50
- Written: 33
- Queued for review: 13
- External quota blocked: 4
- Hard failures: 0
- Average confidence: 0.7
- Total pipeline time (ms): 20182.606
- Average document time (ms): 403.652

## Route Distribution

- Actual routes: {'spacy': 46}
- Requested routes: {'gemini': 1, 'spacy': 45, 'unknown': 4}

## Extractor Timing

- `spacy` -> documents=46 total_ms=17758.365 avg_ms=386.051

## Retry Visibility

- Retry events observed: 4
- Total suggested backoff seconds: 137.181
- `long_noisy_04.pdf` -> status=external_quota_blocked retry_delay_seconds=34.295146429 reason=external_quota
- `long_noisy_06.pdf` -> status=external_quota_blocked retry_delay_seconds=34.295146429 reason=external_quota
- `long_noisy_08.pdf` -> status=external_quota_blocked retry_delay_seconds=34.295146429 reason=external_quota
- `long_noisy_10.pdf` -> status=external_quota_blocked retry_delay_seconds=34.295146429 reason=external_quota
