# Phase 13 Performance Summary

- Generated at: 2026-04-24T20:31:21.776677+00:00
- Documents processed: 6/10
- Written: 6
- Queued for review: 0
- External quota blocked: 4
- Hard failures: 0
- Average confidence: 0.7
- Total pipeline time (ms): 19136.072
- Average document time (ms): 1913.607

## Route Distribution

- Actual routes: {'spacy': 6}
- Requested routes: {'gemini': 1, 'spacy': 5, 'unknown': 4}

## Extractor Timing

- `spacy` -> documents=6 total_ms=16928.126 avg_ms=2821.354

## Retry Visibility

- Retry events observed: 4
- Total suggested backoff seconds: 208.51
- `long_noisy_04.pdf` -> status=external_quota_blocked retry_delay_seconds=52.127445676 reason=external_quota
- `long_noisy_06.pdf` -> status=external_quota_blocked retry_delay_seconds=52.127445676 reason=external_quota
- `long_noisy_08.pdf` -> status=external_quota_blocked retry_delay_seconds=52.127445676 reason=external_quota
- `long_noisy_10.pdf` -> status=external_quota_blocked retry_delay_seconds=52.127445676 reason=external_quota
