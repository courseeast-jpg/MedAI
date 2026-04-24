# Phase 13 Performance Summary

- Generated at: 2026-04-24T21:37:10.559168+00:00
- Documents processed: 46/50
- Written: 46
- Queued for review: 0
- External quota blocked: 4
- Hard failures: 0
- Average confidence: 0.7
- Total pipeline time (ms): 10270.838
- Average document time (ms): 205.417

## Route Distribution

- Actual routes: {'spacy': 46}
- Requested routes: {'gemini': 1, 'spacy': 45, 'unknown': 4}

## Extractor Timing

- `spacy` -> documents=46 total_ms=8765.338 avg_ms=190.551

## Retry Visibility

- Retry events observed: 4
- Total suggested backoff seconds: 216.974
- `long_noisy_04.pdf` -> status=external_quota_blocked retry_delay_seconds=54.243581052 reason=external_quota
- `long_noisy_06.pdf` -> status=external_quota_blocked retry_delay_seconds=54.243581052 reason=external_quota
- `long_noisy_08.pdf` -> status=external_quota_blocked retry_delay_seconds=54.243581052 reason=external_quota
- `long_noisy_10.pdf` -> status=external_quota_blocked retry_delay_seconds=54.243581052 reason=external_quota
