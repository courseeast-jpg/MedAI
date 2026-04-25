# Phase 13 Performance Summary

- Generated at: 2026-04-25T15:20:49.049544+00:00
- Documents processed: 46/50
- Written: 45
- Queued for review: 1
- External quota blocked: 4
- Hard failures: 0
- Average confidence: 0.692
- Total pipeline time (ms): 9347.592
- Average document time (ms): 186.952

## Route Distribution

- Actual routes: {'phi3': 1, 'spacy': 45}
- Requested routes: {'gemini': 1, 'phi3': 1, 'spacy': 44, 'unknown': 4}

## Extractor Timing

- `phi3` -> documents=1 total_ms=122.85 avg_ms=122.85
- `spacy` -> documents=45 total_ms=7661.835 avg_ms=170.263

## Retry Visibility

- Retry events observed: 4
- Total suggested backoff seconds: 64.252
- `long_noisy_04.pdf` -> status=external_quota_blocked retry_delay_seconds=16.06289861 reason=external_quota
- `long_noisy_06.pdf` -> status=external_quota_blocked retry_delay_seconds=16.06289861 reason=external_quota
- `long_noisy_08.pdf` -> status=external_quota_blocked retry_delay_seconds=16.06289861 reason=external_quota
- `long_noisy_10.pdf` -> status=external_quota_blocked retry_delay_seconds=16.06289861 reason=external_quota
