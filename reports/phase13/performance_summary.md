# Phase 13 Performance Summary

- Generated at: 2026-04-24T20:36:27.223157+00:00
- Documents processed: 6/10
- Written: 6
- Queued for review: 0
- External quota blocked: 4
- Hard failures: 0
- Average confidence: 0.7
- Total pipeline time (ms): 14307.62
- Average document time (ms): 1430.762

## Route Distribution

- Actual routes: {'spacy': 6}
- Requested routes: {'gemini': 1, 'spacy': 5, 'unknown': 4}

## Extractor Timing

- `spacy` -> documents=6 total_ms=11982.455 avg_ms=1997.076

## Retry Visibility

- Retry events observed: 4
- Total suggested backoff seconds: 166.372
- `long_noisy_04.pdf` -> status=external_quota_blocked retry_delay_seconds=41.593118662 reason=external_quota
- `long_noisy_06.pdf` -> status=external_quota_blocked retry_delay_seconds=41.593118662 reason=external_quota
- `long_noisy_08.pdf` -> status=external_quota_blocked retry_delay_seconds=41.593118662 reason=external_quota
- `long_noisy_10.pdf` -> status=external_quota_blocked retry_delay_seconds=41.593118662 reason=external_quota
