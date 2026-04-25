# Phase 23 Routing Efficiency Report

- Generated at: `2026-04-25T13:18:33.660436+00:00`
- Dataset: `test_data\final_batch_50`
- Attempted documents: `50`
- Processed documents: `50`
- Written documents: `45`
- Queued for review documents: `5`
- External quota blocked: `0`
- Hard failures: `0`
- Route mismatch count: `10`
- Quota block avoided count: `0`
- Total estimated cost units: `0.025`
- Total saved cost units: `0.175`

## Route Summary

- Intended route counts: `{'gemini': 10, 'spacy': 40}`
- Actual route counts: `{'phi3': 5, 'spacy': 45}`
- Confidence band counts: `{'acceptable': 45, 'review': 5}`
- Review recommendation counts: `{'accept': 40, 'accept_with_route_audit': 5, 'operator_review': 5}`
- Fallback reason counts: `{'Gemini route fallback occurred despite configured key: rules_based; root_cause=429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash\nPlease retry in 29.586774204s. [links {\n  description: "Learn more about Gemini API quotas"\n  url: "https://ai.google.dev/gemini-api/docs/rate-limits"\n}\n, violations {\n  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"\n  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"\n  quota_dimensions {\n    key: "model"\n    value: "gemini-2.5-flash"\n  }\n  quota_dimensions {\n    key: "location"\n    value: "global"\n  }\n  quota_value: 20\n}\n, retry_delay {\n  seconds: 29\n}\n]': 1, 'latency_too_high:2571.000': 1, 'latency_too_high:2593.000': 1, 'latency_too_high:4035.000': 1, 'latency_too_high:4112.000': 1, 'latency_too_high:4721.000': 1, 'latency_too_high:4730.000': 1, 'latency_too_high:5131.000': 1, 'latency_too_high:5274.000': 1, 'latency_too_high:7349.000': 1}`

## Document Audit

- `long_noisy_01.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `long_noisy_02.pdf` -> intended=gemini actual=spacy saved_cost=0.02 quota_avoided=False band=acceptable recommendation=accept_with_route_audit
- `long_noisy_03.pdf` -> intended=gemini actual=phi3 saved_cost=0.015 quota_avoided=False band=review recommendation=operator_review
- `long_noisy_04.pdf` -> intended=gemini actual=spacy saved_cost=0.02 quota_avoided=False band=acceptable recommendation=accept_with_route_audit
- `long_noisy_05.pdf` -> intended=gemini actual=phi3 saved_cost=0.015 quota_avoided=False band=review recommendation=operator_review
- `long_noisy_06.pdf` -> intended=gemini actual=spacy saved_cost=0.02 quota_avoided=False band=acceptable recommendation=accept_with_route_audit
- `long_noisy_07.pdf` -> intended=gemini actual=phi3 saved_cost=0.015 quota_avoided=False band=review recommendation=operator_review
- `long_noisy_08.pdf` -> intended=gemini actual=spacy saved_cost=0.02 quota_avoided=False band=acceptable recommendation=accept_with_route_audit
- `long_noisy_09.pdf` -> intended=gemini actual=phi3 saved_cost=0.015 quota_avoided=False band=review recommendation=operator_review
- `long_noisy_10.pdf` -> intended=gemini actual=spacy saved_cost=0.02 quota_avoided=False band=acceptable recommendation=accept_with_route_audit
- `short_01.pdf` -> intended=gemini actual=phi3 saved_cost=0.015 quota_avoided=False band=review recommendation=operator_review
- `short_02.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_03.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_04.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_05.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_06.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_07.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_08.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_09.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_10.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_11.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_12.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_13.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_14.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_15.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_16.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_17.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_18.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_19.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_20.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_21.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_22.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_23.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_24.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_25.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_26.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_27.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_28.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_29.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_30.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_31.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_32.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_33.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_34.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_35.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_36.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_37.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_38.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_39.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept
- `short_40.pdf` -> intended=spacy actual=spacy saved_cost=0.0 quota_avoided=False band=acceptable recommendation=accept

## Stability Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- Review-band documents remain review-visible and are not silently accepted.
- Quota-safe blocks remain separate from hard failures.
