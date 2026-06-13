# MEDAI Vertex Real Document Single Pilot Pre-16D Privacy Matrix 16C-B

| Privacy requirement | Status (this block) | Required before any future live call |
| --- | --- | --- |
| Redaction/tokenization preflight | demonstrated on synthetic fixtures | yes |
| Outbound payload contains tokens only | demonstrated | yes |
| Token maps excluded from outbound payload | enforced | yes |
| Token maps excluded from public reports | enforced | yes |
| Explicit operator approval | required before 16D | yes |
| Cost cap | required before 16D | yes |
| One-document limit | required before 16D | yes |
| One-call limit | required before 16D | yes |
| Stop-on-first-failure | required before 16D | yes |
| Rollback / failure plan | required before 16D | yes |
| Dedicated future live gate `MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED` | not set / not active | only in 16D, separately authorized |
| GCP synthetic sandbox = environment/auth/request path only | acknowledged | yes |

This block is no-live, synthetic only, and does not set the live gate. 16D is not
started.
