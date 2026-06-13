# MEDAI Vertex Real Document Single Pilot Final Go/No-Go Matrix 16C-C

This block is no-live and does not set the live gate. 16D is not started.

| Final gate | Required state before any future 16D | NO-GO if not met |
| --- | --- | --- |
| Dedicated live gate `MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED` | unset/inactive now; set only in authorized 16D | yes |
| Explicit user authorization | obtained before 16D | yes |
| Explicit operator approval | recorded before 16D | yes |
| Cost cap | confirmed before 16D | yes |
| Redaction/tokenization proof | passed before 16D | yes |
| One-document limit | enforced | yes |
| One-call limit | enforced | yes |
| Stop-on-first-failure | enforced | yes |
| Rollback / failure plan | ready before 16D | yes |
| Evidence capture before and after | ready | yes |
| No token map in outbound payload or public reports | enforced | yes |
| No raw PII/PHI in outbound payload | enforced | yes |
| No corpus processing | enforced | yes |
| No active MKB write / no auto-accept / no medical decision | enforced unless separately authorized | yes |

Any unmet row is a mandatory NO-GO. The gate alone is never enough.
