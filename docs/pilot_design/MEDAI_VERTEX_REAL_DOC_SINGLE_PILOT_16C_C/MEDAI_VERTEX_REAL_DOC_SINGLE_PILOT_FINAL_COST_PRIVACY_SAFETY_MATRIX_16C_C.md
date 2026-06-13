# MEDAI Vertex Real Document Single Pilot Final Cost/Privacy/Safety Matrix 16C-C

This block is no-live; no billing API call and no provider call occur. 16D is not
started.

| Dimension | Requirement before any future 16D |
| --- | --- |
| Cost | Cost cap confirmed (model, project, region, max calls, max tokens, dollar cap, stop-on-quota). |
| Privacy | Redaction/tokenization proof passed; no raw PII/PHI in outbound payload; no token map in outbound payload or public reports; no private corpus read; no private corpus traversal. |
| Safety | One-document limit; one-call limit; stop-on-first-failure; rollback; evidence capture; no active MKB write; no auto-accept; no medical decision; no production queue mutation. |
| Separation | GCP synthetic sandbox evidence proves only environment/auth/request path, not MedAI real-document readiness. |

The dedicated live gate `MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED` remains unset/inactive after 16C-C.
