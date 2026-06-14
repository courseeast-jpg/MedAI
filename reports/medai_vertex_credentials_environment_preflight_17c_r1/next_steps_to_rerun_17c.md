# Next steps to re-run 17C

Result: **LOGIN_REQUIRED**

Credentials require interactive login/reauth (no secret was printed; no model call
was made). A human must run a gcloud/ADC login out-of-band, for example:
   gcloud auth application-default login --project sot-knowledge-ocr
Then re-run this preflight. 17C is NOT re-run here.
