# MEDAI Batch Dry-Run And Cost Control 17B

17B is dry-run only and does not call any AI provider. No billing API call is made.

## Batch Composition

- Exactly the 12 ready tokenized files from 17A are included.
- The 587 blocked files are excluded.
- One outbound request is prepared per ready document.

## Outbound Requests (Private Only)

- Outbound request JSONL is written privately, outside the repo, and is never
  committed.
- Each request carries tokenized content and the prompt contract only — no raw OCR, no
  token map, no raw PI.

## Cost Control (Local Estimate Only)

- Token counts are estimated with a rough local approximation when no tokenizer is
  installed.
- Cost is estimated using configurable model assumptions, with no billing API call.
- Default target model for the future live batch: `gemini-2.5-flash-lite`.
- Reported figures (input tokens, output tokens, dollar cost) are approximate only and
  must be re-confirmed before any live run.

## Privacy Validation

Each outbound request is validated for residual raw PI patterns (email, phone, MRN,
insurance/account/accession/specimen, local paths). If any request fails, the batch is
marked blocked, no provider call occurs, and the 17C live batch does not start.
