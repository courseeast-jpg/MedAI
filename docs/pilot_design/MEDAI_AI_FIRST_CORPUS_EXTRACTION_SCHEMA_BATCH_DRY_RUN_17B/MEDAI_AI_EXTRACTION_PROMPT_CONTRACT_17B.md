# MEDAI AI Extraction Prompt Contract 17B

This document summarizes the prompt contract defined in
`config/medai_ai_extraction_prompt_contract_17b.md`. 17B is dry-run only and does not
call any AI provider.

## The AI Is Instructed To

- Extract only source-visible facts.
- Not infer unsupported diagnoses.
- Preserve token placeholders exactly.
- Not decode tokens.
- Return strict JSON only, matching the 17B extraction schema.
- Use null when a value is missing.
- Not provide medical advice.
- Mark uncertainty with `needs_review=true`.
- Attach tokenized evidence to every extracted fact.

## Safety

The prompt never asks the model to re-identify tokens, never requests a medical
decision, and never authorizes any write. Output is review-bound only.
