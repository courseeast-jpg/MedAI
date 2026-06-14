# MedAI 17C-R2 Sectioned Extraction Contract

Return strict compact JSON only. Do not include markdown, prose, source OCR text, private
paths, credentials, or long copied passages. Preserve token placeholders exactly.
Do not infer clinical meaning, synthesize missing values, or weaken required keys.

Each section call returns:

```json
{"section":"section_name","items":[],"needs_review":true,"warnings":[]}
```

Use empty arrays when absent. Every result remains review-bound. MKB writes,
auto-accept, and medical decisions are not authorized.
