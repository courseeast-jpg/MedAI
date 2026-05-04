# Phase 58 Stratified Problem-Class Fix Plan

- Generated at: `2026-05-03T23:58:48.483434+00:00`
- Source Phase 57A report: `reports\phase57_full_corpus_inventory_audit\phase57_full_corpus_inventory_audit_report.json`
- Source Phase 57A clusters: `reports\phase57_full_corpus_inventory_audit\phase57_full_corpus_problem_clusters.json`

## Corpus Totals (Phase 57A)

- total_filesystem_files: `615`
- total_filesystem_folders: `109`
- total_supported_processed: `603`
- total_unsupported_extension: `11`
- total_ignored_system_files: `1`
- total_processing_errors: `0`
- total_inaccessible_files: `0`
- accepted: `88`
- review: `515`
- review_ocr_quality: `15`
- empty: `438`
- errors: `11`
- reconciliation_passed: `True`

## Explicit Decision

- **fix_first_class:** `empty_extraction`
- fix_first_action_kind: `forensic_subset_phase`
- fix_first_file_count: `438`
- fix_first_priority_score: `40.3693`

**Why fix this first:** empty_extraction ranks highest under the deterministic score (priority=40.3693). It is the most actionable class given the safety/opportunity/difficulty profile encoded in CLASS_PROFILES. Other large classes (empty_extraction, rules_based_low_confidence, possible_lab_table_failure) are deferred because acting on them corpus-wide would risk weakening existing safety gates without subset-level evidence.

**Do not fix yet (deferred classes with non-zero volume):** `['possible_lab_table_failure', 'review_ocr_quality', 'rules_based_low_confidence']`

## Prioritized Fix Queue

### phase59_candidate: `empty_extraction`

- recommended_action_kind: `forensic_subset_phase`
- file_count: `438`
- priority_score: `40.3693`
- safe_id_sample (up to 5): `['corpus_file_000004', 'corpus_file_000011', 'corpus_file_000013', 'corpus_file_000015', 'corpus_file_000024']`

_Largest class but root cause varies (lab table loss, vocabulary gap, sparse rules). Acting before forensic stratification would risk weakening safety gates. Run a Phase 42-style forensics on a small random subset first._

### phase60_candidate: `unsupported_extension`

- recommended_action_kind: `narrow_format_support`
- file_count: `11`
- priority_score: `1.8242`
- safe_id_sample (up to 5): `[]`

_Adding deterministic .docx and .rtf text extraction is well-scoped, well-understood, and produces no PHI exposure beyond what already applies to .txt. .mp3/.ogg/.msg are out of scope (audio/email)._

### phase61_candidate: `pdf_ocr_low_quality`

- recommended_action_kind: `small_followup`
- file_count: `7`
- priority_score: `0.6167`
- safe_id_sample (up to 5): `['corpus_file_000098', 'corpus_file_000211', 'corpus_file_000531', 'corpus_file_000535', 'corpus_file_000537']`

_Tiny class. Likely scanned PDFs where OCR fallback is already active. A targeted diagnostic on these files can confirm whether the existing PDF OCR pipeline is doing the right thing._

## Per-Class Breakdown

| Class | Count | % of supported | Safety | Opportunity | Difficulty | Action | Score |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| `unsupported_extension` | 11 | 1.82% | low | low | low | narrow_format_support | 1.8242 |
| `empty_extraction` | 438 | 72.64% | medium | high_if_narrowly_scoped | high | forensic_subset_phase | 40.3693 |
| `review_ocr_quality` | 15 | 2.49% | high | low | medium | defer | 0.1866 |
| `rules_based_low_confidence` | 511 | 84.74% | high | medium | high | defer | 5.8668 |
| `image_ocr_low_quality` | 5 | 0.83% | medium | low | medium | small_followup | 0.4405 |
| `pdf_ocr_low_quality` | 7 | 1.16% | medium | low | medium | small_followup | 0.6167 |
| `possible_multi_document_pdf` | 0 | 0.00% | high | medium | high | diagnostic_only_phase | 0.0 |
| `pdf_portfolio_or_embedded_files_detected` | 0 | 0.00% | high | low | high | diagnostic_only_phase | 0.0 |
| `possible_lab_table_failure` | 578 | 95.85% | high | medium | high | defer | 6.6361 |
| `possible_ecg_class` | 0 | 0.00% | medium | low | medium | defer | 0.0 |
| `possible_prescription_class` | 0 | 0.00% | low | low | low | defer | 0.0 |
| `possible_microbiology_pcr_class` | 0 | 0.00% | low | low | low | defer | 0.0 |
| `possible_russian_cyrillic_class` | 0 | 0.00% | medium | medium | medium | investigate_language_hint_signal | 0.0 |
| `unknown_other` | 0 | 0.00% | medium | low | low | defer | 0.0 |

## Class Rationales

### `unsupported_extension`

- file_count: `11`
- percent_of_supported_corpus: `0.0182`
- safety_risk: `low`
- automation_opportunity: `low`
- implementation_difficulty: `low`
- recommended_action_kind: `narrow_format_support`
- priority_score: `1.8242`
- safe_id_sample: `[]`

Adding deterministic .docx and .rtf text extraction is well-scoped, well-understood, and produces no PHI exposure beyond what already applies to .txt. .mp3/.ogg/.msg are out of scope (audio/email).

### `empty_extraction`

- file_count: `438`
- percent_of_supported_corpus: `0.7264`
- safety_risk: `medium`
- automation_opportunity: `high_if_narrowly_scoped`
- implementation_difficulty: `high`
- recommended_action_kind: `forensic_subset_phase`
- priority_score: `40.3693`
- safe_id_sample: `['corpus_file_000004', 'corpus_file_000011', 'corpus_file_000013', 'corpus_file_000015', 'corpus_file_000024']`

Largest class but root cause varies (lab table loss, vocabulary gap, sparse rules). Acting before forensic stratification would risk weakening safety gates. Run a Phase 42-style forensics on a small random subset first.

### `review_ocr_quality`

- file_count: `15`
- percent_of_supported_corpus: `0.0249`
- safety_risk: `high`
- automation_opportunity: `low`
- implementation_difficulty: `medium`
- recommended_action_kind: `defer`
- priority_score: `0.1866`
- safe_id_sample: `[]`

review_ocr_quality is by design a safety gate against bad OCR. Phase 45 already reconciled the false-positive prescription case. Touching the gate further requires class-specific OCR evidence, not blanket reclassification.

### `rules_based_low_confidence`

- file_count: `511`
- percent_of_supported_corpus: `0.8474`
- safety_risk: `high`
- automation_opportunity: `medium`
- implementation_difficulty: `high`
- recommended_action_kind: `defer`
- priority_score: `5.8668`
- safe_id_sample: `['corpus_file_000004', 'corpus_file_000011', 'corpus_file_000013', 'corpus_file_000015', 'corpus_file_000024']`

Lowering confidence thresholds or expanding rules globally would weaken the accept-vs-review boundary. Any change must be subset-driven, not corpus-driven.

### `image_ocr_low_quality`

- file_count: `5`
- percent_of_supported_corpus: `0.0083`
- safety_risk: `medium`
- automation_opportunity: `low`
- implementation_difficulty: `medium`
- recommended_action_kind: `small_followup`
- priority_score: `0.4405`
- safe_id_sample: `['corpus_file_000145', 'corpus_file_000204', 'corpus_file_000205', 'corpus_file_000206', 'corpus_file_000573']`

Tiny class. Phase 56 image OCR support already exists. A short diagnostic on the 5 affected files can confirm whether DPI / preprocessing is the lever, but ROI is low until volume grows.

### `pdf_ocr_low_quality`

- file_count: `7`
- percent_of_supported_corpus: `0.0116`
- safety_risk: `medium`
- automation_opportunity: `low`
- implementation_difficulty: `medium`
- recommended_action_kind: `small_followup`
- priority_score: `0.6167`
- safe_id_sample: `['corpus_file_000098', 'corpus_file_000211', 'corpus_file_000531', 'corpus_file_000535', 'corpus_file_000537']`

Tiny class. Likely scanned PDFs where OCR fallback is already active. A targeted diagnostic on these files can confirm whether the existing PDF OCR pipeline is doing the right thing.

### `possible_multi_document_pdf`

- file_count: `0`
- percent_of_supported_corpus: `0.0`
- safety_risk: `high`
- automation_opportunity: `medium`
- implementation_difficulty: `high`
- recommended_action_kind: `diagnostic_only_phase`
- priority_score: `0.0`
- safe_id_sample: `[]`

Splitting multi-document PDFs is not safe to do automatically without operator review. Start with a detection-only diagnostic phase that reports candidate split boundaries via safe_file_id.

### `pdf_portfolio_or_embedded_files_detected`

- file_count: `0`
- percent_of_supported_corpus: `0.0`
- safety_risk: `high`
- automation_opportunity: `low`
- implementation_difficulty: `high`
- recommended_action_kind: `diagnostic_only_phase`
- priority_score: `0.0`
- safe_id_sample: `[]`

PDF portfolios contain embedded files that may be PHI by separate consent. Do not extract embedded content automatically. Surface the count and let the operator decide.

### `possible_lab_table_failure`

- file_count: `578`
- percent_of_supported_corpus: `0.9585`
- safety_risk: `high`
- automation_opportunity: `medium`
- implementation_difficulty: `high`
- recommended_action_kind: `defer`
- priority_score: `6.6361`
- safe_id_sample: `['corpus_file_000001', 'corpus_file_000002', 'corpus_file_000003', 'corpus_file_000004', 'corpus_file_000005']`

Phase 40 / 41 already extended lab-row parsing twice. The current cluster is too broad (any reason code containing 'lab' or 'table'). Further expansion requires a stratified per-format subset, not a corpus-wide rewrite.

### `possible_ecg_class`

- file_count: `0`
- percent_of_supported_corpus: `0.0`
- safety_risk: `medium`
- automation_opportunity: `low`
- implementation_difficulty: `medium`
- recommended_action_kind: `defer`
- priority_score: `0.0`
- safe_id_sample: `[]`

Zero detected on this corpus. Build only when a class subset exists to validate against.

### `possible_prescription_class`

- file_count: `0`
- percent_of_supported_corpus: `0.0`
- safety_risk: `low`
- automation_opportunity: `low`
- implementation_difficulty: `low`
- recommended_action_kind: `defer`
- priority_score: `0.0`
- safe_id_sample: `[]`

Phase 43/45 already routes prescriptions correctly when detected. Zero detected on this corpus's ENGLISH classifier surface. The Cyrillic prescription path was validated on the holdout set.

### `possible_microbiology_pcr_class`

- file_count: `0`
- percent_of_supported_corpus: `0.0`
- safety_risk: `low`
- automation_opportunity: `low`
- implementation_difficulty: `low`
- recommended_action_kind: `defer`
- priority_score: `0.0`
- safe_id_sample: `[]`

Phase 43 already routes microbiology/PCR. Zero detected on this corpus's English surface.

### `possible_russian_cyrillic_class`

- file_count: `0`
- percent_of_supported_corpus: `0.0`
- safety_risk: `medium`
- automation_opportunity: `medium`
- implementation_difficulty: `medium`
- recommended_action_kind: `investigate_language_hint_signal`
- priority_score: `0.0`
- safe_id_sample: `[]`

Zero Cyrillic class flags but language_hint=unknown for the entire corpus. This is an upstream signal problem (language hint not propagating), NOT evidence of zero Cyrillic content. Worth investigating but not the highest-ROI starting point.

### `unknown_other`

- file_count: `0`
- percent_of_supported_corpus: `0.0`
- safety_risk: `medium`
- automation_opportunity: `low`
- implementation_difficulty: `low`
- recommended_action_kind: `defer`
- priority_score: `0.0`
- safe_id_sample: `[]`

Already zero on this corpus thanks to Phase 57A reconciliation. Re-evaluate only if a future inventory surfaces members.

## Privacy Safety

- uses_safe_ids_only: `True`
- raw_filenames_present_in_output: `False`
- raw_paths_present_in_output: `False`
- extracted_text_present_in_output: `False`
- phi_present_in_output: `False`

