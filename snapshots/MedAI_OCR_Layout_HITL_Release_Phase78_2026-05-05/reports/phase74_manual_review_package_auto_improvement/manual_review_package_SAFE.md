# Manual Review Package (SAFE)

Aggregate buckets only — no PHI, no raw filenames, no raw paths.

## 1. OCR Quality Review

**bucket_id:** `ocr_quality_review`  
**aggregate_count:** 12  
**high_priority_item_count:** 3  
**operator_action_required:** True  
**production_change_allowed:** False  

**Why in review:** OCR quality gate triggered or borderline OCR detected. These files may have degraded text recognition that could lead to missed lab values or mis-classified document class.

**What the system knows:** OCR quality metrics flagged (SSIM/MSE/line-count drop). Safety gate retained — files are not accepted without operator review.

**What the system does not know:** Whether the degradation affects clinically meaningful content. Operator would need to open the original file to confirm.

**Safest next action:** Defer until operator reviews. Do not lower quality threshold. Consider improving the review UI to surface OCR confidence details.

**Pending safe IDs (sample):** `file_001`, `file_011`, `file_014`

---

## 2. Empty Extraction Review

**bucket_id:** `empty_extraction_review`  
**aggregate_count:** 382  
**high_priority_item_count:** 0  
**operator_action_required:** False  
**production_change_allowed:** False  

**Why in review:** Extraction produced zero structured results. Root cause varies: vocabulary gap, lab table layout failure, sparse rules, or the file genuinely contains no extractable medical data.

**What the system knows:** File text was obtained (or OCR succeeded). Parser returned empty output. Forensic diagnostics (Phase59/60) identified top root causes at aggregate level.

**What the system does not know:** Per-file root cause. Whether the document genuinely has no lab data, or if a parser rule gap is responsible.

**Safest next action:** Improve empty-extraction summary explanation in review package. Show per-item reason codes in the operator UI. Do not change parser rules or confidence thresholds without validated evidence.

---

## 3. Unknown Document Class Review

**bucket_id:** `unknown_document_class_review`  
**aggregate_count:** 509  
**high_priority_item_count:** 0  
**operator_action_required:** False  
**production_change_allowed:** False  

**Why in review:** Document class could not be reliably identified by the rule-based classifier. Low-confidence documents are routed to review rather than accepted or rejected.

**What the system knows:** Rule-based classification confidence below threshold. No specific alternative class was detected with sufficient confidence.

**What the system does not know:** True document class. Whether the file contains extractable medical data at all. Requires operator judgment to classify.

**Safest next action:** Improve class-confidence explanations in the review package. Aggregate by likely super-class (lab/ECG/prescription/other) where possible. Do not change classification thresholds without validated operator labels.

**Pending safe IDs (sample):** `file_002`, `file_003`, `file_004`, `file_005`, `file_006`, `file_007`, `file_008`, `file_009`, `file_010`, `file_012`

---

## 4. Possible Multi-Document PDF Review

**bucket_id:** `possible_multi_document_pdf_review`  
**aggregate_count:** 578  
**high_priority_item_count:** 0  
**operator_action_required:** False  
**production_change_allowed:** False  

**Why in review:** Document may be a multi-page bundle or contain a complex lab table layout that the extractor cannot reliably split or parse.

**What the system knows:** Phase61/62 geometry diagnostics found column-count signals consistent with bundled reports. Max block depth too shallow for confident header inference.

**What the system does not know:** Whether pages represent distinct encounters or a single continuous report. Splitting logic would require validated examples.

**Safest next action:** Surface geometry signal in the review package so operator can confirm whether splitting is appropriate. Do not implement splitting logic without validated examples.

---

## 5. Unsupported or Deferred Format Review

**bucket_id:** `unsupported_or_deferred_format_review`  
**aggregate_count:** 8  
**high_priority_item_count:** 0  
**operator_action_required:** False  
**production_change_allowed:** False  

**Why in review:** File extension is not supported by the current extraction pipeline (.docx, .msg, .mp3, etc.) or was deferred in Phase63/64/65.

**What the system knows:** Phase64/65 completed RTF support without safety regression. Phase63 triaged remaining unsupported formats. DOCX remains deferred — no evidence shows it outranks higher-priority work.

**What the system does not know:** Whether the unsupported files contain clinically important data. DOCX parsing has not been validated on this corpus.

**Safest next action:** List unsupported files in the review package by extension. No production format support change without a scoped forensics phase.

---

## 6. Completed Manual Boundary Branches

**bucket_id:** `completed_manual_boundary_branches`  
**aggregate_count:** 0  
**high_priority_item_count:** 0  
**operator_action_required:** False  
**production_change_allowed:** False  

**Why in review:** Diagnostic branches that were fully investigated and closed at the manual-review boundary. No production change was warranted.

**What the system knows:** Phase62: geometry header prototype — signal found but insufficient for production. Phase64/65: RTF text parser completed, no safety regression. Phase67: PDF OCR preprocessing comparison — retained manual-review boundary. Phase69: image OCR preprocessing comparison — retained manual-review boundary.

**What the system does not know:** Nothing material — these branches are closed.

**Safest next action:** No action required. Keep closed. Surface in review package as evidence that multiple diagnostic paths have been safely exhausted.

---
