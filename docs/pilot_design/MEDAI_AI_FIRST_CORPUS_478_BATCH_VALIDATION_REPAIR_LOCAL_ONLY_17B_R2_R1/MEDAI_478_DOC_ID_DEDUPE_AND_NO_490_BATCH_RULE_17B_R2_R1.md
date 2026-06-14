# MEDAI 478 Doc-ID Dedupe And No-490-Batch Rule 17B-R2-R1

## Rule

- 478 ready files are the total ready set.
- The old 12-file batch (from 17B-R1) must NOT be added on top of the 478 unless a
  documented doc_id comparison proves the 12 old doc_ids are absent from the 478.
- No 490-file batch is created.

## Dedupe Check

A doc_id comparison is performed between the old 12 (17B-R1) and the 478 ready set. If
the 12 old doc_ids are all present in the 478, the combined batch count remains 478 and
`old_12_added_separately=false`. If the comparison cannot be performed (artifacts
missing), the check is reported as unavailable and no 490 batch is created.

In this run, the 12 old doc_ids were confirmed present in the 478 ready set, so the batch
remains exactly 478.
