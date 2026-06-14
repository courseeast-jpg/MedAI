# MEDAI Private Artifact Visibility Policy 17C-R2-R4

## Operator Context Is The Source Of Truth

A committed repo report is NOT proof that a private artifact exists on disk. Presence
must be checked in the live operator runtime (the same shell that will run the live
batch) via the path resolver, immediately before use.

## Re-Materialization

If the canonical 478 batch or its sidecars are missing or invalid in the operator
context, they are deterministically rebuilt from the 17A tokenized corpus and the
17B-R2-R1 repair logic (ensure_ascii=True, physical-newline framing), re-sealed with a
SHA256 sidecar + doc-id manifest, and set read-only. The rebuild is deterministic, so a
re-materialized file has the same content and SHA256 as the original sealed batch.

## External Mutation

An external writer has been deleting/mutating `MedAI_Private` files. Read-only sealing
reduces accidental deletion but is not a guarantee. The live runner must verify the
canonical batch (existence + SHA256 + counts) immediately before any provider call; a
missing/invalid batch is a hard stop.
