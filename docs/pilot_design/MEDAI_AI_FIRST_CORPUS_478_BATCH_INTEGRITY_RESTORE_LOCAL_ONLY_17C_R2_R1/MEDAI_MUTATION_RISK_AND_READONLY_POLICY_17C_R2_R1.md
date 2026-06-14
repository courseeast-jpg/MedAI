# MEDAI Mutation Risk And Read-Only Policy 17C-R2-R1

## Observed Mutation Risk

The canonical private outbound JSONL was corrupted after the 17B-R2-R1 run that wrote a
clean 478 (it grew to 591 lines with 134 malformed entries). An external writer outside
this session appears to be mutating files under `MedAI_Private`, and has also been
observed advancing the branch HEAD, editing already-committed 17C / 17C-R1 reports, and
adding untracked scripts. This block does not attribute the writer to a specific process
and does not kill any process or delete unrelated files.

## Read-Only Policy

- After an atomic rebuild, the canonical JSONL is set read-only when the OS supports it.
- A future authorized rebuild clears read-only, replaces atomically, then re-applies it.
- A future live runner must verify the SHA256 against the sealed sidecar before any
  provider call; read-only alone is not a substitute for SHA256 verification.

## Reporting Only

This block reports whether the file was writable/read-only before and after, and whether
unexpected untracked scripts or dirty report edits are present. It does not stage,
modify, or remove those unrelated files.
