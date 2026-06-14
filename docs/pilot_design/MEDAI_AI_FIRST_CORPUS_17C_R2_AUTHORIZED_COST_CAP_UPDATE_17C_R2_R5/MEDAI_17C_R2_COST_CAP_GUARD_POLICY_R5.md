# MEDAI 17C-R2 Cost Cap Guard Policy R5

## Caps

- Per-chunk hard cap: $0.05 (unchanged).
- Total hard cap: $0.40 (authorized; previously $0.25).

## Enforcement (unchanged behavior, new total value)

Before any provider call, the 17C-R2 live runner estimates total and per-chunk
input/output tokens locally and refuses to proceed if:

- the estimated total cost exceeds $0.40, or
- any chunk estimate exceeds $0.05.

No billing API is called; the caps are enforced from local token estimates only. The
caps are hard ceilings, not advisory. Stop-on-first-failure remains in force, and the
live gate is set only inside each authorized chunk and cleared after every chunk.
