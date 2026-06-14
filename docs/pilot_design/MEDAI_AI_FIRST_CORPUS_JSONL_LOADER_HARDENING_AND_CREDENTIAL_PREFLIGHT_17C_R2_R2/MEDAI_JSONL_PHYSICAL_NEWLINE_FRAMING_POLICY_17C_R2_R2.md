# MEDAI JSONL Physical-Newline Framing Policy 17C-R2-R2

## Rule

JSONL records are framed by the physical newline `"\n"` only. `str.splitlines()` must
never be used for JSONL framing because it also splits on Unicode line separators
(U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR, U+0085 NEL, and the vertical-tab /
form-feed / file-separator family) that may legally appear inside JSON string values.

## Shared Reader

`execution/jsonl_framing.read_jsonl_lines(path)`:

- reads utf-8 text (replacement on decode errors)
- splits only on `"\n"`
- trims a trailing `"\r"` from each physical line
- skips only the final empty line if present

`execution/jsonl_framing.load_jsonl_objects(path)` returns parsed objects, a malformed
count, and the non-empty record count.

## Defense In Depth

Batches are also written with `ensure_ascii=True`, so every record is exactly one
physical line under both `"\n"`-split and `splitlines()`. Both the physical-newline
reader and the SHA256 seal must agree before any future live run.
