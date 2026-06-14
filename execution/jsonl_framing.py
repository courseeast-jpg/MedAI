"""Shared JSONL physical-newline framing reader.

JSONL records are framed by the physical newline byte ``"\\n"`` only. Unicode line
separators (U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR, U+0085 NEL, and the
vertical-tab/form-feed/file-separator family) can legally appear INSIDE JSON string
values, and ``str.splitlines()`` treats them as line breaks — which over-splits a
single JSONL record into several fragments and produces false "malformed/extra-line"
results. JSONL framing must therefore never use ``str.splitlines()``.

Use :func:`read_jsonl_lines` (or :func:`load_jsonl_objects`) for all JSONL framing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl_lines(path: str | Path) -> list[str]:
    """Return JSONL records split on the physical newline only.

    - reads text (utf-8, replacement on decode errors)
    - splits ONLY on ``"\\n"`` (never ``splitlines()``)
    - trims a trailing ``"\\r"`` from each physical line (CRLF tolerance)
    - skips only the final empty line if present (trailing newline)
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    parts = text.split("\n")
    lines: list[str] = []
    last = len(parts) - 1
    for i, part in enumerate(parts):
        line = part[:-1] if part.endswith("\r") else part
        if i == last and line == "":
            continue  # final empty line from a trailing newline
        lines.append(line)
    return lines


def load_jsonl_objects(path: str | Path) -> tuple[list[Any], int, int]:
    """Return (objects, malformed_count, nonempty_record_count) using physical-newline
    framing. Blank physical lines are ignored (not counted as records or malformed)."""
    objects: list[Any] = []
    malformed = 0
    nonempty = 0
    for line in read_jsonl_lines(path):
        if line.strip() == "":
            continue
        nonempty += 1
        try:
            objects.append(json.loads(line))
        except ValueError:
            malformed += 1
    return objects, malformed, nonempty


__all__ = ["read_jsonl_lines", "load_jsonl_objects"]
