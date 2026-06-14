"""Shared resolver for the canonical 478-request private outbound batch path.

The canonical batch lives OUTSIDE the repo under the local app-data tree. Resolving it
purely via ``os.path.expandvars("%LOCALAPPDATA%\\...")`` is fragile: if ``LOCALAPPDATA``
is unset/empty in the runtime environment, ``expandvars`` returns the literal
``%LOCALAPPDATA%\\...`` (a nonexistent path) and callers falsely report
``canonical_batch_missing``.

This resolver tries, in order: the explicit expected path, the ``%LOCALAPPDATA%``
expansion, and a ``Path.home()``-based path; it returns the first that exists. A
read-only file still resolves (``is_file()`` is True for read-only files).
"""
from __future__ import annotations

import os
from pathlib import Path

REL = ("MedAI_Private", "ai_extraction_17B_R2_R1_478_repaired", "outbound_requests_private.jsonl")
EXPECTED_CANONICAL_BATCH = (
    r"C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17B_R2_R1_478_repaired"
    r"\outbound_requests_private.jsonl"
)


def canonical_batch_candidates() -> list[Path]:
    """Ordered, de-duplicated candidate paths for the canonical batch."""
    cands: list[Path] = [Path(EXPECTED_CANONICAL_BATCH)]
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        cands.append(Path(localappdata).joinpath(*REL))
    cands.append(Path.home().joinpath("AppData", "Local", *REL))
    seen: set[str] = set()
    ordered: list[Path] = []
    for c in cands:
        key = str(c)
        if key not in seen:
            seen.add(key)
            ordered.append(c)
    return ordered


def resolve_canonical_batch() -> tuple[Path, bool]:
    """Return (path, resolved_exists). The first existing candidate wins; if none
    exists, returns the explicit expected path with resolved_exists=False (for
    reporting)."""
    for cand in canonical_batch_candidates():
        if cand.is_file():
            return cand, True
    return Path(EXPECTED_CANONICAL_BATCH), False


def canonical_batch_dir() -> Path:
    """Directory holding the canonical batch + integrity sidecars (resolved)."""
    return resolve_canonical_batch()[0].parent


__all__ = ["EXPECTED_CANONICAL_BATCH", "canonical_batch_candidates",
           "resolve_canonical_batch", "canonical_batch_dir"]
