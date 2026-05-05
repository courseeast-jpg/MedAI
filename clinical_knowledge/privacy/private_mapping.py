"""Private mapping file management for CKA-B02.

Private replacement_map files are written only to gitignored paths.
They must never be staged, tracked, or included in public reports.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]  # clinical_knowledge/privacy -> clinical_knowledge -> repo root

DEFAULT_PRIVATE_MAPPING_PATH = (
    ROOT / "reports" / "cka_block02_privacy_boundary"
    / "private_sanitization_mapping_PRIVATE.json"
)


def write_private_mapping(mapping: Dict[str, str], path: Path = DEFAULT_PRIVATE_MAPPING_PATH) -> None:
    """Write replacement_map to a local private file.

    This file must be gitignored. Caller should verify with is_gitignored().
    The file is never committed or tracked.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")


def is_gitignored(path: Path, repo_root: Path = ROOT) -> bool:
    """Return True if *path* is covered by .gitignore rules."""
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        rel = path

    # git check-ignore: exits 0 if ignored, 1 if not, 128 on error
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(rel)],
        cwd=repo_root,
        capture_output=True,
    )
    return result.returncode == 0


def is_tracked_by_git(path: Path, repo_root: Path = ROOT) -> bool:
    """Return True if *path* is tracked (staged or committed) by Git."""
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        rel = path

    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(rel)],
        cwd=repo_root,
        capture_output=True,
    )
    return result.returncode == 0
