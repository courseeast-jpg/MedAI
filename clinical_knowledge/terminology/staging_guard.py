"""CKA-TERM-01H staging guards for terminology safety checks.

The helpers inspect staged path names only. They never read terminology
content and never stage or unstage files.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TerminologyStagingGuardResult:
    terminology_data_staged: bool = False
    data_terminology_staged: bool = False
    blocked_reason_codes: tuple[str, ...] = field(default_factory=tuple)
    staged_count_safe: int = 0

    def safe_public_summary(self) -> dict:
        return {
            "terminology_data_staged": self.terminology_data_staged,
            "data_terminology_staged": self.data_terminology_staged,
            "blocked_reason_codes": list(self.blocked_reason_codes),
            "staged_count_safe": self.staged_count_safe,
        }


def check_terminology_staging(
    *,
    repo_root: Path | None = None,
    staged_paths: Iterable[str] | None = None,
) -> TerminologyStagingGuardResult:
    """Return a public-safe guard result for staged terminology artifacts."""
    if staged_paths is None:
        staged_paths = _git_staged_paths(repo_root)
    normalized = [str(path).replace("\\", "/").strip() for path in staged_paths if str(path).strip()]
    terminology_data_staged = any(_under(path, "terminology_data/") for path in normalized)
    data_terminology_staged = any(_under(path, "data/terminology/") for path in normalized)
    reasons: list[str] = []
    if terminology_data_staged:
        reasons.append("terminology_data_staged")
    if data_terminology_staged:
        reasons.append("data_terminology_staged")
    return TerminologyStagingGuardResult(
        terminology_data_staged=terminology_data_staged,
        data_terminology_staged=data_terminology_staged,
        blocked_reason_codes=tuple(reasons),
        staged_count_safe=len(normalized),
    )


def _git_staged_paths(repo_root: Path | None) -> list[str]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent.parent
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _under(path: str, prefix: str) -> bool:
    prefix = prefix.replace("\\", "/")
    return path == prefix.rstrip("/") or path.startswith(prefix)
