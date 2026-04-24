from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "phase10"
DEFAULT_SNAPSHOTS_DIR = ROOT / "artifacts" / "phase10_snapshots"

EXCLUDE_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".env",
    "artifacts",
}


def copy_tree_filtered(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        if item.name in EXCLUDE_NAMES:
            continue
        target = dst / item.name
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            copy_tree_filtered(item, target)
        else:
            shutil.copy2(item, target)


def git_commit_hash() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def write_metadata(snapshot_dir: Path, artifacts_dir: Path, timestamp: str) -> None:
    metadata = {
        "timestamp": timestamp,
        "git_commit_hash": git_commit_hash(),
        "artifacts_dir": str(artifacts_dir),
        "python": sys.version,
    }
    (snapshot_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "metadata" / "snapshot_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def create_zip(snapshot_dir: Path) -> Path:
    zip_path = snapshot_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in snapshot_dir.rglob("*"):
            archive.write(file_path, file_path.relative_to(snapshot_dir.parent))
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--snapshot-root", default=str(DEFAULT_SNAPSHOTS_DIR))
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    snapshot_root = Path(args.snapshot_root)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = snapshot_root / f"phase10_snapshot_{timestamp}"

    (snapshot_dir / "source").mkdir(parents=True, exist_ok=True)
    copy_tree_filtered(ROOT, snapshot_dir / "source")

    if artifacts_dir.exists():
        shutil.copytree(artifacts_dir, snapshot_dir / "artifacts", dirs_exist_ok=True)

    write_metadata(snapshot_dir, artifacts_dir, timestamp)

    summary_lines = [
        "# Phase 10 Final Stable Snapshot",
        "",
        f"- Timestamp: {timestamp}",
        f"- Git commit hash: {git_commit_hash()}",
        f"- Source snapshot: {snapshot_dir / 'source'}",
        f"- Artifacts copied: {artifacts_dir.exists()}",
    ]
    (snapshot_dir / "PHASE10_SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    zip_path = create_zip(snapshot_dir)
    print(json.dumps({
        "snapshot_dir": str(snapshot_dir),
        "snapshot_zip": str(zip_path),
        "git_commit_hash": git_commit_hash(),
        "timestamp": timestamp,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
