"""Confirmation-gated quarantine and empty MKB DB recreation tool."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import DB_PATH  # noqa: E402
from app.startup_preflight import file_size_bucket, safe_relative_path  # noqa: E402
from mkb.sqlite_store import SQLiteStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quarantine and recreate local MKB DB with explicit confirmation.")
    parser.add_argument("--db-path", default=str(DB_PATH))
    parser.add_argument("--backup-root", default="data/mkb_quarantine")
    parser.add_argument("--confirm-quarantine-recreate", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def create_fresh_db(db_path: Path) -> None:
    app_key = os.getenv("DB_ENCRYPTION_KEY", "default_dev_key")
    SQLiteStore(db_path, app_key)


def main() -> int:
    args = parse_args()
    db_path = resolve_path(args.db_path)
    backup_root = resolve_path(args.backup_root)
    if not args.confirm_quarantine_recreate:
        print(
            json.dumps(
                {
                    "conclusion": "blocked_confirmation_required",
                    "db_path_label": safe_relative_path(db_path),
                    "db_modified": False,
                    "backup_or_quarantine_created": False,
                    "required_flag": "--confirm-quarantine-recreate",
                },
                indent=2,
            )
        )
        return 2

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_path = backup_root / f"mkb_quarantine_{timestamp}.db"
    copied = False
    original_bucket = file_size_bucket(db_path)
    if db_path.exists():
        shutil.move(str(db_path), str(backup_path))
        copied = True
    db_path.parent.mkdir(parents=True, exist_ok=True)
    create_fresh_db(db_path)
    print(
        json.dumps(
            {
                "conclusion": "mkb_db_quarantined_and_recreated",
                "db_path_label": safe_relative_path(db_path),
                "backup_label": safe_relative_path(backup_path),
                "original_db_size_bucket": original_bucket,
                "new_db_size_bucket": file_size_bucket(db_path),
                "db_modified": True,
                "backup_or_quarantine_created": copied,
                "fresh_db_created": db_path.exists(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
