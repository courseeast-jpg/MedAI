"""CKA-SEC-06 — operator-facing CLI to rotate the SQLCipher encryption key.

Modes:
  --dry-run                                        Synthetic temp DB rotation
                                                   rehearsal. No real DB touched.
  --db-path <path> --backup-path <path>            Real-DB rotation. Refuses
       [--approve-real-rotation]                   without --approve-real-rotation
       [--test-mode]                               unless the path is in temp.
                                                   Old key prompted via getpass
                                                   ONCE; new key prompted via
                                                   getpass TWICE.

Hard rules (defensive):
- Refuse `--old-key`, `--new-key`, `--key`, `--encryption-key` on the CLI.
- Never echo, store, or log either key.
- Refuse empty old key, empty new key, mismatched new key, same old/new key.
- Require a verified backup BEFORE rotation; refuse otherwise.
"""
from __future__ import annotations

import argparse
import getpass
import os
import secrets
import sys
import tempfile
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.security import (    # noqa: E402
    KeyRotationError,
    LauncherError,
    detect_sqlcipher_provider,
    reject_command_line_key,
    rotate_sqlcipher_key,
    rotation_passed,
    run_synthetic_rotation_rehearsal,
)
from clinical_knowledge.security.empty_store_initializer import (    # noqa: E402
    _lock_path_for,
)


_OLD_TEST_KEY_ENV = "CKA_SEC06_OLD_TEST_KEY"
_NEW_TEST_KEY_ENV = "CKA_SEC06_NEW_TEST_KEY"


def reject_rotation_command_line_keys(argv) -> Optional[str]:
    """Refuse `--old-key=`, `--new-key=`, plus the SEC-05 `--key=` markers."""
    for token in argv:
        for prefix in ("--old-key", "--new-key"):
            if token == prefix or token.startswith(prefix + "="):
                return "command_line_key_not_accepted"
    return reject_command_line_key(argv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cka_encrypted_store_rotate_key",
        description="Rotate the SQLCipher encryption key on an encrypted CKA store.",
    )
    p.add_argument("--db-path", default=None,
                   help="Path to the encrypted DB to rotate.")
    p.add_argument("--backup-path", default=None,
                   help="Path where the pre-rotation backup will be written.")
    p.add_argument("--approve-real-rotation", action="store_true",
                   help="Required for non-temp DB paths.")
    p.add_argument("--dry-run", action="store_true",
                   help="Synthetic temp rotation rehearsal; no real DB touched.")
    p.add_argument("--test-mode", action="store_true",
                   help="(test-only) read keys from CKA_SEC06_OLD_TEST_KEY / "
                        "CKA_SEC06_NEW_TEST_KEY env vars.")
    # Defensive: refuse CLI key flags.
    p.add_argument("--old-key", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--new-key", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--key", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--encryption-key", action="store_true", help=argparse.SUPPRESS)
    return p


def prompt_old_key() -> str:
    a = getpass.getpass("Old encryption key: ")
    if a == "":
        raise LauncherError("empty_old_key_refused")
    return a


def prompt_new_key_twice() -> str:
    a = getpass.getpass("New encryption key: ")
    if a == "":
        raise LauncherError("empty_new_key_refused")
    b = getpass.getpass("Confirm new encryption key: ")
    if a != b:
        raise LauncherError("new_key_confirmation_mismatch")
    return a


def resolve_keys(args: argparse.Namespace) -> tuple[str, str]:
    if args.old_key or args.new_key or args.key or args.encryption_key:
        raise LauncherError("command_line_key_not_accepted")
    if args.dry_run:
        a = "synth_op_" + secrets.token_hex(16)
        b = "synth_op_" + secrets.token_hex(16)
        while a == b:
            b = "synth_op_" + secrets.token_hex(16)
        return a, b
    if args.test_mode:
        ok = os.environ.get(_OLD_TEST_KEY_ENV)
        nk = os.environ.get(_NEW_TEST_KEY_ENV)
        if not ok or not nk:
            raise LauncherError("test_mode_env_keys_missing")
        if ok == nk:
            raise LauncherError("same_old_new_key_refused")
        return ok, nk
    old = prompt_old_key()
    new = prompt_new_key_twice()
    if old == new:
        raise LauncherError("same_old_new_key_refused")
    return old, new


def _print_safe_summary(label: str, summary: dict) -> None:
    print(f"=== {label} ===")
    for k, v in summary.items():
        if isinstance(v, str) and v.startswith("synth_op_"):
            print(f"  {k}: <redacted>")
        else:
            print(f"  {k}: {v}")


def main(argv: Optional[List[str]] = None) -> int:
    raw = argv if argv is not None else sys.argv[1:]
    refusal = reject_rotation_command_line_keys(raw)
    if refusal:
        print(f"ERROR: {refusal} (encryption keys cannot be supplied on the command line)",
              file=sys.stderr)
        return 2

    parser = build_parser()
    args = parser.parse_args(raw)

    provider = detect_sqlcipher_provider()
    if not provider.available:
        print("ERROR: sqlcipher_provider_unavailable", file=sys.stderr)
        return 3

    try:
        old_key, new_key = resolve_keys(args)
    except LauncherError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        try:
            result, _src, _bkp = run_synthetic_rotation_rehearsal(record_count=3)
        except KeyRotationError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 4
        _print_safe_summary("CKA-SEC-06 rotation (dry-run)", result.safe_public_summary())
        # Drop key references.
        old_key = ""    # noqa: F841
        new_key = ""    # noqa: F841
        del old_key, new_key
        return 0 if rotation_passed(result) else 1

    # Non-dry-run: require explicit DB + backup paths.
    if not args.db_path or not args.backup_path:
        print("ERROR: --db-path and --backup-path are required in non-dry-run mode.",
              file=sys.stderr)
        return 2

    try:
        result = rotate_sqlcipher_key(
            db_path=args.db_path,
            old_key=old_key,
            new_key=new_key,
            backup_path=args.backup_path,
            require_verified_backup=True,
            dry_run=False,
            approve_real_rotation=args.approve_real_rotation,
            test_mode=bool(args.test_mode),
        )
    except KeyRotationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4

    _print_safe_summary("CKA-SEC-06 rotation", result.safe_public_summary())
    # Drop key references.
    old_key = ""    # noqa: F841
    new_key = ""    # noqa: F841
    del old_key, new_key
    return 0 if rotation_passed(result) else 1


if __name__ == "__main__":
    sys.exit(main())
