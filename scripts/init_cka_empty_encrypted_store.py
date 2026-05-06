"""CKA-SEC-03A — operator-facing initializer for the empty encrypted future store.

Modes:
  --dry-run                         Use a synthetic temp DB and an in-process
                                    synthetic key. Never creates a real store.
  --target <path>
  --approve-real-store-creation     Required to create a real store outside
                                    the system temp directory. The script
                                    will prompt for the encryption key TWICE
                                    via getpass and refuse on mismatch.
  --test-mode                       (Internal/test only.) Read the synthetic
                                    key from the env var CKA_SEC03A_TEST_KEY
                                    instead of prompting. Allowed only for
                                    --dry-run or for explicitly-approved
                                    temp targets.
  --overwrite                       Permit overwriting an existing target.
                                    Refused outside temp dir without
                                    --approve-real-store-creation.

Hard rules:
- The encryption key is NEVER accepted on the command line.
- The encryption key is NEVER written to .env, reports, or stdout.
- Confirm-twice + getpass is mandatory in non-test, non-dry-run paths.
"""
from __future__ import annotations

import argparse
import getpass
import os
import secrets
import sys
import tempfile
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.security import (    # noqa: E402
    EmptyStoreInitError,
    initialize_empty_encrypted_store,
    initializer_will_create_real_store,
)


_TEST_KEY_ENV = "CKA_SEC03A_TEST_KEY"


def _new_synth_key() -> str:
    return "synth_op_" + secrets.token_hex(16)


def _prompt_key_twice() -> str:
    """Prompt for the encryption key twice via getpass; refuse mismatch."""
    a = getpass.getpass("Encryption key: ")
    b = getpass.getpass("Confirm encryption key: ")
    if a != b:
        raise EmptyStoreInitError("key_confirmation_mismatch")
    return a


def _make_temp_target() -> str:
    fd, path = tempfile.mkstemp(prefix="cka_sec03a_init_", suffix=".db")
    os.close(fd)
    try:
        os.unlink(path)
    except OSError:
        pass
    return path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="init_cka_empty_encrypted_store",
        description="Initialize the encrypted empty future CKA store.",
    )
    p.add_argument("--target", default=None,
                   help="Target DB path. If omitted in --dry-run, a temp path is used.")
    p.add_argument("--dry-run", action="store_true",
                   help="Use a synthetic temp DB and synthetic key. No real store.")
    p.add_argument("--approve-real-store-creation", action="store_true",
                   help="Approval flag required to create a real store outside temp.")
    p.add_argument("--test-mode", action="store_true",
                   help="Read key from env var CKA_SEC03A_TEST_KEY (test-only).")
    p.add_argument("--overwrite", action="store_true",
                   help="Permit overwriting an existing target file.")
    # Defensive: refuse any --key / --encryption-key flag.
    p.add_argument("--key", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--encryption-key", action="store_true",
                   help=argparse.SUPPRESS)
    return p


def _resolve_key(args: argparse.Namespace) -> str:
    if args.key or args.encryption_key:
        # Defense-in-depth: refuse explicitly.
        raise EmptyStoreInitError("command_line_key_not_accepted")
    if args.dry_run:
        return _new_synth_key()
    if args.test_mode:
        v = os.environ.get(_TEST_KEY_ENV)
        if not v:
            raise EmptyStoreInitError("test_mode_env_key_missing")
        return v
    # Interactive mode: getpass twice.
    return _prompt_key_twice()


def _resolve_target(args: argparse.Namespace) -> str:
    if args.target:
        return args.target
    if args.dry_run:
        return _make_temp_target()
    raise EmptyStoreInitError("target_required_outside_dry_run")


def main(argv: Optional[list] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # The argv tokens are NEVER allowed to carry a key value.
    raw_argv = argv if argv is not None else sys.argv[1:]
    for token in raw_argv:
        if token.startswith("--key=") or token.startswith("--encryption-key="):
            print("ERROR: key cannot be passed on the command line.", file=sys.stderr)
            return 2

    try:
        target = _resolve_target(args)
    except EmptyStoreInitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    will_create_real = initializer_will_create_real_store(
        target, args.approve_real_store_creation
    )
    if will_create_real and not args.approve_real_store_creation:
        print("ERROR: real_store_creation_not_approved", file=sys.stderr)
        return 2

    try:
        key = _resolve_key(args)
    except EmptyStoreInitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        result = initialize_empty_encrypted_store(
            target_path=target,
            encryption_key=key,
            approve_real_store_creation=args.approve_real_store_creation,
            overwrite=args.overwrite,
            create_manifest=True,
        )
    except EmptyStoreInitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Print only the SAFE PUBLIC SUMMARY. The key is out of scope here.
    summary = result.safe_public_summary()
    # Defensive: ensure no unexpected key-shaped string slips out.
    safe_keys = (
        "success", "target_safe_hash", "schema_created", "records_count",
        "ledger_events_count", "correct_key_read_passed",
        "wrong_key_failure_passed", "plaintext_absence_verified",
        "runtime_active", "main_store_migration_performed",
        "real_data_migrated", "operator_approved_creation", "overwrite_used",
        "lock_file_used", "lock_file_left_behind", "manifest_written",
        "db_file_staged",
    )
    print("=== CKA-SEC-03A initialization summary ===")
    for k in safe_keys:
        print(f"  {k}: {summary.get(k)}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
