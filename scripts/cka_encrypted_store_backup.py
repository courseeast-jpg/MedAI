"""CKA-SEC-07 — operator-facing backup CLI for the encrypted future store.

Modes:
  --dry-run                              Synthetic temp source + backup;
                                         no real DB touched. Always passes
                                         a synthetic key.
  --source <path> --backup <path>        Real backup. Operator key prompted
                                         twice via getpass; never echoed,
                                         never stored, never logged.
  --overwrite                            Permit overwriting an existing
                                         backup target (default refuses).
  --test-mode                            (test-only) Read key from
                                         CKA_SEC04_TEST_KEY env var instead
                                         of prompting.

Hard rules:
- The encryption key is NEVER accepted on the command line.
- The encryption key is NEVER printed, written to disk, or logged.
- No real source DB is touched in --dry-run.
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
    EncryptedBackupError,
    EncryptedRuntimeConfig,
    LauncherError,
    build_cka_runtime_store,
    create_encrypted_backup,
    detect_sqlcipher_provider,
    prompt_key_twice,
    reject_command_line_key,
)
from clinical_knowledge.security.empty_store_initializer import (    # noqa: E402
    _lock_path_for,
)


_TEST_KEY_ENV = "CKA_SEC04_TEST_KEY"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cka_encrypted_store_backup",
        description="Create an encrypted byte-level backup of an encrypted CKA store.",
    )
    p.add_argument("--source", default=None,
                   help="Path to the encrypted source DB.")
    p.add_argument("--backup", default=None,
                   help="Target path for the backup file.")
    p.add_argument("--dry-run", action="store_true",
                   help="Use a synthetic temp source + backup; no real DB touched.")
    p.add_argument("--overwrite", action="store_true",
                   help="Permit overwriting an existing backup target.")
    p.add_argument("--test-mode", action="store_true",
                   help="(test-only) read key from CKA_SEC04_TEST_KEY env var.")
    # Defensive: refuse --key / --encryption-key.
    p.add_argument("--key", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--encryption-key", action="store_true", help=argparse.SUPPRESS)
    return p


def _resolve_key(args: argparse.Namespace) -> str:
    if args.key or args.encryption_key:
        raise LauncherError("command_line_key_not_accepted")
    if args.dry_run:
        return "synth_op_" + secrets.token_hex(16)
    if args.test_mode:
        v = os.environ.get(_TEST_KEY_ENV)
        if not v:
            raise LauncherError("test_mode_env_key_missing")
        return v
    return prompt_key_twice()


def _new_temp_db(prefix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".db")
    os.close(fd)
    try:
        os.unlink(path)
    except OSError:
        pass
    return path


def _cleanup(p: str) -> None:
    path = Path(p)
    try:
        path.unlink(missing_ok=True)
    except Exception:    # noqa: BLE001
        pass
    for ext in (".manifest.json", ".backup-manifest.json"):
        sib = path.parent / (path.stem + ext)
        try:
            sib.unlink(missing_ok=True)
        except Exception:    # noqa: BLE001
            pass
    lock = _lock_path_for(path)
    try:
        lock.unlink(missing_ok=True)
    except Exception:    # noqa: BLE001
        pass


def _print_safe_summary(label: str, summary: dict) -> None:
    print(f"=== {label} ===")
    for k, v in summary.items():
        if isinstance(v, str) and v.startswith("synth_op_"):
            print(f"  {k}: <redacted>")
        else:
            print(f"  {k}: {v}")


def main(argv: Optional[List[str]] = None) -> int:
    raw = argv if argv is not None else sys.argv[1:]
    refusal = reject_command_line_key(raw)
    if refusal:
        print(f"ERROR: {refusal} (the encryption key cannot be supplied on the command line)",
              file=sys.stderr)
        return 2

    parser = _build_parser()
    args = parser.parse_args(raw)

    provider = detect_sqlcipher_provider()
    if not provider.available:
        print("ERROR: sqlcipher_provider_unavailable", file=sys.stderr)
        return 3

    try:
        key = _resolve_key(args)
    except LauncherError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        # Build a synthetic source + backup in temp paths.
        src = _new_temp_db("cka_sec07_dryrun_src_")
        bk = _new_temp_db("cka_sec07_dryrun_bk_")
        try:
            cfg = EncryptedRuntimeConfig.for_test(
                src, key, encrypted_runtime_requested=True, create_if_missing=True,
            )
            r = build_cka_runtime_store(cfg)
            con = r.store._con
            con.execute(
                "INSERT INTO cka_future_records (record_id, label, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("rec_dry_001", "synthetic_dry_label",
                 "synthetic_dry_payload", "2026-05-06T00:00:00Z"),
            )
            con.commit()
            r.store.close()
            br = create_encrypted_backup(src, bk, key)
            _print_safe_summary("CKA-SEC-07 backup (dry-run)", br.safe_public_summary())
            return 0 if br.success else 1
        finally:
            _cleanup(src)
            _cleanup(bk)

    # Normal mode: source + backup paths required.
    if not args.source or not args.backup:
        print("ERROR: --source and --backup are required in non-dry-run mode.",
              file=sys.stderr)
        return 2

    try:
        result = create_encrypted_backup(
            source_path=args.source,
            backup_path=args.backup,
            encryption_key=key,
            overwrite=args.overwrite,
            write_manifest=True,
        )
    except EncryptedBackupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4

    _print_safe_summary("CKA-SEC-07 backup", result.safe_public_summary())
    # Defensive: drop the local key reference.
    key = ""    # noqa: F841
    del key
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
