"""CKA-SEC-05 — operator-facing encrypted-runtime launcher for MedAI.

Modes:
  --dry-run                           Parse + provider check; no Streamlit, no real DB.
  --self-test [--test-mode]           Open an encrypted runtime against a synthetic
                                      temp DB; verify; exit. No Streamlit launch.
  --store-path <path> [--port 8501]
                       [--create-if-missing] [--no-browser]
                                      Normal interactive run: prompt key via getpass
                                      twice, set env on a CHILD process only, launch
                                      Streamlit. Wrong key → no silent fallback.

Hard rules:
- Encryption key is NEVER accepted on the command line.
- Encryption key is NEVER printed, written to disk, or logged.
- Default behaviour is unchanged unless this script is run.
- Streamlit child env carries the key; the parent's os.environ is NOT modified.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.security import (    # noqa: E402
    LauncherError,
    build_arg_parser,
    build_child_env,
    build_streamlit_command,
    detect_sqlcipher_provider,
    reject_command_line_key,
    resolve_key,
    run_self_test,
)
from clinical_knowledge.security.empty_store_initializer import (    # noqa: E402
    EmptyStoreInitError,
    initialize_empty_encrypted_store,
)


def _print_safe_summary(label: str, summary: dict) -> None:
    """Print a public-safe summary dict. Filters out anything key-shaped."""
    print(f"=== {label} ===")
    for k, v in summary.items():
        # Defensive: never print a value that looks like a key.
        if isinstance(v, str) and len(v) >= 12 and v.startswith("synth_op_"):
            print(f"  {k}: <redacted>")
        else:
            print(f"  {k}: {v}")


def _maybe_create_real_store(
    target_path: str, encryption_key: str, allowed: bool,
) -> bool:
    """Create the encrypted store at `target_path` only if allowed and missing.

    Returns True if a new store was created. Does NOT log or print the key.
    """
    p = Path(target_path)
    if p.exists():
        return False
    if not allowed:
        raise LauncherError("encrypted_target_missing_create_if_missing_false")
    try:
        initialize_empty_encrypted_store(
            target_path=target_path,
            encryption_key=encryption_key,
            approve_real_store_creation=True,
            overwrite=False,
            create_manifest=True,
        )
    except EmptyStoreInitError as exc:
        raise LauncherError(f"empty_store_init_{exc}") from None
    return True


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = argv if argv is not None else sys.argv[1:]

    # 1. Defensive: refuse --key / --encryption-key BEFORE argparse runs.
    refusal = reject_command_line_key(raw_argv)
    if refusal:
        print(f"ERROR: {refusal} (the encryption key cannot be supplied on the command line)",
              file=sys.stderr)
        return 2

    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)

    # 2. Provider must be available for any non-vacuous path.
    provider = detect_sqlcipher_provider()
    if not provider.available and not args.dry_run:
        print("ERROR: sqlcipher_provider_unavailable", file=sys.stderr)
        return 3

    # 3. Resolve the key (prompts twice in interactive mode).
    try:
        encryption_key = resolve_key(args)
    except LauncherError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # 4. --dry-run: validate + exit. No DB created. No Streamlit.
    if args.dry_run:
        print("dry_run=ok")
        print(f"provider_available={provider.available}")
        print(f"store_path_supplied={bool(args.store_path)}")
        print(f"create_if_missing={args.create_if_missing}")
        print("encryption_runtime_default_off=True")
        return 0

    # 5. --self-test: open encrypted runtime against a temp DB. No Streamlit.
    if args.self_test:
        result = run_self_test(encryption_key)
        _print_safe_summary("CKA-SEC-05 self-test", result.safe_public_summary())
        return 0 if result.passed else 1

    # 6. Normal launch path.
    if not args.store_path:
        print("ERROR: --store-path is required for normal runs (use --dry-run or --self-test otherwise).",
              file=sys.stderr)
        return 2

    target = Path(args.store_path)
    created_now = False
    try:
        if not target.exists():
            created_now = _maybe_create_real_store(
                str(target), encryption_key, args.create_if_missing,
            )
    except LauncherError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # 7. Build child env (key only goes here, never to os.environ).
    child_env = build_child_env(
        encryption_key=encryption_key,
        store_path=str(target),
        create_if_missing=args.create_if_missing,
        base_env=os.environ,
    )

    # 8. Spawn Streamlit. We deliberately do NOT mutate the parent
    # process's os.environ — only the child sees the key.
    cmd = build_streamlit_command(
        port=int(args.port),
        no_browser=bool(args.no_browser),
        repo_root=REPO_ROOT,
    )
    print(f"Launching Streamlit on port {args.port} (encrypted runtime ON, child-only env).")
    if created_now:
        print("Created an empty encrypted future store at the configured path.")
    try:
        proc = subprocess.Popen(
            cmd, env=child_env, cwd=str(REPO_ROOT),
        )
    except FileNotFoundError:
        print("ERROR: python -m streamlit is not installed in this environment.",
              file=sys.stderr)
        return 4

    try:
        return proc.wait()
    except KeyboardInterrupt:
        # Clean shutdown on Ctrl+C.
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)    # type: ignore[attr-defined]
            else:
                proc.send_signal(signal.SIGINT)
            try:
                return proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                return 130
        except Exception:    # noqa: BLE001
            try:
                proc.kill()
            except Exception:    # noqa: BLE001
                pass
            return 130
    finally:
        # Defensive: clear the local key reference. (Python strings are
        # immutable + may live longer in memory pools, but we drop the
        # last reference we hold.)
        encryption_key = ""    # noqa: F841
        del encryption_key


if __name__ == "__main__":
    sys.exit(main())
