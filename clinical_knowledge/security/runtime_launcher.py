"""CKA-SEC-05 — runtime launcher helpers.

Pure-Python helpers used by `scripts/start_cka_encrypted_runtime_ui.py`
and by tests. They do NOT spawn Streamlit themselves — the script does
that. They DO encapsulate:

- key-prompt (twice, no echo, no logging)
- argparse refusal of --key / --encryption-key
- child-environment construction (key passed to child only)
- self-test against a temp encrypted store
"""
from __future__ import annotations

import argparse
import getpass
import os
import secrets
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from clinical_knowledge.security.empty_store_initializer import (
    EmptyStoreInitError,
    _lock_path_for,
)
from clinical_knowledge.security.encrypted_store import (
    EncryptedCKAStore,
    EncryptedStoreError,
)
from clinical_knowledge.security.key_policy import (
    KeyPolicyError,
    validate_operator_key,
)
from clinical_knowledge.security.runtime_config import (
    EncryptedRuntimeConfig,
)
from clinical_knowledge.security.runtime_factory import (
    RuntimeFactoryError,
    build_cka_runtime_store,
)


_TEST_KEY_ENV = "CKA_SEC04_TEST_KEY"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LauncherError(Exception):
    """Raised by the launcher when configuration / key handling fails safely."""


# ---------------------------------------------------------------------------
# argparse: refuse --key / --encryption-key by design
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="start_cka_encrypted_runtime_ui",
        description=(
            "Launch the MedAI Streamlit UI with the CKA encrypted runtime "
            "enabled. The encryption key is prompted via getpass — never "
            "accepted on the command line, never echoed, never logged."
        ),
    )
    p.add_argument(
        "--store-path", default=None,
        help="Path to the encrypted DB. Required for non-dry-run, non-self-test runs.",
    )
    p.add_argument("--port", type=int, default=8501,
                   help="Streamlit port (default 8501).")
    p.add_argument("--create-if-missing", action="store_true",
                   help="Create the empty encrypted store at --store-path if it does not exist.")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse config + validate provider; do NOT launch Streamlit; do NOT create real DB.")
    p.add_argument("--self-test", action="store_true",
                   help="Open an encrypted runtime against a synthetic temp DB; do NOT launch Streamlit.")
    p.add_argument("--test-mode", action="store_true",
                   help="Allow test-only behavior (e.g. CKA_SEC04_TEST_KEY env var). Ignored in production runs.")
    p.add_argument("--no-browser", action="store_true",
                   help="Pass --server.headless=true to Streamlit; do not open browser.")
    p.add_argument("--timeout-seconds", type=int, default=180,
                   help="Subprocess timeout for self-test runs.")
    # Defensive flags — refuse if present.
    p.add_argument("--key", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--encryption-key", action="store_true", help=argparse.SUPPRESS)
    return p


def reject_command_line_key(argv: Sequence[str]) -> Optional[str]:
    """Return a safe error string if --key / --encryption-key was passed; else None.

    This is checked BEFORE argparse runs, so we can refuse `--key=<value>`
    forms (which argparse would parse as an attempted value-on-store_true).
    """
    for token in argv:
        if token.startswith("--key=") or token == "--key":
            return "command_line_key_not_accepted"
        if token.startswith("--encryption-key=") or token == "--encryption-key":
            return "command_line_key_not_accepted"
    return None


# ---------------------------------------------------------------------------
# Key prompting
# ---------------------------------------------------------------------------


def prompt_key_twice(prompt_a: str = "Encryption key: ",
                     prompt_b: str = "Confirm encryption key: ") -> str:
    """Prompt for the key twice via getpass; refuse on mismatch.

    Calls getpass.getpass twice. Raises LauncherError on mismatch or empty.
    Tests monkey-patch `getpass.getpass` to drive deterministic input.
    """
    a = getpass.getpass(prompt_a)
    if a == "":
        raise LauncherError("empty_key_refused")
    b = getpass.getpass(prompt_b)
    if a != b:
        raise LauncherError("key_confirmation_mismatch")
    return a


def resolve_key(args: argparse.Namespace) -> str:
    """Resolve the operator key.

    - In `--dry-run`: return a synthetic in-process key (never logged).
    - In `--self-test` with `--test-mode`: read CKA_SEC04_TEST_KEY env var.
    - Otherwise: prompt via getpass twice.
    """
    if args.key or args.encryption_key:
        # argparse store_true was triggered (e.g., bare --key) — defensive refusal.
        raise LauncherError("command_line_key_not_accepted")
    if args.dry_run:
        return "synth_op_" + secrets.token_hex(16)
    if args.self_test and args.test_mode:
        v = os.environ.get(_TEST_KEY_ENV)
        if not v:
            raise LauncherError("test_mode_env_key_missing")
        return v
    if args.self_test and not args.test_mode:
        # Self-test without test-mode prompts the operator like a real run.
        return prompt_key_twice()
    # Normal interactive run.
    return prompt_key_twice()


# ---------------------------------------------------------------------------
# Child environment
# ---------------------------------------------------------------------------


def build_child_env(
    encryption_key: str,
    store_path: str,
    *,
    create_if_missing: bool = False,
    base_env: Optional[dict] = None,
) -> dict:
    """Build the env dict for the Streamlit child process.

    The encryption key is added to the child env ONLY. The caller is
    responsible for passing this dict to subprocess.run/Popen and not
    persisting it back into os.environ.
    """
    env = dict(base_env if base_env is not None else os.environ)
    env["MEDAI_LOCAL_ONLY"] = "1"
    env["MEDAI_ALLOW_EXTERNAL_API"] = "0"
    env["MEDAI_REQUIRE_PII_SCRUB"] = "1"
    env["MEDAI_PRIVACY_AUDIT"] = "1"
    env["MEDAI_CKA_ENCRYPTED_STORE_ENABLED"] = "1"
    env["MEDAI_CKA_ENCRYPTED_STORE_PATH"] = store_path
    env["MEDAI_CKA_ENCRYPTION_KEY"] = encryption_key
    if create_if_missing:
        env["MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING"] = "1"
    else:
        env.pop("MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING", None)
    return env


def child_env_keys_for_encrypted_runtime() -> tuple:
    """Sentinel — what env vars the encrypted-runtime child env adds."""
    return (
        "MEDAI_LOCAL_ONLY",
        "MEDAI_ALLOW_EXTERNAL_API",
        "MEDAI_REQUIRE_PII_SCRUB",
        "MEDAI_PRIVACY_AUDIT",
        "MEDAI_CKA_ENCRYPTED_STORE_ENABLED",
        "MEDAI_CKA_ENCRYPTED_STORE_PATH",
        "MEDAI_CKA_ENCRYPTION_KEY",
    )


# ---------------------------------------------------------------------------
# Self-test against a synthetic temp store
# ---------------------------------------------------------------------------


@dataclass
class SelfTestResult:
    """Public-report-safe result of the encrypted-runtime self-test."""

    passed: bool = False
    encrypted_store_opened: bool = False
    records_count: int = 0
    runtime_encryption_active: bool = False
    temp_files_staged: bool = False
    real_db_created: bool = False
    blocked_reason: Optional[str] = None

    def safe_public_summary(self) -> dict:
        return {
            "passed": self.passed,
            "encrypted_store_opened": self.encrypted_store_opened,
            "records_count": self.records_count,
            "runtime_encryption_active": self.runtime_encryption_active,
            "temp_files_staged": self.temp_files_staged,
            "real_db_created": self.real_db_created,
            "blocked_reason": self.blocked_reason,
        }


def _new_temp_db_path(prefix: str = "cka_sec05_selftest_") -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".db")
    os.close(fd)
    try:
        os.unlink(path)
    except OSError:
        pass
    return path


def _safe_unlink_pair(db_path: Optional[str]) -> None:
    if not db_path:
        return
    p = Path(db_path)
    try:
        p.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass
    sib = p.parent / (p.stem + ".manifest.json")
    try:
        sib.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass
    lock = _lock_path_for(p)
    try:
        lock.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass


def run_self_test(encryption_key: str) -> SelfTestResult:
    """Open an encrypted runtime against a synthetic temp DB; verify.

    The temp DB is deleted before this function returns. The path is
    never logged or returned in the public summary.
    """
    try:
        validate_operator_key(encryption_key)
    except KeyPolicyError as exc:
        return SelfTestResult(
            passed=False,
            blocked_reason=f"key_policy_{exc}",
        )

    db = _new_temp_db_path()
    try:
        cfg = EncryptedRuntimeConfig.for_test(
            db, encryption_key,
            encrypted_runtime_requested=True,
            create_if_missing=True,
        )
        try:
            r = build_cka_runtime_store(cfg)
        except RuntimeFactoryError as exc:
            return SelfTestResult(
                passed=False,
                blocked_reason=f"factory_{exc}",
            )

        if not isinstance(r.store, EncryptedCKAStore):
            return SelfTestResult(
                passed=False,
                blocked_reason="factory_returned_unexpected_store",
            )

        # Confirm an empty schema is in place.
        try:
            con = r.store._con
            row = con.execute("SELECT count(*) FROM cka_future_records").fetchone() if con else None
            count = int(row[0]) if row else 0
        except Exception:    # noqa: BLE001
            count = -1
        try:
            r.store.close()
        except Exception:    # noqa: BLE001
            pass

        return SelfTestResult(
            passed=(count == 0 and r.runtime_encryption_active is True),
            encrypted_store_opened=True,
            records_count=count if count >= 0 else 0,
            runtime_encryption_active=r.runtime_encryption_active,
            temp_files_staged=False,
            real_db_created=False,
            blocked_reason=None if count == 0 else "records_count_not_zero",
        )
    finally:
        _safe_unlink_pair(db)


# ---------------------------------------------------------------------------
# Streamlit child-launch construction (no actual launch here)
# ---------------------------------------------------------------------------


def build_streamlit_command(
    *,
    port: int,
    no_browser: bool,
    repo_root: Optional[Path] = None,
) -> List[str]:
    """Build the Streamlit subprocess argument list.

    Returns argv only; does NOT execute. The launcher script is
    responsible for invoking subprocess.run / Popen with this argv and
    the env dict from `build_child_env`.
    """
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app/main.py",
        "--server.port", str(int(port)),
        "--browser.gatherUsageStats", "false",
    ]
    if no_browser:
        cmd += ["--server.headless", "true"]
    return cmd
