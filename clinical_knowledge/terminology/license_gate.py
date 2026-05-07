"""CKA-TERM-01 — license gate.

Refuses real licensed-import unless the operator explicitly acknowledges
their license. Two acknowledgment paths are permitted:

1. **Test-mode env var** — `CKA_TERM01_TEST_LICENSE_ACK=1` (only when
   the caller passes `test_mode=True`). Used by the validation script
   and tests with synthetic data only.

2. **Local gitignored ack file** — `terminology_data/LICENSE_ACK_PRIVATE.json`
   on the operator's machine. The contents of this file are never
   read into a public report — only its presence (and a small set of
   safe fields) is checked.

The license-text itself is NEVER copied into reports, logs, or stdout.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set

from clinical_knowledge.terminology.models import TerminologySystem


_TEST_LICENSE_ACK_ENV = "CKA_TERM01_TEST_LICENSE_ACK"
_LICENSE_ACK_FILENAME = "LICENSE_ACK_PRIVATE.json"
_TERMINOLOGY_ROOT_DEFAULT = "terminology_data"

# Real licensed systems require an acknowledgment. The synthetic test
# system never does.
_REAL_SYSTEMS: Set[str] = {
    TerminologySystem.UMLS.value,
    TerminologySystem.SNOMED_CT.value,
    TerminologySystem.RXNORM.value,
    TerminologySystem.LOINC.value,
}


class LicenseGateError(Exception):
    """Raised when a real licensed-import is attempted without acknowledgment."""


def _truthy(raw: Optional[str]) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_local_ack_file(path: Optional[Path]) -> Optional[dict]:
    """Read the local ack JSON. Returns None if missing or unreadable.

    The ack file has a simple structure:
        {
          "operator_acknowledged": true,
          "acknowledged_systems": ["umls", "snomed_ct", ...]
        }
    No license text or operator identity is read. Even if the ack file
    contains extra fields, only the two fields above are inspected.
    """
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:    # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    return data


def _resolve_default_ack_file(repo_root: Optional[Path] = None) -> Path:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / _TERMINOLOGY_ROOT_DEFAULT / _LICENSE_ACK_FILENAME


def license_acknowledged_for(
    system: TerminologySystem,
    *,
    local_ack_file: Optional[Path] = None,
    env: Optional[dict] = None,
    test_mode: bool = False,
) -> bool:
    """Return True if the operator has acknowledged the license for `system`.

    - The synthetic-test system always returns True (no license needed).
    - In `test_mode=True`, the env var `CKA_TERM01_TEST_LICENSE_ACK=1`
      is honoured.
    - For real systems in non-test-mode, ONLY the local ack file path
      is consulted; env-var test bypass is ignored.
    """
    if system == TerminologySystem.SYNTHETIC_TEST:
        return True

    e = env if env is not None else os.environ
    if test_mode and _truthy(e.get(_TEST_LICENSE_ACK_ENV)):
        return True

    ack_path = local_ack_file if local_ack_file is not None else _resolve_default_ack_file()
    data = _read_local_ack_file(ack_path)
    if not data:
        return False
    if data.get("operator_acknowledged") is not True:
        return False
    acked = data.get("acknowledged_systems") or []
    if not isinstance(acked, list):
        return False
    return system.value in {str(s).lower() for s in acked}


def verify_operator_license_acknowledgment(
    system: TerminologySystem,
    *,
    local_ack_file: Optional[Path] = None,
    env: Optional[dict] = None,
    test_mode: bool = False,
) -> dict:
    """Return a public-report-safe dict describing the license-gate state.

    - Never copies license text.
    - Never includes the operator's identity / file paths.
    - Returns only flags + a safe `system` value.
    """
    acked = license_acknowledged_for(
        system,
        local_ack_file=local_ack_file,
        env=env,
        test_mode=test_mode,
    )
    return {
        "system": system.value,
        "license_required": system.value in _REAL_SYSTEMS,
        "license_confirmed": acked,
        "test_mode": bool(test_mode),
        "license_text_written_to_public_reports": False,
    }


def require_license_acknowledgment(
    system: TerminologySystem,
    *,
    local_ack_file: Optional[Path] = None,
    env: Optional[dict] = None,
    test_mode: bool = False,
) -> None:
    """Raise LicenseGateError if no acknowledgment is in place for `system`."""
    if not license_acknowledged_for(
        system, local_ack_file=local_ack_file, env=env, test_mode=test_mode,
    ):
        raise LicenseGateError(f"license_not_acknowledged:{system.value}")
