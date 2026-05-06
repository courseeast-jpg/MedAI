"""CKA-SEC-02 — key management policy.

This module is *policy + enforcement helpers*. It does NOT generate,
store, or log encryption keys. It defines:

- the rules an operator-supplied key must satisfy
- the validation function used by the migration plan + rehearsal
- the public-report-safe summary of policy state

Rules (binding):
- No hardcoded encryption key.
- No key in reports.
- No key in Git.
- No key in environment dump.
- Key must be operator-provided at migration time.
- Key must be confirmed twice by operator in future SEC-03.
- Recovery warning: lost key means encrypted DB cannot be recovered.
- Key rotation is OUT OF SCOPE for SEC-02.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


# Forbidden hardcoded markers used in tests/validation. If any of these
# strings appear as the *whole* key, validation refuses the key. They
# are NOT real keys — they are sentinels of bad practice.
_HARDCODED_KEY_MARKERS: tuple = (
    "password",
    "changeme",
    "default",
    "test",
    "test123",
    "admin",
    "key",
    "secret",
    "0000",
    "1234",
    "abcd",
    "your_key_here",
    "<key>",
    "REPLACE_ME",
)

_MIN_KEY_LEN = 12


class KeyPolicyError(ValueError):
    """Raised when an operator-supplied key violates policy."""


def validate_operator_key(candidate: object) -> None:
    """Validate an operator-supplied encryption key.

    Raises KeyPolicyError if the candidate is empty, too short, the
    wrong type, or matches a hardcoded marker. Never logs the key.
    """
    if not isinstance(candidate, str):
        raise KeyPolicyError("key_must_be_str")
    if candidate == "":
        raise KeyPolicyError("empty_key_refused")
    if len(candidate) < _MIN_KEY_LEN:
        raise KeyPolicyError("key_too_short")
    if candidate.strip().lower() in _HARDCODED_KEY_MARKERS:
        raise KeyPolicyError("hardcoded_key_marker_refused")


@dataclass
class KeyPolicyStatus:
    """Public-report-safe summary of key policy state."""

    no_hardcoded_key_in_code: bool = True
    operator_provided_key_required: bool = True
    key_logged_in_reports: bool = False
    key_committed_to_git: bool = False
    key_in_environment_dump: bool = False
    confirm_twice_required_in_sec03: bool = True
    recovery_warning_documented: bool = True
    rotation_out_of_scope_for_sec02: bool = True

    def safe_public_summary(self) -> dict:
        # Public report exposes only flags — never any key value.
        return {
            "no_hardcoded_key_in_code": self.no_hardcoded_key_in_code,
            "operator_provided_key_required": self.operator_provided_key_required,
            "key_logged_in_reports": self.key_logged_in_reports,
            "key_committed_to_git": self.key_committed_to_git,
            "key_in_environment_dump": self.key_in_environment_dump,
            "confirm_twice_required_in_sec03": self.confirm_twice_required_in_sec03,
            "recovery_warning_documented": self.recovery_warning_documented,
            "rotation_out_of_scope_for_sec02": self.rotation_out_of_scope_for_sec02,
        }


def get_key_policy_status() -> KeyPolicyStatus:
    """Return the binding key-policy status for SEC-02."""
    return KeyPolicyStatus()


def policy_ready() -> bool:
    """Return True if the SEC-02 key policy is satisfied at scaffold level."""
    s = get_key_policy_status()
    return all([
        s.no_hardcoded_key_in_code,
        s.operator_provided_key_required,
        not s.key_logged_in_reports,
        not s.key_committed_to_git,
        not s.key_in_environment_dump,
        s.confirm_twice_required_in_sec03,
        s.recovery_warning_documented,
        s.rotation_out_of_scope_for_sec02,
    ])


def operator_approval_checklist() -> List[str]:
    """Return the binding operator approval checklist for SEC-03 execution.

    SEC-02 only *generates* this checklist. SEC-03 (the future real
    migration block) is what consumes it.
    """
    return [
        "Operator confirms a verified backup of the main CKA store exists.",
        "Operator confirms the encryption key is stored in a password manager (not in code, env, or chat).",
        "Operator confirms the rollback plan has been read and understood.",
        "Operator confirms the real migration is approved by the responsible authority.",
        "Operator confirms no active Streamlit / pipeline process is using the database.",
        "Operator confirms this approval is for SEC-03 execution, NOT for SEC-02 planning.",
    ]
