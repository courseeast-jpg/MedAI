"""CKA-SEC-04 — runtime rollback plan.

This module produces the rollback *plan* the operator follows if the
encrypted runtime needs to be turned off. It does NOT perform any
destructive action automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RuntimeRollbackPlan:
    """Public-report-safe rollback plan."""

    steps: List[str] = field(default_factory=list)
    no_destructive_action_on_rollback: bool = True
    no_data_migration_in_sec04: bool = True
    do_not_delete_encrypted_store: bool = True
    do_not_delete_unencrypted_store: bool = True
    operator_verifies_ui_after_rollback: bool = True

    def __post_init__(self) -> None:
        if not self.steps:
            self.steps = [
                "1. Unset MEDAI_CKA_ENCRYPTED_STORE_ENABLED in the operator's environment.",
                "2. Optionally unset MEDAI_CKA_ENCRYPTION_KEY and MEDAI_CKA_ENCRYPTED_STORE_PATH.",
                "3. Restart the MedAI Streamlit app (Start_MedAI_UI.bat or python -m streamlit run app/main.py).",
                "4. Confirm the Clinical Knowledge Safety tab loads and all 9 panels render.",
                "5. Confirm runtime_encryption_active=False and external_api_used=False.",
                "6. Do NOT delete the encrypted store file. Do NOT delete the existing unencrypted store.",
                "7. Document the rollback in the operator-review log.",
            ]

    def safe_public_summary(self) -> dict:
        return {
            "steps_count": len(self.steps),
            "no_destructive_action_on_rollback": self.no_destructive_action_on_rollback,
            "no_data_migration_in_sec04": self.no_data_migration_in_sec04,
            "do_not_delete_encrypted_store": self.do_not_delete_encrypted_store,
            "do_not_delete_unencrypted_store": self.do_not_delete_unencrypted_store,
            "operator_verifies_ui_after_rollback": self.operator_verifies_ui_after_rollback,
        }


def get_runtime_rollback_plan() -> RuntimeRollbackPlan:
    return RuntimeRollbackPlan()


def rollback_plan_ready(plan: RuntimeRollbackPlan = None) -> bool:
    p = plan if plan is not None else get_runtime_rollback_plan()
    return all([
        len(p.steps) >= 5,
        p.no_destructive_action_on_rollback,
        p.no_data_migration_in_sec04,
        p.do_not_delete_encrypted_store,
        p.do_not_delete_unencrypted_store,
        p.operator_verifies_ui_after_rollback,
    ])
