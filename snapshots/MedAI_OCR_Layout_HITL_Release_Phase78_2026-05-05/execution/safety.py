"""Thin safety wrapper around the existing medication DDI gate."""

from __future__ import annotations

from typing import Any

from app.schemas import MKBRecord
from execution.medication_safety_gate import MedicationSafetyGate


class ExecutionSafety:
    """Delegates medication write decisions to the existing DDI gate."""

    def __init__(
        self,
        medication_gate: Any | None = None,
        active_medications_provider=None,
    ):
        self.medication_gate = medication_gate or MedicationSafetyGate(
            active_medications_provider=active_medications_provider,
        )

    def check_medication(self, record: MKBRecord, session_id: str = "") -> tuple[str, str, list[Any]]:
        if record.fact_type != "medication" or self.medication_gate is None:
            return "allow", "", []
        return self.medication_gate.gate_medication_write(record, session_id=session_id)
