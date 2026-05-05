"""Deterministic local medication safety gate for execution writes."""

from __future__ import annotations

from typing import Callable

from app.schemas import DDIFinding, MKBRecord


LOCAL_DDI_RULES = {
    frozenset({"lamotrigine", "valproate"}): {
        "severity": "HIGH",
        "management": "Do not combine without clinician review.",
    },
    frozenset({"warfarin", "ibuprofen"}): {
        "severity": "MEDIUM",
        "management": "Needs medication acknowledgment before write.",
    },
    frozenset({"sertraline", "ibuprofen"}): {
        "severity": "LOW",
        "management": "Low-severity interaction noted.",
    },
}


class MedicationSafetyGate:
    """Deterministic local gate with no external dependency."""

    def __init__(
        self,
        *,
        active_medications_provider: Callable[[], list[MKBRecord]] | None = None,
        available: bool = True,
        interaction_rules: dict | None = None,
    ):
        self.active_medications_provider = active_medications_provider or (lambda: [])
        self.available = available
        self.interaction_rules = interaction_rules or LOCAL_DDI_RULES

    def gate_medication_write(self, candidate: MKBRecord, session_id: str = ""):
        if candidate.fact_type != "medication":
            return "allow", "Not a medication record", []

        if not self.available:
            finding = {
                "drug_a": candidate.structured.get("name", candidate.content),
                "drug_b": "",
                "severity": "UNAVAILABLE",
                "management": "Medication safety check unavailable.",
            }
            candidate.ddi_checked = False
            candidate.ddi_status = "pending_ddi_check"
            candidate.ddi_findings = [finding]
            candidate.safety_action = "pending_ddi_check"
            candidate.structured["ddi_note"] = "Medication safety check unavailable."
            return "pending", "Medication safety check unavailable.", [finding]

        med_name = str(candidate.structured.get("name", candidate.content)).strip()
        findings = self._find_interactions(med_name)
        candidate.ddi_checked = True
        candidate.ddi_findings = [self._finding_to_dict(item) for item in findings]

        if not findings:
            candidate.ddi_status = "clear"
            candidate.safety_action = "allow"
            return "allow", "No interactions found.", []

        max_severity = self._max_severity(findings)
        if max_severity == "HIGH":
            candidate.ddi_status = "high_blocked"
            candidate.safety_action = "block_write"
            candidate.structured["ddi_note"] = "High-severity medication interaction."
            return "block", "High-severity interaction detected.", findings
        if max_severity == "MEDIUM":
            candidate.ddi_status = "medium"
            candidate.safety_action = "needs_review"
            candidate.structured["ddi_note"] = "Medium-severity medication interaction requires acknowledgment."
            return "review", "Medium-severity interaction requires acknowledgment.", findings

        candidate.ddi_status = "low"
        candidate.safety_action = "allow_with_note"
        candidate.structured["ddi_note"] = "Low-severity medication interaction noted."
        return "allow", "Low-severity interaction noted.", findings

    def _find_interactions(self, med_name: str) -> list[DDIFinding]:
        active_medications = self.active_medications_provider()
        candidate_key = med_name.lower()
        findings: list[DDIFinding] = []
        for active in active_medications:
            active_name = str(active.structured.get("name", active.content)).strip()
            rule = self.interaction_rules.get(frozenset({candidate_key, active_name.lower()}))
            if not rule:
                continue
            findings.append(DDIFinding(
                drug_a=med_name,
                drug_b=active_name,
                severity=rule["severity"],
                management=rule["management"],
            ))
        return findings

    def _max_severity(self, findings: list[DDIFinding]) -> str:
        for severity in ("HIGH", "MEDIUM", "LOW"):
            if any(item.severity == severity for item in findings):
                return severity
        return "NONE"

    def _finding_to_dict(self, finding):
        if hasattr(finding, "model_dump"):
            return finding.model_dump()
        return dict(finding)
