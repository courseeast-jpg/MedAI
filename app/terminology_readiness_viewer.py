"""Operator-facing terminology readiness viewer.

This module reads only public CKA terminology reports and renders a safe
readiness summary for the Streamlit UI. It never opens private license
acknowledgment files and never inspects terminology data contents.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).parent.parent

TERM_REPORTS: dict[str, tuple[str, Path]] = {
    "TERM-01": (
        "Readiness",
        PROJECT_ROOT
        / "reports"
        / "cka_term01_real_terminology_readiness"
        / "cka_term01_real_terminology_readiness_report.json",
    ),
    "TERM-01A": (
        "Intake automation",
        PROJECT_ROOT
        / "reports"
        / "cka_term01a_intake_automation"
        / "cka_term01a_intake_automation_report.json",
    ),
    "TERM-01B": (
        "Import planner",
        PROJECT_ROOT
        / "reports"
        / "cka_term01b_import_planner"
        / "cka_term01b_import_planner_report.json",
    ),
    "TERM-01C": (
        "Synthetic import executor",
        PROJECT_ROOT
        / "reports"
        / "cka_term01c_import_executor"
        / "cka_term01c_import_executor_report.json",
    ),
    "TERM-01D": (
        "Golden lookup QA",
        PROJECT_ROOT
        / "reports"
        / "cka_term01d_terminology_qa"
        / "cka_term01d_terminology_qa_report.json",
    ),
}

UNSAFE_KEYS = frozenset(
    {
        "replacement_map",
        "source_response_raw",
        "raw_source_text",
        "private_payload",
        "private_text",
        "license_text",
    }
)

SYSTEMS = ("umls", "snomed_ct", "rxnorm", "loinc")


@dataclass(frozen=True)
class TerminologyPhaseStatus:
    phase: str
    label: str
    status: str
    conclusion: str
    report_loaded: bool


@dataclass(frozen=True)
class TerminologyReadinessSummary:
    phase_statuses: list[TerminologyPhaseStatus]
    systems_missing: list[str]
    systems_with_files_present: list[str]
    systems_requiring_private_license_ack: list[str]
    systems_import_ready: list[str]
    real_import_not_run: bool
    no_external_apis_downloads: bool
    next_manual_action: str
    public_reports_only: bool
    private_ack_file_loaded: bool
    terminology_data_files_read: bool
    raw_paths_displayed: bool
    clinical_advice_generated: bool

    def safe_public_summary(self) -> dict[str, Any]:
        return {
            "phase_statuses": [
                {
                    "phase": item.phase,
                    "label": item.label,
                    "status": item.status,
                    "conclusion": item.conclusion,
                    "report_loaded": item.report_loaded,
                }
                for item in self.phase_statuses
            ],
            "systems_missing": list(self.systems_missing),
            "systems_with_files_present": list(self.systems_with_files_present),
            "systems_requiring_private_license_ack": list(self.systems_requiring_private_license_ack),
            "systems_import_ready": list(self.systems_import_ready),
            "real_import_not_run": self.real_import_not_run,
            "no_external_apis_downloads": self.no_external_apis_downloads,
            "next_manual_action": self.next_manual_action,
            "public_reports_only": self.public_reports_only,
            "private_ack_file_loaded": self.private_ack_file_loaded,
            "terminology_data_files_read": self.terminology_data_files_read,
            "raw_paths_displayed": self.raw_paths_displayed,
            "clinical_advice_generated": self.clinical_advice_generated,
        }


def _is_private_path(path: Path) -> bool:
    lowered = path.name.lower()
    return "_private" in lowered or "private_" in lowered or lowered == "license_ack_private.json"


def _strip_unsafe_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_unsafe_keys(v) for k, v in value.items() if k not in UNSAFE_KEYS}
    if isinstance(value, list):
        return [_strip_unsafe_keys(item) for item in value]
    return value


def load_public_report(path: Path) -> dict[str, Any] | None:
    """Load one public JSON report; private-looking paths are refused."""
    if _is_private_path(path) or not path.exists() or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return _strip_unsafe_keys(data)


def load_terminology_readiness_reports(
    report_paths: dict[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load only public terminology reports keyed by TERM phase."""
    reports: dict[str, dict[str, Any]] = {}
    paths = {phase: path for phase, (_label, path) in TERM_REPORTS.items()}
    if report_paths:
        paths.update(report_paths)
    for phase, path in paths.items():
        data = load_public_report(Path(path))
        if data is not None:
            reports[phase] = data
    return reports


def build_terminology_readiness_summary(
    reports: dict[str, dict[str, Any]] | None = None,
) -> TerminologyReadinessSummary:
    """Build a safe operator summary from loaded public reports."""
    loaded = reports if reports is not None else load_terminology_readiness_reports()
    statuses = [_phase_status(phase, loaded.get(phase)) for phase in TERM_REPORTS]
    missing: set[str] = set()
    files_present: set[str] = set()
    blocked_ack: set[str] = set()
    import_ready: set[str] = set()

    term01 = loaded.get("TERM-01", {})
    inventory = term01.get("inventory_summary", {}) if isinstance(term01, dict) else {}
    for source in inventory.get("sources", []) if isinstance(inventory, dict) else []:
        if not isinstance(source, dict):
            continue
        system = str(source.get("system") or "")
        if system not in SYSTEMS:
            continue
        file_count = int(source.get("file_count") or 0)
        status = str(source.get("status") or "")
        if file_count > 0:
            files_present.add(system)
        if status == "missing":
            missing.add(system)
        if file_count > 0 and not bool(source.get("license_confirmed", False)):
            blocked_ack.add(system)

    # Only top-level/current readiness fields should drive operator state.
    # Synthetic validation case internals may include "import_ready" scenarios
    # and must not be displayed as current readiness for real terminology files.
    term01b = loaded.get("TERM-01B", {})
    for system in term01b.get("systems_missing", []) if isinstance(term01b, dict) else []:
        if system in SYSTEMS:
            missing.add(system)
    for system in term01b.get("systems_blocked_license", []) if isinstance(term01b, dict) else []:
        if system in SYSTEMS:
            blocked_ack.add(system)
    for system in term01b.get("systems_import_ready", []) if isinstance(term01b, dict) else []:
        if system in SYSTEMS:
            import_ready.add(system)

    real_import_not_run = not any(bool(r.get("real_terminology_imported") or r.get("real_import_performed")) for r in loaded.values())
    no_external = not any(
        bool(r.get("external_api_used") or r.get("external_terminology_api_used"))
        for r in loaded.values()
    )
    clinical_advice = any(
        bool(r.get("clinical_recommendations_generated") or r.get("prescription_dosing_advice_generated"))
        for r in loaded.values()
    )

    return TerminologyReadinessSummary(
        phase_statuses=statuses,
        systems_missing=sorted(missing),
        systems_with_files_present=sorted(files_present),
        systems_requiring_private_license_ack=sorted(blocked_ack),
        systems_import_ready=sorted(import_ready),
        real_import_not_run=real_import_not_run,
        no_external_apis_downloads=no_external,
        next_manual_action=_next_manual_action(loaded),
        public_reports_only=True,
        private_ack_file_loaded=False,
        terminology_data_files_read=False,
        raw_paths_displayed=False,
        clinical_advice_generated=clinical_advice,
    )


def render_readiness_text(summary: TerminologyReadinessSummary) -> str:
    """Return non-empty safe text for tests, reports, and non-Streamlit display."""
    phase_lines = [
        f"- {item.phase} {item.label}: {item.status} ({item.conclusion})"
        for item in summary.phase_statuses
    ]
    return "\n".join(
        [
            "Terminology Readiness",
            *phase_lines,
            f"Systems missing: {_format_list(summary.systems_missing)}",
            f"Systems with files present: {_format_list(summary.systems_with_files_present)}",
            f"Systems requiring private license ack: {_format_list(summary.systems_requiring_private_license_ack)}",
            f"Systems import-ready: {_format_list(summary.systems_import_ready)}",
            "Warning: real import not run.",
            "Warning: no external APIs or downloads are used by this viewer.",
            f"Next manual action: {summary.next_manual_action}",
        ]
    )


def render_terminology_readiness_panel(summary: TerminologyReadinessSummary | None = None) -> None:
    """Render Streamlit panel. Import is local so tests can import without Streamlit."""
    import streamlit as st

    data = summary or build_terminology_readiness_summary()
    st.subheader("Terminology Readiness")
    st.caption("Public terminology readiness reports only. Real import remains blocked until licensed files and private acknowledgment are provided.")
    cols = st.columns(4)
    cols[0].metric("Missing systems", len(data.systems_missing))
    cols[1].metric("Files present", len(data.systems_with_files_present))
    cols[2].metric("Need license ack", len(data.systems_requiring_private_license_ack))
    cols[3].metric("Import-ready", len(data.systems_import_ready))

    st.warning("Real terminology import has not run. TERM-02 remains blocked until the operator provides licensed files and private license acknowledgment.")
    st.info("No external APIs or downloads are used by this readiness viewer.")

    st.markdown("**Phase status**")
    for item in data.phase_statuses:
        label = "ready" if item.status == "ready" else item.status
        st.write(f"{item.phase} - {item.label}: {label}")

    st.markdown("**Systems**")
    st.write(f"Missing: {_format_list(data.systems_missing)}")
    st.write(f"Files present: {_format_list(data.systems_with_files_present)}")
    st.write(f"Requiring private license ack: {_format_list(data.systems_requiring_private_license_ack)}")
    st.write(f"Import-ready: {_format_list(data.systems_import_ready)}")
    st.markdown(f"**Next manual action:** {data.next_manual_action}")


def _phase_status(phase: str, report: dict[str, Any] | None) -> TerminologyPhaseStatus:
    label = TERM_REPORTS[phase][0]
    if not report:
        return TerminologyPhaseStatus(
            phase=phase,
            label=label,
            status="missing",
            conclusion="public_report_missing",
            report_loaded=False,
        )
    conclusion = str(report.get("conclusion") or "present")
    status = "ready" if bool(report.get("all_passed", False)) or conclusion.endswith("_ready") else "present"
    return TerminologyPhaseStatus(
        phase=phase,
        label=label,
        status=status,
        conclusion=conclusion,
        report_loaded=True,
    )


def _next_manual_action(reports: dict[str, dict[str, Any]]) -> str:
    for phase in ("TERM-01D", "TERM-01C", "TERM-01B", "TERM-01A", "TERM-01"):
        report = reports.get(phase, {})
        action = report.get("next_manual_action") or report.get("next_recommended_action")
        if action:
            return str(action)
    return "operator downloads licensed files and creates private license ack"


def _format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"
