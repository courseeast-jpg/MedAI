"""CKA-B10 validation: System Preflight + Scaffold — 12 synthetic cases (A-L).

All cases use only local in-memory operations. No external API calls.
No clinical advice. No real patient data. No active writes.

Run:  python -m scripts.run_cka_block10_preflight_scaffold_validation
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Forbidden-phrase safety checker (same pattern as B09 validation)
# ---------------------------------------------------------------------------

# Tuple format: (positive_assertion_phrase, [safe_negation_contexts])
_FORBIDDEN_CLAIMS: List[Tuple[str, List[str]]] = [
    ("diagnoses patients", []),
    ("is a medical device", ["NOT a medical device", "is not a medical device"]),
    ("autonomous diagnosis", ["not autonomous", "no autonomous"]),
    ("prescribes medication", ["does NOT", "does not"]),
    ("issues medication orders", []),
    ("real patient data", ["no real patient", "not real patient"]),
    ("production-autonomous", ["NOT production-autonomous", "not production-autonomous"]),
    ("external api active", []),
    ("active write enabled: true", []),
]

_FORBIDDEN_ADVICE = [
    "take this dose",
    "recommended dose is",
    "you should take",
    "mg per day",
]


def _check_text_safe(text: str, context: str = "") -> List[str]:
    """Return list of safety violations found in text."""
    violations = []
    lo = text.lower()
    for phrase, negation_contexts in _FORBIDDEN_CLAIMS:
        if phrase.lower() in lo:
            safe = any(nc.lower() in lo for nc in negation_contexts)
            if not safe:
                violations.append(f"{context}: forbidden claim '{phrase}'")
    for phrase in _FORBIDDEN_ADVICE:
        if phrase.lower() in lo:
            violations.append(f"{context}: forbidden advice '{phrase}'")
    return violations


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ok(case: str, desc: str, details: dict | None = None) -> dict:
    return {"case": case, "description": desc, "passed": True, "details": details or {}}


def _fail(case: str, desc: str, error: str, details: dict | None = None) -> dict:
    return {"case": case, "description": desc, "passed": False, "error": error, "details": details or {}}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_a_all_modules_import() -> dict:
    """All CKA B01-B09 modules import without error."""
    modules = [
        "clinical_knowledge.models",
        "clinical_knowledge.store",
        "clinical_knowledge.privacy.sanitizer",
        "clinical_knowledge.privacy.outbound_audit",
        "clinical_knowledge.decision_engine.engine",
        "clinical_knowledge.decision_engine.safe_mode",
        "clinical_knowledge.truth_resolution.engine",
        "clinical_knowledge.truth_resolution.quarantine",
        "clinical_knowledge.medication_safety.models",
        "clinical_knowledge.medication_safety.ddi_stub",
        "clinical_knowledge.enrichment.models",
        "clinical_knowledge.enrichment.integration",
        "clinical_knowledge.medical_coding.models",
        "clinical_knowledge.medical_coding.synthetic_mapper",
        "clinical_knowledge.connectors.registry",
        "clinical_knowledge.consensus.engine",
        "app.clinical_knowledge_safety_viewer",
        "clinical_knowledge.preflight",
        "clinical_knowledge.scaffold",
    ]
    import importlib
    failed = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as exc:
            failed.append(f"{m}: {exc}")
    if failed:
        return _fail("A", "All CKA modules import", f"Import failures: {failed}")
    return _ok("A", "All CKA modules import", {"modules_checked": len(modules)})


def case_b_preflight_passes() -> dict:
    """Preflight runner returns overall_status=pass with no failures."""
    from clinical_knowledge.preflight import run_cka_preflight, PreflightStatus
    report = run_cka_preflight()
    if report.overall_status != PreflightStatus.PASS:
        failed = [c.name for c in report.checks if c.status == PreflightStatus.FAIL]
        warned = [c.name for c in report.checks if c.status == PreflightStatus.WARN]
        return _fail("B", "Preflight passes", f"Status={report.overall_status.value}", {
            "failed_checks": failed, "warned_checks": warned,
        })
    return _ok("B", "Preflight passes", {
        "checks_total": len(report.checks),
        "checks_passed": report.checks_passed,
    })


def case_c_external_api_blocked() -> dict:
    """Preflight confirms EXTERNAL_APIS_ENABLED=False."""
    from clinical_knowledge.preflight import run_cka_preflight, PreflightStatus
    report = run_cka_preflight()
    api_check = next(
        (c for c in report.checks if c.name == "external_api_blocked"),
        None,
    )
    if api_check is None:
        return _fail("C", "External API blocked confirmed", "Check 'external_api_blocked' not found in report.")
    if api_check.status != PreflightStatus.PASS:
        return _fail("C", "External API blocked confirmed", f"Check status={api_check.status.value}")
    if not report.external_api_blocked:
        return _fail("C", "External API blocked confirmed", "report.external_api_blocked is False")
    return _ok("C", "External API blocked confirmed", {"external_api_blocked": report.external_api_blocked})


def case_d_hitl_freeze_present() -> dict:
    """Preflight confirms HITL freeze document is present."""
    from clinical_knowledge.preflight import run_cka_preflight, PreflightStatus
    report = run_cka_preflight()
    hitl_check = next(
        (c for c in report.checks if c.name == "hitl_freeze_present"),
        None,
    )
    if hitl_check is None:
        return _fail("D", "HITL freeze document present", "Check 'hitl_freeze_present' not found.")
    if hitl_check.status != PreflightStatus.PASS:
        return _fail("D", "HITL freeze document present", f"Check status={hitl_check.status.value}")
    if not report.hitl_freeze_confirmed:
        return _fail("D", "HITL freeze document present", "report.hitl_freeze_confirmed is False")
    return _ok("D", "HITL freeze document present", {"hitl_freeze_confirmed": report.hitl_freeze_confirmed})


def case_e_mkb_store_ledger_functional() -> dict:
    """MKBStore instantiates in-memory and ledger write succeeds."""
    from clinical_knowledge.store import MKBStore
    from clinical_knowledge.models import (
        MKBRecord, TrustLevel, SourceType, LedgerEvent, LedgerEventType,
    )
    from clinical_knowledge.ledger import make_created_event
    from clinical_knowledge.safe_ids import new_record_id, make_safe_record_id
    store = MKBStore(":memory:")
    rec_id = new_record_id()
    safe_id = make_safe_record_id(rec_id)
    record = MKBRecord(
        record_id=rec_id,
        safe_record_id=safe_id,
        session_id="preflight_test",
        fact_type="symptom",
        entity_text="fatigue",
        trust_level=TrustLevel.MODEL_SUGGESTED,
        source_type=SourceType.SYNTHETIC,
    )
    event = make_created_event(
        record_id=rec_id,
        safe_record_id=safe_id,
        tier=record.tier.value,
        trust_level=record.trust_level.value,
    )
    store.insert_record(record, event)
    fetched = store.fetch_by_record_id(record.record_id)
    if fetched is None:
        return _fail("E", "MKBStore + ledger functional", "Record not found after insert.")
    ledger_count = store.count_ledger_events()
    return _ok("E", "MKBStore + ledger functional", {
        "insert_ok": True,
        "fetch_ok": True,
        "ledger_events": ledger_count,
    })


def case_f_privacy_gate_blocks_secret() -> dict:
    """B02 privacy gate detects and blocks SECRET-category content."""
    from clinical_knowledge.privacy.sanitizer import sanitize_text
    from clinical_knowledge.privacy.patterns import ALWAYS_BLOCK_CATEGORIES
    if "SECRET" not in ALWAYS_BLOCK_CATEGORIES:
        return _fail("F", "Privacy gate blocks SECRET", "SECRET not in ALWAYS_BLOCK_CATEGORIES.")
    # Verify a secret-like string is detected
    result = sanitize_text("api_key=sk-abc123def456ghi789jkl012mno345p")
    if not result.findings:
        return _fail("F", "Privacy gate blocks SECRET", "No findings for secret-pattern text.")
    secret_found = any(f.category == "SECRET" for f in result.findings)
    if not secret_found:
        return _fail("F", "Privacy gate blocks SECRET", "SECRET category not detected.")
    return _ok("F", "Privacy gate blocks SECRET", {
        "secret_detected": True,
        "findings_count": len(result.findings),
    })


def case_g_registry_rejects_external() -> dict:
    """ConnectorRegistry rejects allow_external=True specs."""
    from clinical_knowledge.connectors.registry import ConnectorRegistry, ConnectorRegistryError
    from clinical_knowledge.connectors.models import ConnectorSpec, ConnectorKind, ConnectorCapability
    registry = ConnectorRegistry()
    bad_spec = ConnectorSpec(
        name="bad_external",
        kind=ConnectorKind.DXGPT_STUB,
        enabled=True,
        capabilities=[ConnectorCapability.DIAGNOSIS_SUPPORT],
        allow_external=True,
        synthetic_only=True,
    )
    try:
        registry.register(bad_spec)
        return _fail("G", "Registry rejects external=True", "No error raised for allow_external=True.")
    except ConnectorRegistryError as exc:
        return _ok("G", "Registry rejects external=True", {"error_type": "ConnectorRegistryError"})


def case_h_consensus_engine_instantiable() -> dict:
    """Consensus engine can be imported and run_consensus is callable."""
    from clinical_knowledge.consensus.engine import run_consensus
    from clinical_knowledge.store import MKBStore
    store = MKBStore(":memory:")
    # Empty connector results → ALL_RESPONSES_DISCARDED
    from clinical_knowledge.consensus.models import ConsensusStatus
    result = run_consensus([], store=store, min_confidence=0.4, safe_mode=True)
    if result.status != ConsensusStatus.ALL_RESPONSES_DISCARDED:
        return _fail("H", "Consensus engine instantiable", f"Unexpected status={result.status.value}")
    return _ok("H", "Consensus engine instantiable", {"status": result.status.value})


def case_i_operator_ui_loads_public_only() -> dict:
    """Operator UI viewer loads public reports and confirms no private files read."""
    from app.clinical_knowledge_safety_viewer import load_cka_safety_snapshot
    snapshot = load_cka_safety_snapshot()
    if snapshot.get("private_files_read") is not False:
        return _fail("I", "Operator UI loads public only", "private_files_read is not False.")
    if snapshot.get("replacement_map_loaded") is not False:
        return _fail("I", "Operator UI loads public only", "replacement_map_loaded is not False.")
    return _ok("I", "Operator UI loads public only", {
        "private_files_read": False,
        "replacement_map_loaded": False,
        "blocks_loaded": snapshot.get("blocks_loaded", []),
    })


def case_j_scaffold_builds_safe_defaults() -> dict:
    """CKASystemScaffold.build() succeeds with safe defaults."""
    from clinical_knowledge.scaffold import CKASystemScaffold
    s = CKASystemScaffold.build()
    summary = s.safe_public_summary()
    checks = [
        ("safe_mode", summary.get("safe_mode") is True),
        ("allow_active_write", summary.get("allow_active_write") is False),
        ("production_autonomous", summary.get("production_autonomous") is False),
        ("external_api_used", summary.get("external_api_used") is False),
        ("config_external_apis_enabled", summary.get("config_external_apis_enabled") is False),
        ("store_initialized", summary.get("store_initialized") is True),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        return _fail("J", "Scaffold builds with safe defaults", f"Failed checks: {failed}", summary)
    return _ok("J", "Scaffold builds with safe defaults", summary)


def case_k_scaffold_is_ready() -> dict:
    """CKASystemScaffold.is_ready() returns True after successful preflight."""
    from clinical_knowledge.scaffold import CKASystemScaffold
    s = CKASystemScaffold.build()
    if not s.is_ready():
        pf = s._last_preflight
        failed = [c.name for c in pf.checks if c.status.value == "fail"] if pf else []
        return _fail("K", "Scaffold is_ready returns True", f"is_ready()=False. Failed: {failed}")
    return _ok("K", "Scaffold is_ready returns True", {
        "is_ready": True,
        "preflight_passed": s._last_preflight.passed if s._last_preflight else None,
    })


def case_l_scaffold_allow_active_write_raises() -> dict:
    """CKASystemScaffold raises ValueError when allow_active_write=True."""
    from clinical_knowledge.scaffold import CKASystemScaffold
    from clinical_knowledge.store import MKBStore
    from clinical_knowledge.connectors.registry import ConnectorRegistry
    from clinical_knowledge.config import CKAConfig
    try:
        CKASystemScaffold(
            store=MKBStore(":memory:"),
            registry=ConnectorRegistry.default(),
            config=CKAConfig(),
            allow_active_write=True,
        )
        return _fail("L", "Scaffold raises on allow_active_write=True", "No ValueError raised.")
    except ValueError:
        return _ok("L", "Scaffold raises on allow_active_write=True", {"error_type": "ValueError"})


# ---------------------------------------------------------------------------
# Public report + privacy check
# ---------------------------------------------------------------------------

def _build_report(results: List[dict]) -> dict:
    from clinical_knowledge.preflight import run_cka_preflight
    report = run_cka_preflight()
    summary = report.safe_public_summary()
    return {
        "block_id": "CKA-B10",
        "conclusion": "cka_b10_preflight_scaffold_ready",
        "synthetic_cases_run": len(results),
        "cases_passed": sum(1 for r in results if r["passed"]),
        "all_passed": all(r["passed"] for r in results),
        "case_results": results,
        "preflight_summary": summary,
        "external_api_used": False,
        "production_autonomous": False,
        "allow_active_write": False,
        "hitl_freeze_confirmed": report.hitl_freeze_confirmed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_reports(report: dict) -> None:
    out_dir = Path(__file__).parent.parent / "reports" / "cka_block10_preflight_scaffold"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "cka_block10_preflight_scaffold_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# CKA-B10 Preflight + Scaffold Validation Report",
        "",
        f"- block_id: {report['block_id']}",
        f"- cases_run: {report['synthetic_cases_run']}",
        f"- cases_passed: {report['cases_passed']}",
        f"- all_passed: {report['all_passed']}",
        f"- external_api_used: {report['external_api_used']}",
        f"- production_autonomous: {report['production_autonomous']}",
        f"- allow_active_write: {report['allow_active_write']}",
        f"- hitl_freeze_confirmed: {report['hitl_freeze_confirmed']}",
        "",
        "## Preflight Summary",
        "",
    ]
    pf = report["preflight_summary"]
    md_lines += [
        f"- overall_status: {pf['overall_status']}",
        f"- checks_total: {pf['checks_total']}",
        f"- checks_passed: {pf['checks_passed']}",
        f"- checks_failed: {pf['checks_failed']}",
        f"- checks_warned: {pf['checks_warned']}",
        "",
        "## Case Results",
        "",
    ]
    for r in report["case_results"]:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        md_lines.append(f"- Case {r['case']}: {status} {r['description']}")
        if not r["passed"]:
            md_lines.append(f"  Error: {r.get('error', 'unknown')}")
    md_lines.append("")

    md_path = out_dir / "cka_block10_preflight_scaffold_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def _check_report_privacy(report: dict) -> None:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload
    result = check_public_report_payload(report)
    if not result.passed:
        raise RuntimeError(
            f"Privacy check failed on B10 report: {result.leak_examples_redacted}"
        )


def run_validation() -> dict:
    cases = [
        case_a_all_modules_import,
        case_b_preflight_passes,
        case_c_external_api_blocked,
        case_d_hitl_freeze_present,
        case_e_mkb_store_ledger_functional,
        case_f_privacy_gate_blocks_secret,
        case_g_registry_rejects_external,
        case_h_consensus_engine_instantiable,
        case_i_operator_ui_loads_public_only,
        case_j_scaffold_builds_safe_defaults,
        case_k_scaffold_is_ready,
        case_l_scaffold_allow_active_write_raises,
    ]
    results = []
    for fn in cases:
        try:
            results.append(fn())
        except Exception as exc:
            label = fn.__name__.split("_")[1].upper()
            results.append(_fail(label, fn.__doc__ or fn.__name__, f"Unexpected exception: {exc}"))

    report = _build_report(results)

    # Safety text check on all case descriptions
    for r in results:
        violations = _check_text_safe(r["description"], f"Case {r['case']}")
        if violations:
            r["passed"] = False
            r["error"] = f"Safety text violations: {violations}"
            report["all_passed"] = False

    _check_report_privacy(report)
    _write_reports(report)
    return report


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report = run_validation()
    status = "[PASS]" if report["all_passed"] else "[FAIL]"
    print(f"\nCKA-B10 Preflight + Scaffold Validation — {status}")
    print(f"  Cases: {report['cases_passed']}/{report['synthetic_cases_run']} passed")
    for r in report["case_results"]:
        marker = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"  Case {r['case']}: {marker} {r['description']}")
        if not r["passed"]:
            print(f"         Error: {r.get('error')}")
    pf = report["preflight_summary"]
    print(f"\n  Preflight: {pf['overall_status']} "
          f"({pf['checks_passed']}/{pf['checks_total']} checks passed)")
    print(f"  HITL freeze confirmed: {report['hitl_freeze_confirmed']}")
    print(f"  External API used: {report['external_api_used']}")
    if not report["all_passed"]:
        sys.exit(1)
